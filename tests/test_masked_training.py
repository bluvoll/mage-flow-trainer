"""Painted loss masks: crop alignment, dataset plumbing, loss parity and config/GUI wiring."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from trainer.data.cache import cache_path, load_and_crop, parse_cache_filename
from trainer.data.cache_provenance import atomic_save
from trainer.data.dataset import DatasetConfig, MageFlowDataset, SubsetConfig, collate
from trainer.data.masks import feather_mask, find_masks, grow_mask, load_mask, loss_weights
from trainer.data.packed import collate_native
from trainer.training.flow import flow_loss, hf_loss


def _subject(size, box):
    """Black frame with one white rectangle -- as an image and as its own mask."""
    a = np.zeros((size[1], size[0]), np.uint8)
    l, t, r, b = box
    a[t:b, l:r] = 255
    return Image.fromarray(np.stack([a] * 3, -1)), Image.fromarray(a, mode="L")


def _folder(tmp_path, specs, name="images"):
    images, masks = tmp_path / name, tmp_path / f"{name}_masks"
    images.mkdir()
    masks.mkdir()
    for stem, size, box in specs:
        img, mask = _subject(size, box)
        img.save(images / f"{stem}.png")
        mask.save(masks / f"{stem}.png")
        (images / f"{stem}.txt").write_text("subject", encoding="utf-8")
    return images, masks


def _coverage(pixels):
    """(3,1,H,W) pixels of a black/white image -> its white fraction per 16x16 latent cell."""
    lum = (pixels[0, 0] + 1) / 2
    h, w = lum.shape
    return lum.view(h // 16, 16, w // 16, 16).mean(dim=(1, 3)).unsqueeze(0)


# Sources chosen so the bucket crops without resampling: 320x200 -> 320x192 drops 4 rows, top
# first; 336x192 -> 320x192 drops 8 columns, left first. An uncropped or wrongly-cropped mask is
# then off by a quarter or half of a latent cell, which the exact comparison below catches.
SPECS = [("tall", (320, 200), (40, 36, 200, 100)), ("wide", (336, 192), (56, 20, 136, 180))]


def test_mask_follows_the_image_crop_exactly(tmp_path):
    images, masks = _folder(tmp_path, SPECS)
    ds = MageFlowDataset(DatasetConfig(path=str(images), mask_path=str(masks), resolution=256,
                                       source="encode", mask_outside_weight=.25))
    assert ds.masked and len(ds) == 2
    for i in range(len(ds)):
        sample = ds[i]
        assert sample["pixels"].shape[-2:] == (192, 320) and sample["mask"].shape == (1, 12, 20)
        torch.testing.assert_close(sample["mask"], loss_weights(_coverage(sample["pixels"]), .25),
                                   rtol=0, atol=1e-6)


def test_resampled_crop_and_lower_resolution_mask_line_up(tmp_path):
    image, mask = _subject((900, 500), (300, 120, 620, 420))
    path = tmp_path / "a.png"
    image.save(path)
    mask.resize((450, 250), Image.NEAREST).save(tmp_path / "half.png")
    mgr = DatasetConfig(path=str(tmp_path), resolution=256).build_bucket_manager(256)
    crop, meta = load_and_crop(path, mgr)
    bucket, resized, _ = mgr.select_bucket(900, 500)
    assert bucket == meta["bucket"] and resized != (900, 500)
    coverage = load_mask(tmp_path / "half.png", (900, 500), bucket, resized)
    pixels = torch.from_numpy(np.array(crop, np.float32) / 127.5 - 1).permute(2, 0, 1).unsqueeze(1)
    torch.testing.assert_close(coverage, _coverage(pixels), rtol=0, atol=.08)
    mask.resize((450, 300)).save(tmp_path / "other_frame.png")
    with pytest.raises(ValueError, match="same aspect ratio"):
        load_mask(tmp_path / "other_frame.png", (900, 500), bucket, resized)


def test_latents_only_masks_match_images_mode(tmp_path):
    images, masks = _folder(tmp_path, SPECS)
    cfg = dict(mask_path=str(masks), resolution=256, mask_outside_weight=.1, mask_dilate_px=3,
               mask_feather_px=1.5)
    live = MageFlowDataset(DatasetConfig(path=str(images), source="encode", **cfg))
    expected = {e.path.stem: e.mask for e in live.entries}
    for e in live.entries:
        _, meta = load_and_crop(e.path, live.bucket_managers[e.resolution])
        w, h = e.bucket
        atomic_save({"latents": torch.zeros(128, 1, h // 16, w // 16)}, cache_path(e.path, e.bucket), {
            "bucket": f"{w}x{h}", "original_size": "{}x{}".format(*meta["original_size"]),
            "crop_ltrb": ",".join(map(str, meta["crop_ltrb"]))})
        e.path.unlink()
    cached = MageFlowDataset(DatasetConfig(path=str(images), source="latents", **cfg))
    assert {parse_cache_filename(e.latent_path)[0] for e in cached.entries} == set(expected)
    for i, e in enumerate(cached.entries):
        stem = parse_cache_filename(e.latent_path)[0]
        torch.testing.assert_close(cached[i]["mask"], expected[stem], rtol=0, atol=0)
        assert cached[i]["latents"].shape[-2:] == e.mask.shape[-2:]


def test_missing_duplicate_and_empty_masks(tmp_path):
    images, masks = _folder(tmp_path, SPECS)
    (masks / "wide.png").unlink()
    with pytest.raises(ValueError, match=r"no mask for 1 image.*'wide'"):
        MageFlowDataset(DatasetConfig(path=str(images), mask_path=str(masks), source="encode"))
    _subject((336, 192), (0, 0, 0, 0))[1].save(masks / "wide.png")
    with pytest.warns(UserWarning, match="entirely black"):
        MageFlowDataset(DatasetConfig(path=str(images), mask_path=str(masks), source="encode"))
    _subject((336, 192), (0, 0, 9, 9))[1].save(masks / "wide.jpg")
    with pytest.raises(ValueError, match="share the stem 'wide'"):
        find_masks(masks)


def test_unmasked_subsets_collate_and_toggle(tmp_path):
    images, masks = _folder(tmp_path, SPECS)
    plain, _ = _folder(tmp_path, [("reg", (320, 192), (0, 0, 1, 1))], name="reg")
    subsets = [SubsetConfig(path=str(images), mask_path=str(masks)), SubsetConfig(path=str(plain))]
    ds = MageFlowDataset(DatasetConfig(subsets=subsets, resolution=256, source="encode"))
    reg = next(i for i, e in enumerate(ds.entries) if e.path.stem == "reg")
    torch.testing.assert_close(ds[reg]["mask"], torch.ones(1, 12, 20))
    batch = collate([ds[i] for i in range(len(ds)) if ds.entries[i].bucket == (320, 192)])
    assert batch["mask"].shape == (3, 1, 12, 20)
    packed = collate_native([ds[i] for i in range(len(ds))])
    assert [m.shape for m in packed["mask"]] == [(1, 1, 12, 20)] * 3
    off = MageFlowDataset(DatasetConfig(subsets=subsets, resolution=256, source="encode",
                                        use_masks=False))
    assert not off.masked and "mask" not in off[0] and all(e.mask is None for e in off.entries)


def test_grow_feather_and_weights():
    dot = np.zeros((9, 9), np.float32)
    dot[4, 4] = 1
    assert grow_mask(dot, 2).sum() == 25 and grow_mask(grow_mask(dot, 2), -2).sum() == 1
    soft = feather_mask(grow_mask(dot, 2), 1.5)
    assert 0 < soft.min() < soft.max() < 1 and soft[4, 4] == soft.max()
    torch.testing.assert_close(loss_weights(torch.tensor([0., .5, 1.]), .2), torch.tensor([.2, .6, 1.]))


@pytest.mark.parametrize("packed", [False, True])
def test_trainer_step_weights_flow_and_hf_loss(monkeypatch, packed):
    from trainer.training.config import Config
    from trainer.training.flow_rng import FlowRNG
    from trainer.training.train import Trainer
    import trainer.training.train as module

    cfg = Config(dataset=DatasetConfig(path="/unused", mask_path="/unused_masks"))
    cfg.train.pack_resolutions = packed
    cfg.flow.hf_scale = .2
    trainer = Trainer.__new__(Trainer)
    trainer.flow_rng = FlowRNG(42, 0, 1, "cpu")
    trainer.cfg = cfg
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer.dtype = torch.float32
    trainer.phase = trainer.self_flow_ema = trainer.rti_interface = None
    trainer.adversarial = trainer.perceptual = trainer.paired_texture = None
    trainer.hf_patch = 1
    trainer._encode = lambda captions: None
    trainer._preserve_term = lambda: 0.
    weight = torch.tensor(.7, requires_grad=True)
    trainer.transformer = lambda hidden_states, **kw: (
        [x * weight for x in hidden_states] if isinstance(hidden_states, list) else hidden_states * weight,)

    def prepare(x, *args, **kwargs):
        return x * .5, x.new_full((x.shape[0],), .6), -x, None, None
    monkeypatch.setattr(module, "prepare_training_flow_batch", prepare)

    xs = [torch.randn(1, 4, 1, 8, 10), torch.randn(1, 4, 1, 12 if packed else 8, 10)]
    ms = [torch.rand(1, x.shape[-2], x.shape[-1]) for x in xs]
    ms[0][..., :5] = 0
    batch = {"latents": xs if packed else torch.cat(xs), "captions": ["a", "b"], "paths": ["a", "b"],
             "mask": [m.unsqueeze(0) for m in ms] if packed else torch.stack(ms)}
    actual, hf = trainer._step(batch)

    if packed:
        terms = []
        for x, m in zip(xs, ms):
            n, t, target, _, _ = prepare(x)
            pred, m5 = n * weight, m.view(1, 1, 1, *m.shape[-2:])
            terms.append(flow_loss(pred, target, mask=m5) + .2 * hf_loss(
                pred, n, x, t, 1, cfg.flow.hf_exponent, mask=m5, **cfg.flow.hf_gate_kwargs()))
        expected = torch.stack(terms).mean()
    else:
        x, m5 = torch.cat(xs), torch.stack(ms).unsqueeze(2)
        n, t, target, _, _ = prepare(x)
        pred = n * weight
        expected = flow_loss(pred, target, mask=m5) + .2 * hf_loss(
            pred, n, x, t, 1, cfg.flow.hf_exponent, mask=m5, **cfg.flow.hf_gate_kwargs())
        manual = ((pred - target) ** 2 * m5).sum() / (m5.expand_as(pred).sum())
        torch.testing.assert_close(flow_loss(pred, target, mask=m5), manual)
    torch.testing.assert_close(actual, expected)
    unmasked, _ = trainer._step({k: v for k, v in batch.items() if k != "mask"})
    assert not torch.isclose(actual, unmasked)
    grad = torch.autograd.grad(actual, weight)[0]
    assert torch.isfinite(grad) and grad != 0


def test_config_guards():
    from trainer.training.config import Config, validate_model_options
    from trainer.training.curriculum import Phase
    from trainer.training.paired_texture import PairedTextureConfig

    with pytest.raises(ValueError, match="mask_path on each subset"):
        DatasetConfig(subsets=[SubsetConfig(path="/a")], mask_path="/m")
    for bad in ({"mask_outside_weight": 1.5}, {"mask_feather_px": -1.}, {"mask_dilate_px": 1.5}):
        with pytest.raises(ValueError, match="dataset.mask_"):
            DatasetConfig(path="/a", **bad)
    assert not DatasetConfig(path="/a").masked and not DatasetConfig(path="/a", mask_path="").masked
    assert not DatasetConfig(path="/a", mask_path="/m", use_masks=False).masked

    def check(setup, match):
        cfg = Config(dataset=DatasetConfig(path="/a", mask_path="/m"))
        setup(cfg)
        with pytest.raises(ValueError, match=match):
            validate_model_options(cfg)
    check(lambda c: setattr(c, "paired_texture", PairedTextureConfig(scale=.1)), "paired_texture")
    check(lambda c: setattr(c.self_flow, "enabled", True), "self_flow")
    check(lambda c: setattr(c.perceptual, "enabled", True), "perceptual")
    check(lambda c: c.curriculum.phases.append(Phase(mode="texture")), "texture crops")

    cfg = Config(dataset=DatasetConfig(subsets=[SubsetConfig(path="/a", mask_path="/m", texture=False),
                                                SubsetConfig(path="/b")]))
    cfg.curriculum.phases.append(Phase(mode="texture"))
    try:
        validate_model_options(cfg)
    except Exception as exc:  # later model checks may still object; just not to the masks
        assert "mask" not in str(exc)


def test_gui_mask_controls_roundtrip(tmp_path, monkeypatch):
    from PySide6 import QtWidgets
    from trainer.gui import bridge
    from trainer.gui.app import TrainingGUI
    from trainer.training.config import load_config

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(TrainingGUI, "_load_presets", lambda self: None)
    gui = TrainingGUI()
    try:
        gui._apply(bridge.defaults() | {"dataset.path": "/images"})
        knobs = ("mask_outside_weight", "mask_dilate_px", "mask_feather_px")
        assert not any(gui.editors[f"dataset.{k}"].widget.isEnabled() for k in knobs)
        gui.editors["dataset.mask_path"].set("/images_masks")
        gui.editors["dataset.mask_outside_weight"].set(.2)
        gui._on_edit()
        assert all(gui.editors[f"dataset.{k}"].widget.isEnabled() for k in knobs)
        path = tmp_path / "masked.toml"
        bridge.write_toml(path, bridge.prune_defaults(gui.collect()))
        cfg = load_config(path)
        assert cfg.dataset.mask_path == "/images_masks" and cfg.dataset.mask_outside_weight == .2

        subsets = gui.editors["dataset.subsets"]
        subsets._add_row()
        assert not gui.editors["dataset.mask_path"].widget.isEnabled()
        subsets.table.cellWidget(0, 0).findChild(QtWidgets.QLineEdit).setText("/a")
        subsets.table.cellWidget(0, 3).findChild(QtWidgets.QLineEdit).setText("/a_masks")
        flat = gui.collect()
        assert flat["dataset.mask_path"] is None
        assert flat["dataset.subsets"] == [{"path": "/a", "texture": False, "mask_path": "/a_masks"}]
        assert bridge.validate(flat) == (True, "")
        bridge.write_toml(path, bridge.prune_defaults(flat))
        cfg = load_config(path)
        assert cfg.dataset.subsets[0].mask_path == "/a_masks" and cfg.dataset.mask_path is None
        gui._apply(bridge.defaults() | bridge.flatten(bridge.read_toml(path)))
        assert subsets.table.cellWidget(0, 3).findChild(QtWidgets.QLineEdit).text() == "/a_masks"
        gui.editors["dataset.use_masks"].set(False)
        gui._on_edit()
        assert not any(gui.editors[f"dataset.{k}"].widget.isEnabled() for k in knobs)
    finally:
        gui.close()
        gui.deleteLater()
        app.processEvents()
