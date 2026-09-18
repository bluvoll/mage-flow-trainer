import copy

import pytest
import torch

from trainer.modeling.mage_flow import MageFlow, MageFlowParams
from trainer.modeling.region_tokens import BudgetSchedule, RegionInterface, build_region_plan, gilbert_order


def test_gilbert_rectangles_and_mandatory_cut_floor():
    for h, w in ((84, 48), (48, 84), (33, 45), (84, 33), (1, 1)):
        order, jumps, floor = gilbert_order(h, w)
        assert sorted(order) == list(range(h * w))
        assert floor == sum(jumps) + 1
        plan = build_region_plan(torch.randn(h * w, 4), [(h, w)], [h * w], .01, torch.device("cpu"))
        assert plan.region_lengths == [max(floor, max(1, int(h * w * .01 + .5)))]
        assert plan.counts.sum().item() == h * w


def test_flat_plan_read_write_identity_and_gradients():
    torch.manual_seed(3)
    dense = torch.randn(1, 23, 8, requires_grad=True)
    freqs = torch.polar(torch.ones(23, 4), torch.randn(23, 4))
    plan = build_region_plan(dense[0].detach(), [(3, 5), (2, 4)], [15, 8], .5, dense.device)
    interface = RegionInterface(8, core_start=1, core_end=3)
    region_in, region_freqs = interface.read(dense, freqs, plan)
    assert region_in.shape[1] == sum(plan.region_lengths)
    assert region_freqs.shape[0] == sum(plan.region_lengths)
    restored = interface.write(dense, region_in, region_in, plan)
    torch.testing.assert_close(restored, dense)
    interface.write(dense, region_in * 1.1, region_in, plan).square().mean().backward()
    assert interface.write_map.weight.grad.abs().sum() > 0
    assert interface.read_score.weight.grad.abs().sum() > 0


def test_budget_schedule_is_stateless_and_snaps_deterministically():
    schedule = BudgetSchedule(identity_steps=2, warmup_steps=3, anneal_steps=10, budget_steps=(1.0, .75, .5), target_keep=.5)
    assert schedule.resolve(0) == (1.0, "identity")
    assert schedule.resolve(2) == (1.0, "warmup")  # .98 snaps to the nearest grid value.
    assert schedule.resolve(100) == (.5, "target")
    assert schedule.resolve(7) == schedule.resolve(7)


def test_packed_model_restores_dense_shape_through_rti_cpu_block_seam():
    model = MageFlow(MageFlowParams(128, 128, 24, 32, 4, 4, [2, 2, 4], False))
    model.configure_rti(1, 1, 17)
    model.configure_execution(False, attention_backend="torch_varlen")
    # The varlen kernel is CUDA-only; this verifies the production control flow without changing it.
    model.block_forward = lambda *args: (args[2], args[1])
    images = [torch.randn(1, 128, 1, 3, 5), torch.randn(1, 128, 1, 2, 4)]
    context = (torch.randn(2, 5, 24), torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool))
    output = model(images, torch.tensor([.2, .7]), context, keep_fraction=.5)[0]
    assert [x.shape for x in output] == [x.shape for x in images]
    restored = MageFlow(copy.deepcopy(model.params))
    restored.load_state_dict(model.state_dict())


def test_rti_rejects_invalid_spans_and_dual_conditioning():
    model = MageFlow(MageFlowParams(128, 128, 24, 32, 4, 4, [2, 2, 4], False))
    with pytest.raises(ValueError):
        model.configure_rti(0, 0)
    model.configure_rti(1, 1)
    model.configure_execution(False, attention_backend="torch_varlen")
    image = [torch.randn(1, 128, 1, 2, 2)]
    context = (torch.randn(1, 2, 24), torch.ones(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="cannot be combined"):
        model.forward_packed(image, torch.tensor([.5]), context, second_timestep=torch.tensor([.4]), timestep_mask=[torch.zeros(1, 2, 2, dtype=torch.bool)])
