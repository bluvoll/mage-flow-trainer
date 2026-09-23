"""Experimental Mage-Flow Self-Flow teacher and native-resolution forwards."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from ..modeling.batched import _double_stream_block_forward
from ..modeling.mageflow_attention import packed_attention_metadata

class BlockCall(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, *args):
        return _double_stream_block_forward(self.block, *args)


def fingerprint_indices(numel, device):
    count = min(32, numel)
    # FP32 linspace can round numel-1 up to numel on large model matrices.
    return torch.arange(count, device=device, dtype=torch.long) * max(numel-1, 0) // max(count-1, 1)


class AdapterEMA:
    def __init__(self, model, depth, device, decay, dtype=torch.float32, stochastic_rounding=False,
                 full_finetune=False, adaln_fp32=False):
        self.decay = decay
        self.stochastic_rounding = stochastic_rounding
        self.shadow = []
        self.params = []
        from trainer.training.params import classify
        def storage_dtype(name):
            return torch.float32 if adaln_fp32 and classify(name) == 'adaln' else dtype
        for index, block in enumerate(model.transformer_blocks[:depth]):
            params = {n: p for n, p in block.named_parameters() if p.requires_grad}
            if full_finetune:
                params = dict(block.named_parameters())
            elif not params or any('lycoris_adapter.' not in n for n in params):
                raise ValueError('Teacher EMA must contain only adapter parameters')
            self.params.append(params)
            self.shadow.append({n: self.dense(p, storage_dtype(f'transformer_blocks.{index}.{n}')).to(device=device, copy=True)
                                for n, p in params.items()})
        self.input_indices = {}
        if full_finetune:
            for name in ('img_in', 'txt_norm', 'txt_in', 'time_text_embed', 'modulation_down'):
                module = getattr(model, name, None)
                if module is None:
                    continue
                self.input_indices[name] = len(self.params)
                params = dict(module.named_parameters())
                self.params.append(params)
                self.shadow.append({n: self.dense(p, storage_dtype(f'{name}.{n}')).to(device=device, copy=True)
                                    for n, p in params.items()})

    @staticmethod
    @torch.no_grad()
    def dense(param, dtype):
        # .to() on SDNQTensor preserves quantized storage; EMA needs ordinary floats.
        from sdnq.training.tensor import SDNQTensor
        return (param.dequantize(dtype) if isinstance(param, SDNQTensor)
                else param.detach().to(dtype=dtype)).detach()

    @torch.no_grad()
    def forward_input(self, name, module, *args):
        index = self.input_indices.get(name)
        if index is None:
            return module(*args)
        state = {n: v.to(device=self.params[index][n].device, dtype=self.params[index][n].dtype)
                 for n, v in self.shadow[index].items()}
        return torch.func.functional_call(module, state, args, strict=True)

    @property
    def nbytes(self):
        return sum(v.numel() * v.element_size() for group in self.shadow for v in group.values())

    @torch.no_grad()
    def forward_block(self, index, block, *args):
        # Cast only this block, rather than allocating a BF16 copy of all EMA weights.
        state = {'block.' + n: v.to(device=self.params[index][n].device,
                                  dtype=self.params[index][n].dtype)
                 for n, v in self.shadow[index].items()}
        return torch.func.functional_call(BlockCall(block), state, args, strict=False)

    @torch.no_grad()
    def update(self):
        for params, shadow in zip(self.params, self.shadow):
            for name, value in shadow.items():
                current = self.dense(params[name], torch.float32).to(device=value.device)
                if value.dtype == torch.float32:
                    value.lerp_(current, 1 - self.decay)
                else:
                    from sdnq.optim.utils import copy_stochastic_
                    updated = value.float().lerp_(current, 1 - self.decay)
                    copy_stochastic_(value, updated, use_stochastic_rounding=self.stochastic_rounding)

    def state_dict(self):
        return {'decay': self.decay, 'stochastic_rounding': self.stochastic_rounding,
                'shadow': [{n: v.cpu() for n, v in group.items()} for group in self.shadow]}

    def load_state_dict(self, state):
        if state['decay'] != self.decay or state['stochastic_rounding'] != self.stochastic_rounding:
            raise ValueError('Resume Self-Flow with the saved EMA decay/rounding settings')
        if len(state['shadow']) != len(self.shadow):
            raise ValueError('Self-Flow EMA checkpoint layout mismatch')
        for current, saved in zip(self.shadow, state['shadow']):
            if current.keys() != saved.keys():
                raise ValueError('Self-Flow EMA checkpoint keys mismatch')
            for name, value in current.items():
                source = saved[name]
                if source.shape != value.shape or source.dtype != value.dtype:
                    raise ValueError(f'Self-Flow EMA checkpoint precision/shape mismatch: {name}')
                value.copy_(source)


def forward_probe(model, image, times, context, *, token_ids=None,
                  capture_layer=4, stop_at=None, ema=None):
    """Batch-one native token path; image IDs select from a two-timestep table.

    Text retains the primary timestep. With uniform IDs this reduces to the
    ordinary model forward. Teacher evaluation stops at its feature layer.
    """
    if isinstance(image, (list, tuple)):
        return forward_packed(model, image, times, context, token_ids=token_ids,
                              capture_layer=capture_layer, stop_at=stop_at, ema=ema)
    if image.shape[0] != 1 or image.shape[2] != 1:
        raise ValueError('Probe supports one image per batch')
    _, _, _, h, w = image.shape
    text, text_mask = context
    def call(name, *args):
        module = getattr(model, name)
        return ema.forward_input(name, module, *args) if ema else module(*args)
    img = call('img_in', image.squeeze(2).flatten(2).transpose(1, 2))
    txt = call('txt_in', call('txt_norm', text))
    temb = call('time_text_embed', times.to(img.dtype), img)
    block_temb = (call('modulation_down', nn.functional.silu(temb).to(model.modulation_down.weight.dtype))
                  if model.params.modulation_rank else temb)
    freqs = model.pos_embed([(1, h, w)], device=img.device)
    if model.compiled_blocks:
        freqs = torch.view_as_real(freqs)
    mask = torch.cat((text_mask.bool(), torch.ones(1, h*w, device=img.device,
                                                 dtype=torch.bool)), dim=1)[:, None, None]
    metadata = () if model.attention_backend == 'sdpa' else packed_attention_metadata(mask)
    image_ids = (torch.zeros(h*w, device=img.device, dtype=torch.long)
                 if token_ids is None else token_ids.flatten().long())
    text_ids = torch.zeros(text.shape[1], device=img.device, dtype=torch.long)
    captured = None
    for i, block in enumerate(model.transformer_blocks):
        args = (img, txt, block_temb, freqs, mask, model.num_attention_heads,
                model.attention_backend, metadata, False, (image_ids, text_ids))
        if ema is not None:
            txt, img = ema.forward_block(i, block, *args)
        elif model.training and i in model.checkpoint_blocks:
            txt, img = checkpoint(model.block_forward, block, *args, use_reentrant=False)
        else:
            txt, img = model.block_forward(block, *args)
        if i + 1 == capture_layer:
            captured = img
        if stop_at == i + 1:
            return None, img
    scale, shift = model.norm_out.linear(model.norm_out.silu(temb).to(img.dtype)).chunk(2, -1)
    img = model.norm_out.norm(img) * (1 + scale[image_ids][None]) + shift[image_ids][None]
    output = model.proj_out(img).transpose(1, 2).reshape(1, model.out_channels, 1, h, w)
    return output, captured


def forward_packed(model, images, times, context, *, token_ids=None,
                   capture_layer=4, stop_at=None, ema=None):
    """Pack independent images/captions; timesteps are interleaved per image.

    Features and predictions are split per image so losses can weight samples
    equally, irrespective of resolution. Attention never crosses sample boundaries.
    """
    from ..modeling.mage_flow import _packed_varlen_metadata
    text, text_mask = context
    stride = 2 if token_ids is not None else 1
    if not images or text.shape[0] != len(images) or times.numel() != stride * len(images):
        raise ValueError('Self-Flow packed image/caption/timestep counts must match')
    if token_ids is not None and len(token_ids) != len(images):
        raise ValueError('Self-Flow needs one timestep mask per image')
    shapes = [im.shape[-2:] for im in images]
    if any(im.ndim != 5 or im.shape[:3] != (1, model.in_channels, 1) for im in images):
        raise ValueError('Packed Self-Flow latents must be [1,C,1,H,W]')
    lengths = [h*w for h,w in shapes]
    text_lengths = text_mask.sum(1).tolist()
    if any(n == 0 for n in text_lengths):
        raise ValueError('Each caption needs at least one valid token')
    device = images[0].device
    def call(name, *args):
        module = getattr(model, name)
        return ema.forward_input(name, module, *args) if ema else module(*args)
    img = call('img_in', torch.cat([im.flatten(2).transpose(1, 2) for im in images], dim=1))
    txt = call('txt_in', call('txt_norm', text[text_mask.bool()].unsqueeze(0)))
    temb = call('time_text_embed', times.to(img.dtype), img)
    block_temb = (call('modulation_down', nn.functional.silu(temb).to(model.modulation_down.weight.dtype))
                  if model.params.modulation_rank else temb)
    owners = torch.repeat_interleave(torch.arange(len(images), device=device),
                                    torch.tensor(lengths, device=device))
    txt_owners = torch.repeat_interleave(torch.arange(len(images), device=device),
                                        torch.tensor(text_lengths, device=device))
    img_ids = owners * stride
    if token_ids is not None:
        if any(ids.shape != shape for ids, shape in zip(token_ids, shapes)):
            raise ValueError('Self-Flow timestep masks must match latent grids')
        img_ids = img_ids + torch.cat([ids.flatten().long() for ids in token_ids])
    txt_ids = txt_owners * stride
    freqs = torch.cat([model.pos_embed([(1,h,w)], device=device) for h,w in shapes])
    if model.compiled_blocks:
        freqs = torch.view_as_real(freqs)
    if model.attention_backend == 'sdpa':
        joint_owners = torch.cat((txt_owners, owners))
        mask = (joint_owners[:, None] == joint_owners[None, :])[None, None]
        metadata = ()
    else:
        mask = None
        metadata = _packed_varlen_metadata(text_lengths, lengths, device)
    captured = None
    for i, block in enumerate(model.transformer_blocks):
        args = (img, txt, block_temb, freqs, mask, model.num_attention_heads,
                model.attention_backend, metadata, False, (img_ids, txt_ids))
        if ema is not None:
            txt, img = ema.forward_block(i, block, *args)
        elif model.training and i in model.checkpoint_blocks:
            txt, img = checkpoint(model.block_forward, block, *args, use_reentrant=False)
        else:
            txt, img = model.block_forward(block, *args)
        if i + 1 == capture_layer:
            captured = list(img.split(lengths, dim=1))
        if stop_at == i + 1:
            return None, list(img.split(lengths, dim=1))
    scale, shift = model.norm_out.linear(model.norm_out.silu(temb).to(img.dtype)).chunk(2, -1)
    img = model.norm_out.norm(img) * (1+scale[img_ids][None]) + shift[img_ids][None]
    outputs = model.proj_out(img).split(lengths, dim=1)
    return [x.transpose(1,2).reshape(1, model.out_channels, 1, h, w)
            for x,(h,w) in zip(outputs, shapes)], captured


def distributed_probe_forward(self, *args, probe=False, project_features=False, **kwargs):
    if not probe:
        return self._probe_original_forward(*args, **kwargs)
    prediction, features = forward_probe(self, *args, **kwargs)
    if project_features:
        if isinstance(features, list):
            lengths = [x.shape[1] for x in features]
            features = list(self.self_flow_projector(torch.cat(features, dim=1)).split(lengths, dim=1))
        else:
            features = self.self_flow_projector(features)
    return prediction, features
