# SPDX-License-Identifier: GPL-3.0-or-later
# Forward adapted from ComfyUI's comfy/ldm/mage_flow/model.py.
import torch
import comfy.ops
import comfy.model_base
from comfy.ldm.mage_flow.model import MageFlowTransformer2DModel


class FP32Linear(comfy.ops.manual_cast.Linear):
    # Device moves must not round the calibrated factors through the trunk dtype.
    def _apply(self, fn, recurse=True):
        def preserve(tensor):
            if not tensor.is_floating_point():
                return fn(tensor)
            return tensor.to(device=fn(tensor.new_empty(0)).device, dtype=torch.float32)
        return super()._apply(preserve, recurse=recurse)


class ModulationHead(FP32Linear):
    def __init__(self, *args, output_dtype, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dtype = output_dtype

    def forward(self, x):
        return super().forward(x).to(self.output_dtype)


class CompressedMageFlowTransformer(MageFlowTransformer2DModel):
    def __init__(self, modulation_rank, operations, **kwargs):
        dim = kwargs['num_attention_heads'] * kwargs['attention_head_dim']
        compute_dtype = kwargs['dtype']

        class CompressedOperations(operations):
            @staticmethod
            def Linear(in_features, out_features, bias=True, device=None, dtype=None):
                if in_features == dim and out_features == 6 * dim:
                    return ModulationHead(modulation_rank, out_features, bias=bias,
                                          device=device, dtype=torch.float32,
                                          output_dtype=compute_dtype)
                return operations.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)

        super().__init__(operations=CompressedOperations, **kwargs)
        for block in self.transformer_blocks:
            block.img_mod[0] = torch.nn.Identity()
            block.txt_mod[0] = torch.nn.Identity()
        self.modulation_down = FP32Linear(dim, modulation_rank, device=kwargs.get('device'), dtype=torch.float32)

    def _forward(self, x, timestep, context, attention_mask=None, ref_latents=None, transformer_options={}, control=None, **kwargs):
        if attention_mask is not None and not torch.is_floating_point(attention_mask):
            attention_mask = (attention_mask - 1).to(x.dtype) * torch.finfo(x.dtype).max

        hidden_states, img_ids, orig_shape = self.process_img(x)
        num_embeds = hidden_states.shape[1]

        if ref_latents is not None:
            ref_num_tokens = []
            index = 0
            for ref in ref_latents:
                index += 1
                kontext, kontext_ids, _ = self.process_img(ref, index=index)
                hidden_states = torch.cat([hidden_states, kontext], dim=1)
                img_ids = torch.cat([img_ids, kontext_ids], dim=1)
                ref_num_tokens.append(kontext.shape[1])
            transformer_options = transformer_options.copy()
            transformer_options["reference_image_num_tokens"] = ref_num_tokens

        # Text tokens are not rotated in Mage-Flow: RoPE at position 0 is the
        # identity rotation.
        txt_ids = torch.zeros((x.shape[0], context.shape[1], 3), device=x.device)

        hidden_states = self.img_in(hidden_states)
        context = self.txt_norm(context)
        context = self.txt_in(context)

        temb = self.time_text_embed(timestep, hidden_states)
        block_temb = self.modulation_down(torch.nn.functional.silu(temb).float())

        patches_replace = transformer_options.get("patches_replace", {})
        patches = transformer_options.get("patches", {})
        blocks_replace = patches_replace.get("dit", {})

        if "post_input" in patches:
            for p in patches["post_input"]:
                out = p({"img": hidden_states, "txt": context, "img_ids": img_ids, "txt_ids": txt_ids, "transformer_options": transformer_options})
                hidden_states = out["img"]
                context = out["txt"]
                img_ids = out["img_ids"]
                txt_ids = out["txt_ids"]

        ids = torch.cat((txt_ids, img_ids), dim=1)
        image_rotary_emb = self.pe_embedder(ids).contiguous()
        del ids, txt_ids, img_ids

        transformer_options["total_blocks"] = len(self.transformer_blocks)
        transformer_options["block_type"] = "double"
        for i, block in enumerate(self.transformer_blocks):
            transformer_options["block_index"] = i
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["txt"], out["img"] = block(hidden_states=args["img"], encoder_hidden_states=args["txt"], encoder_hidden_states_mask=attention_mask, temb=args["vec"], image_rotary_emb=args["pe"], transformer_options=args["transformer_options"])
                    return out
                out = blocks_replace[("double_block", i)]({"img": hidden_states, "txt": context, "vec": block_temb, "pe": image_rotary_emb, "transformer_options": transformer_options}, {"original_block": block_wrap})
                hidden_states = out["img"]
                context = out["txt"]
            else:
                context, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=context,
                    encoder_hidden_states_mask=attention_mask,
                    temb=block_temb,
                    image_rotary_emb=image_rotary_emb,
                    transformer_options=transformer_options,
                )

            if "double_block" in patches:
                for p in patches["double_block"]:
                    out = p({"img": hidden_states, "txt": context, "x": x, "block_index": i, "transformer_options": transformer_options})
                    hidden_states = out["img"]
                    context = out["txt"]

            if control is not None:  # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        hidden_states[:, :add.shape[1]] += add

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states[:, :num_embeds]
        h, w = orig_shape
        return hidden_states.reshape(x.shape[0], h, w, self.out_channels).movedim(-1, 1)


class CompressedMageFlowBase(comfy.model_base.MageFlow):
    def __init__(self, config, device=None):
        comfy.model_base.QwenImage.__init__(self, config, comfy.model_base.ModelType.FLOW,
                                           device=device, unet_model=CompressedMageFlowTransformer)
