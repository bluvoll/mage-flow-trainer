"""Isolated Qwen3-VL benchmark; run each mode in a fresh process on one GPU."""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from trainer.modeling.loader import load_components
from trainer.modeling.batched import PROMPT_TEMPLATE_ENCODE, PROMPT_TEMPLATE_ENCODE_START_IDX
from trainer.modeling.mageflow_text import encode_text_hidden
from trainer.training.quant import QuantConfig, quantize_module, text_encoder_quant_config


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--mode', choices=['bf16', 'int8', 'int8_matmul'], required=True)
    p.add_argument('--model', default='mage-flow')
    p.add_argument('--captions', default='/home/bluvoll/distills/wai')
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--compile-blocks', action='store_true')
    p.add_argument('--batches', nargs='+', type=int, default=[1, 4, 8])
    p.add_argument('--limits', nargs='+', type=int, default=[128, 512])
    p.add_argument('--fixed-padding', action='store_true',
                   help='Use the largest sampled caption length at every batch size.')
    a = p.parse_args()
    if min(a.batches + a.limits + [a.steps]) < 1:
        p.error('Batches, limits, and steps must be positive')
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'captions.json'
    if manifest.exists():
        captions = json.loads(manifest.read_text())
    else:
        paths = sorted(Path(a.captions).glob('*.txt'))
        paths = [x for x in paths if not x.stem.endswith('_nl')][:64]
        captions = [x.read_text().strip() for x in paths]
        if len(captions) < 8:
            raise ValueError('Need at least 8 real captions')
        manifest.write_text(json.dumps(captions))
    if len(captions) < max(a.batches):
        raise ValueError('Caption manifest is smaller than the largest requested batch')
    torch.manual_seed(42)
    c = load_components(a.model, load_transformer=False, load_vae=False)
    if a.mode != 'bf16':
        q = text_encoder_quant_config(QuantConfig(mode='frozen', quantize_text_encoder=True))
        c.text_encoder = quantize_module(c.text_encoder, q, torch.device('cuda:0'), torch.bfloat16,
                                        a.mode == 'int8_matmul')
    enc = c.text_encoder.eval().to('cuda:0')
    if a.compile_blocks:
        for block in enc.model.language_model.layers:
            block.compile(dynamic=True)
    # Verify the backbone's final output against the wrapper: Transformers
    # versions differ in the hidden-state capture/normalization convention.
    def embedding_only(ids, mask):
        return enc.model(input_ids=ids, attention_mask=mask, use_cache=False,
                         output_hidden_states=False, return_dict=True).last_hidden_state
    results = []
    for batch in a.batches:
        for limit in a.limits:
            max_length = limit + PROMPT_TEMPLATE_ENCODE_START_IDX
            if a.fixed_padding:
                all_tokens = c.tokenizer(
                    [PROMPT_TEMPLATE_ENCODE.format(t) for t in captions[:max(a.batches)]],
                    truncation=True, max_length=max_length)
                max_length = max(map(len, all_tokens['input_ids']))
            tokens = c.tokenizer([PROMPT_TEMPLATE_ENCODE.format(t) for t in captions[:batch]],
                padding='max_length' if a.fixed_padding else True,
                truncation=True, max_length=max_length,
                return_tensors='pt').to('cuda:0')
            ids, mask = tokens.input_ids, tokens.attention_mask
            reference_path = out / f'bf16-b{batch}-l{limit}.pt'
            baseline = encode_text_hidden(enc, ids, mask)
            if a.mode == 'bf16' and not a.compile_blocks:
                torch.save(baseline.cpu(), reference_path)
            reference = torch.load(reference_path, weights_only=True).to('cuda:0')
            valid = mask.bool(); valid[:, :PROMPT_TEMPLATE_ENCODE_START_IDX] = False
            for method, fn in [('wrapper', lambda: encode_text_hidden(enc, ids, mask)),
                               ('embedding_only', lambda: embedding_only(ids, mask))]:
                y = fn()
                same_path_error = (y.float() - baseline.float()).abs().max().item()
                expected, actual = reference[valid].float(), y[valid].float()
                error = (actual - expected).square().mean().sqrt() / expected.square().mean().sqrt()
                cosine = F.cosine_similarity(actual, expected, dim=-1).mean().item()
                del y, actual, expected
                for _ in range(3):
                    warm = fn(); del warm
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                durations = []
                for _ in range(a.steps):
                    start = time.perf_counter(); y = fn(); torch.cuda.synchronize()
                    durations.append(time.perf_counter() - start); del y
                row = dict(mode=a.mode, compiled=a.compile_blocks, method=method, batch=batch, caption_limit=limit,
                    padded_tokens=ids.shape[1], valid_tokens=int(mask.sum()),
                    median_ms=statistics.median(durations)*1000,
                    captions_per_second=batch/statistics.median(durations),
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    relative_rmse=error.item(), mean_token_cosine=cosine,
                    versus_wrapper_max_abs=same_path_error)
                results.append(row); print(json.dumps(row), flush=True)
            del baseline, reference
    name = a.mode + ('-compiled' if a.compile_blocks else '')
    (out / f'{name}.json').write_text(json.dumps(dict(
        device=torch.cuda.get_device_name(), torch=torch.__version__,
        arguments=vars(a), results=results), indent=2))


if __name__ == '__main__':
    main()
