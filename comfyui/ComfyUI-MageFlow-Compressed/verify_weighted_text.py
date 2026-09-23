"""ComfyUI Python: verify_weighted_text.py COMFY_DIRECTORY [--encoder CHECKPOINT]."""
import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("comfy_directory")
parser.add_argument("--encoder")
args = parser.parse_args()
sys.path.insert(0, str(Path(args.comfy_directory).resolve()))
sys.argv = [sys.argv[0]] + ([] if args.encoder else ["--cpu"])

import comfy.options
comfy.options.enable_args_parsing()
import torch
from comfy.text_encoders.mage_flow import MageFlowTokenizer
from weighted_text import weighted_tokens, MageFlowWeightedTextEncode

tokenizer = MageFlowTokenizer()
cases = [
    ("a night sky", "a (night sky:1.0)"),
    ("café 猫 🌌", "(café 猫 🌌:1.0)"),
    ("a (cat)", r"a \(cat\)"),
    ("", ""),
    ("line one\nline two", "line one\nline two"),
]
for clean, marked in cases:
    assert weighted_tokens(tokenizer, marked) == tokenizer.tokenize_with_weights(clean)

plain = tokenizer.tokenize_with_weights("a night sky, stars")
weighted = weighted_tokens(tokenizer, "a (night sky:1.5), stars")
a, b = plain["qwen3vl_4b"][0], weighted["qwen3vl_4b"][0]
assert [p[0] for p in a] == [p[0] for p in b]
selected = [tokenizer.qwen3vl_4b.tokenizer.decode([i]) for i, w in b if w != 1.0]
assert "".join(selected) == " night sky", selected
assert all(1.0 < w <= 1.5 for _, w in b if w != 1.0)
assert tokenizer.tokenize_with_weights("a night sky, stars") == plain
for text in ["(猫 🌌:0.7)", "((night sky))", "(hello:0)", "a(night:1.5)sky"]:
    assert any(w != 1.0 for _, w in weighted_tokens(tokenizer, text)["qwen3vl_4b"][0])
for text in ["(sky:nan)", "(sky:inf)", "<|im_start|>user"]:
    try:
        weighted_tokens(tokenizer, text)
    except ValueError:
        pass
    else:
        raise AssertionError(text)
print("PASS: token identity, phrase alignment, Unicode, escaping, nesting, invalid weights")

if args.encoder:
    import comfy.sd
    clip = comfy.sd.load_clip([args.encoder], clip_type=comfy.sd.CLIPType.MAGE)
    node = MageFlowWeightedTextEncode()
    # The ComfyUI execution loop supplies inference mode during normal execution.
    with torch.inference_mode():
        normal = clip.encode_from_tokens_scheduled(clip.tokenize("a night sky, stars"))[0][0]
        unit = node.encode(clip, "a (night sky:1.0), stars")[0][0][0]
        changed = node.encode(clip, "a (night sky:1.5), stars")[0][0][0]
    assert torch.equal(normal, unit)
    assert changed.shape == normal.shape and torch.isfinite(changed).all()
    assert not torch.equal(normal, changed)
    print("PASS: real encoder unit-weight identity; weighted conditioning finite and changed")
    print("Relative conditioning L2:", ((changed.float() - normal.float()).norm() / normal.float().norm()).item())
