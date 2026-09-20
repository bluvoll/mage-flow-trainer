"""Print readable safetensors metadata without loading checkpoint weights."""

import argparse
import json

from safetensors import safe_open


def read_metadata(path):
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
    for key in ("training_config", "training_state", "training_versions",
                "adapter_config", "model_config", "source_checkpoint"):
        if key in metadata:
            try:
                metadata[key] = json.loads(metadata[key])
            except (ValueError, TypeError):
                pass  # Preserve older or third-party metadata verbatim.
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--section", choices=("all", "config", "state", "versions", "source"),
                        default="all")
    args = parser.parse_args()
    metadata = read_metadata(args.checkpoint)
    if args.section != "all":
        key = "source_checkpoint" if args.section == 'source' else "training_" + args.section
        if key not in metadata:
            parser.error(f"This checkpoint has no {key} metadata; older exports cannot recover it.")
        metadata = metadata[key]
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
