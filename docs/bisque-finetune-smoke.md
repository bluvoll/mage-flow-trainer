# Bisque full-finetuning smoke test

Ten optimizer steps passed on GPU 1 (RTX 4090), batch size 1, accumulation 1, using the Bisque dataset (384 images), existing 1024-area latents (1344×768 bucket), and Cached Text Encoder with five caption variations. All **4,115,745,408 transformer parameters** were enabled for training, including AdaLN and input/output projections. No adapters were attached. The encoder remained frozen and was unloaded before transformer training.

The test used INT8 SDNQ **training** mode, `all_adaln` quantization protection, BF16 model computation, compiled packed `torch_varlen` attention, full gradient checkpointing, and AdamW at 2e-5 with Kahan compensation, quantized optimizer states, and CPU state offloading.

| Measurement | Result |
| --- | ---: |
| Completed steps | 10 |
| Peak training allocated VRAM | 14.28 GiB |
| Peak training reserved VRAM | 16.31 GiB |
| Median step time, steps 3–10 | 6.75 seconds |
| Trainable parameters | 4.116 billion |

Losses were finite. A 4,096-element sample of the first image-attention Q weight changed in 438 elements, confirming base-weight updates. This short test checks execution and memory, not resulting image quality. Checkpoint saving was disabled; the original model was not overwritten.

## Precision audit

The trainer explicitly loads BF16 weights and sets Accelerate mixed precision to `no`; it does not rely on autocast over an FP32 model. Runtime inspection found:

- 1,397,836,416 ordinary parameter elements in BF16.
- 2,717,908,992 SDNQ parameter elements with INT8 physical storage and BF16 logical/dequantized dtype.
- All observed trainable gradients in BF16.
- Quantized optimizer state on CPU with UINT8 storage and FP32 logical dtype; small unquantized states in BF16.
- FP32 intermediates in optimizer arithmetic and loss calculation.

Thus BF16 model/gradient training is explicit, but “every tensor and operation is BF16” would be inaccurate when SDNQ and optimizer quantization are enabled.

## Text-cache correction

The initial test rebuilt a workspace copy of the cache because the old fingerprint included transformer quantization mode. That was unnecessary: the text encoder uses frozen INT8 in both LoRA and full finetuning. Fingerprinting now uses effective encoder settings and reuses compatible legacy namespaces. A subsequent integration check ran the actual Bisque cache preparation with model loading forbidden: zero embeddings were pending and no encoder was loaded. The untouched original Bisque cache also contained 1,919 compatible embeddings under its legacy key.

See the [full-finetuning preset](../configs/mageflow-finetune-cached-int8.toml) and [per-step results and dtype audit](bisque-finetune-results.json). Set your dataset/model paths before running. The preset intentionally enables AdaLN for this full-model test; LoRA's AdaLN freeze rule is unchanged.

## Two-GPU test without optimizer-state offloading

With the same full-model configuration, batch size 1 per GPU, quantized optimizer states and Kahan enabled, setting `optimizer.offload_state=false` produced CUDA OOM on **both GPUs during the first optimizer update**, in the Kahan calculation. Zero optimizer steps completed. Forward/backward and distributed initialization succeeded.

This test limited compilation to one worker per rank and monitored host available RAM every second, with a 12 GiB stop threshold. Available host RAM never fell below **45.93 GiB**; the guard did not trigger. The test ended on CUDA OOM and released the GPUs. This result applies with Kahan enabled; disabling Kahan was not part of this trial.

See [per-rank OOM records and host-memory summary](bisque-finetune-no-offload-results.json).
