# WAI AdamW SDNQ checkpoint magnitude audit

Checkpoint: `output/WAI-mageflow-Finetune-AdamW-SDNQ/WAI-mageflow-Finetune-AdamW-SDNQ.safetensors`.

Relative L2 percent = `100 * ||checkpoint - base||₂ / ||base||₂`. Every tensor was compared against the same rank-256 compressed MageTrail base, in CPU chunks. Tensor key sets, shapes, and aggregate reference squared norms match the archived audits. No GPUs used. The previous checkpoint files are not needed to read their saved measurements.

New checkpoint metadata: `sdnq.optim.adamw.AdamW`, quantized UINT8 optimizer buffers, INT8 transformer storage, LR 1e-5, betas (0.9, 0.999), weight decay 0.01, constant schedule with 5 warmup steps, 252 updates, effective batch 12 on two GPUs. Trainer gradient clipping disabled; optimizer reports final_norm_mode=clip. Compressed AdaLN trained in FP32 and exported in BF16. No nonfinite saved values.

| Run | All weights (%) | Shared projection (%) | Block modulation weights (%) | Image attention (%) | Text attention (%) | MLP (%) |
|---|---:|---:|---:|---:|---:|---:|
| AdamW, 1e-5 | 0.070791 | 2.950547 | 0.443234 | 0.967603 | 0.931266 | 1.020100 |
| Adafactor rms_clip, 1e-5 | 0.069560 | 0.518254 | 0.250647 | 0.967856 | 0.930085 | 1.014862 |
| Adafactor relative, 1e-5 | 0.068102 | 0.049936 | 0.035834 | 0.954027 | 0.910480 | 1.001134 |
| Adafactor relative, 1e-4 | 0.073517 | 0.182265 | 0.189183 | 0.962840 | 0.920933 | 1.006554 |
| Adafactor rms_clip + first moment, 1e-5 | 0.068097 | 0.208929 | 0.055096 | 0.953962 | 0.910584 | 1.001419 |
| Adafactor relative + first moment, 1e-4 | 0.072598 | 0.075362 | 0.051087 | 0.954006 | 0.910509 | 1.001256 |
| Older checkpoint named AdamW (378 steps) | 0.081418 | 7.293997 | 0.960974 | 1.025227 | 1.012472 | 1.103271 |
| Adafactor rms_clip, vsf3 VAE swap (168 steps) | 0.068879 | 0.565328 | 0.246582 | 0.960355 | 0.919376 | 1.007842 |
| Adafactor rms_clip, f2vae VAE swap (168 steps) | 0.068856 | 0.513630 | 0.229752 | 0.960425 | 0.919537 | 1.007832 |

The original optimizer-comparison runs were measured at 252 steps and effective batch 12. Both VAE-swap runs were measured at 168 steps and effective batch 18: Adafactor rms_clip without first moment, configured LR 1e-5, REX schedule with 50 warmup steps, saved LR 9.950080379e-6. They use the same base, dataset path, and INT8 transformer storage; RTI is disabled. The configured VAE sources are Flux2 Anime VSF3_epoch_1 and FLUX.2-dev respectively. Since training reads cached latents, metadata identifies the intended VAE but does not independently verify the cache contents. These runs have equal nominal sample exposure (3024 sample slots) to 252 x 12, but different update counts and scheduling. The older checkpoint metadata says step 378 and an Adafactor run name despite its AdamW filename; it lacks embedded optimizer settings, so its optimizer and LR cannot be independently confirmed from that checkpoint. These are run comparisons, not a controlled optimizer-only quality experiment.

New absolute delta L2: 20.73105316. Reference L2: 29284.77898352. Whole-model change from epoch 1 to final: 0.01729827%.

AdamW has 1.77% more whole-model relative change than rms_clip at 1e-5, and 3.71% less than relative at 1e-4. Its shared projection changes 5.69x as much as rms_clip, and block modulation weights 1.77x as much. Other AdaLN parameters change 3.034179%, compared with rms_clip 1.091793%.

Whole-model relative L2 is dominated by large compressed modulation biases in the denominator. Group-level comparisons are more informative. These differences also include quantization and export rounding; without a matched untrained quantized/exported baseline they do not isolate optimizer-induced learning. Larger weight differences do not establish stronger style, better quality, or better generalization.

Both added checkpoints contain no nonfinite saved values. VSF3 and FLUX.2 have similar whole-model relative L2 changes, slightly below the original rms_clip run. VSF3 changes both the shared projection and per-block modulation weights more than the FLUX.2 run. These measurements cannot determine whether a VAE swap improves image quality.
