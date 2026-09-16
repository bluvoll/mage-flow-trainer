# Persistent caption variation caching

Enable **Cache text embeddings** and set **Caption variations per image** above zero. The trainer builds a SQLite database before loading Mage-Flow, then releases Qwen3-VL. The database defaults to `caption_variations.sqlite` in the first dataset folder; set **Caption cache database** to place it on a larger drive. One database can cover all subsets and share exact caption matches across them.

```toml
[train]
cache_text_embeddings = true
caption_variations = 100
text_cache_batch_size = 4
caption_cache_path = "/path/to/storage/captions.sqlite"

[dataset.caption]
caption_mode = "mixed"
shuffle_tags = true
tag_dropout_percent = 0.1
caption_dropout_percent = 0.1
```

Use the normal caption controls: tags, NL, tags+NL, NL+tags, weighted mixed mode, protected tags, tag shuffling/dropout, and sentence shuffling. `caption_variations = 0` retains the original fixed-caption RAM cache and requires caption augmentation off. Latent caching is independent; cached latents provide the lowest training memory.

`train.text_cache_batch_size` defaults to **4 captions per GPU** for both SQLite
and fixed-caption RAM caching. Increase it when the encoder has spare VRAM;
reduce it to 1 for the lowest caching memory. Captions are padded within each
batch and stored individually without padding. Changing this setting reuses
existing cached embeddings; numerical differences from batched GPU execution
are possible, especially with BF16/quantized encoders. It does not change the training batch size.

## Slot semantics

N means N deterministic augmentation attempts per source image, not N distinct strings. If 100 attempts produce 37 unique captions, SQLite retains 100 slots pointing to 37 embeddings. It neither retries until all captions differ nor collapses the training sequence to 37 visits. Identical captions from other images share the same embedding too.

Each source cycles through a deterministically permuted slot sequence. Dataset repeats and resolution variants advance that sequence within an epoch; epochs continue it. A new cycle permutes the slots again. This is an O(1)-space affine permutation rather than a fully uniform random shuffle of all possible permutations. The sampler's traversal order need not match the slot order, but a complete set of visits covers all slots. Dropped/incomplete batches may skip visits. DDP padding may duplicate a visit, as with the existing sampler.

Whole-caption dropout is sampled at training time using one shared empty-prompt embedding. A dropped visit still consumes its slot. It therefore does not spend the stored slot budget on repeated empty prompts. The configured caption-mode frequencies are sampled during slot generation; duplicates preserve those realized frequencies. A finite pool cannot reproduce unlimited online augmentation diversity.

Fixed config/seed/dataset ordering reproduces caption selection after resume, independently of data-loader worker count. Increasing N appends generated slots and reuses existing embeddings, but changes the sampling permutation; do it between runs when exact continuation matters.

## Persistence and memory

Embeddings are unpadded tensors in SQLite BLOBs, including dtype and dimensions. Training reads only the requested batch and uses an 8 MiB SQLite page cache per reader connection. No full embedding collection is loaded into RAM. Each worker opens its own read-only connection; the main training process reads the selected embeddings.

All DDP ranks participate in encoding missing embeddings, using their selected GPUs. Rank 0 plans the caption slots, then caption hashes assign each pending embedding to exactly one rank. Completed embeddings are reused, even when resuming with a different GPU count. Ranks with no pending work skip loading Qwen3-VL.

A file lock serializes cache-building jobs; ranks within the active job share SQLite through short write transactions. Each batch commits before the next GPU forward pass, so encoding never holds the database's writer lock. SQLite WAL permits concurrent readers, and committed entries survive interrupted preprocessing. Use a shared local filesystem that supports SQLite WAL and file locking; do not manually delete the WAL/SHM files while a process is using the database.

The trainer sets the distributed process-group timeout to **60 minutes**, including cache-planning and completion barriers. This is a limit on an individual collective wait, not on total caching time. Multi-GPU encoding reduces the time other ranks spend waiting, but a stalled rank or a wait longer than 60 minutes can still time out. Each active GPU loads its own encoder, increasing aggregate CPU/GPU memory during preprocessing; the encoders are released before Mage-Flow loads.

Caption pools are keyed by source identity, raw caption contents, augmentation settings, and seed. Embeddings are keyed separately by exact caption and an encoder fingerprint covering model/tokenizer file paths, sizes and modification timestamps, text length, prompt template/prefix, dtype, effective text-encoder quantization settings, and library versions. A changed fingerprint builds a separate embedding namespace. This is a file-metadata fingerprint, not a content hash of every model weight. Old namespaces remain on disk; rebuilding never silently deletes them.

Switching the transformer between LoRA and full finetuning does not invalidate text embeddings. The encoder stays frozen and uses its own fixed skip policy and disabled quantized matmul. The cache now fingerprints those effective settings, rather than the transformer’s training mode or skip list. Compatible older namespaces are reused directly without copying embedding blobs. Changing actual encoder precision, quantization, token limit, model files, or library versions still invalidates the cache.

At 200 tokens and 2,560 channels, BF16 embeddings cost about 1 MB each. At 20,000 images and 100 unique captions each, plan for roughly 2 TB plus SQLite overhead. Deduplication can reduce this substantially, depending on actual captions and augmentation settings. Preprocessing time is proportional to newly needed unique embeddings, not slot count alone.

Texture curricula and concept-preservation probes remain unsupported with text caching and are rejected. The text encoder is still needed to build missing entries; a fully populated cache skips loading it on later starts.

## Validation

Tests cover duplicate slot frequencies, pool extension, fingerprint separation, empty-caption reuse, worker serialization, and padded batch reconstruction. Real training used the ten-image kuse benchmark, 1024-area buckets, rank-32 LoRA, BF16 adapters, SDNQ INT8, and four variations per image. Forty caption embeddings plus one empty embedding were built; 20 optimizer steps completed at 5.395 GiB peak PyTorch allocation (5.600 GiB reserved). This excludes CUDA context/other-process memory and is not a minimum GPU capacity guarantee.

The run used shuffle and 10% tag dropout; 10% whole-caption dropout was enabled but happened to produce zero empty-caption visits in those 20 steps. The deterministic dropout edge cases are covered separately by tests. This is a functional/memory smoke run, not a long-run quality test or a 20,000-image scale test.

A second 20-step run reused all entries with two data-loader workers and no encoder load, again reaching 5.395 GiB peak allocated memory. The test database occupied 25.39 MiB. Cold/warm four-step checks selected the same captions; bitwise training-loss identity is not guaranteed by these stochastic/compiled paths. The cache builder preserves the Torch RNG stream around text encoding.

## Text-encoder batching check

A short RTX 4090 test encoded the same 16 synthetic captions, with mixed lengths
and long repetitive text, using the frozen INT8 SDNQ encoder with BF16 compute.
These are encoder-only timings after a warmup, including transfer to CPU;
SQLite writes and model loading are excluded.

| Captions per batch | Captions/second | Peak PyTorch allocated VRAM |
| --- | ---: | ---: |
| 1 | 15.48 | 4.70 GiB |
| 4 | 31.91 | 4.99 GiB |
| 8 | 29.51 | 5.40 GiB |

Larger batches are not always faster: padding increases work. Batched INT8/BF16
embeddings were not numerically identical to batch 1 (maximum relative L2
difference about 18.9%, minimum cosine similarity about 0.982 on this synthetic
probe). Repeated batch-1 encoding was identical. An unquantized FP32 control
reduced the batch-1 versus batch-4 maximum relative difference to 0.0059%,
supporting a numerical-precision explanation rather than a padding/assignment
error. This is not an image-quality evaluation; use batch size 1 if matching the
previous single-caption encoding path matters.
