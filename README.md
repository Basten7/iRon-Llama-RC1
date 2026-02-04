# llama.cpp Metal (macOS Intel + AMD dGPU) – W6800X Duo / Xeon

This repository is a **llama.cpp (ggml-metal) focused release** tuned for **macOS Intel (MacPro7,1)** with **discrete AMD GPUs** (e.g. **Radeon PRO W6800X Duo**) and **Intel Xeon** CPUs.

The goal is to restore/ensure:
- **Correct GPU device selection** via `GGML_METAL_DEVICE_INDEX`
- **Correct + fast mmap loading on discrete GPUs** by mirroring mmapped weights into **VRAM (StorageModePrivate)** when possible
- **Stable text generation** on AMD dGPU by avoiding known non-determinism in Metal concurrency paths on discrete devices
- **VRAM budget & fallback**: if VRAM budget is exceeded, the backend falls back to shared mmap (slow but correct) instead of crashing

> Target hardware: Mac Pro (Intel) + AMD dGPU (RDNA2 / W6800X Duo).  
> Metal feature note: simdgroup matrix multiply ops are not available on these Macs; this release focuses on correctness + bandwidth-aware memory placement.

---

## Highlights

### ✅ GPU Device Selection
Select which GPU to use (macOS):

export GGML_METAL_DEVICE_INDEX=2   # example index

✅ Discrete GPU mmap → VRAM Private Mirror (Fast mmap)

When loading a model with mmap (default), mmapped weights are first mapped as a shared buffer and then one-shot blit copied into a private VRAM mirror (when enabled and budget allows). Kernels read from the private mirror for bandwidth and stability.
✅ VRAM Budget + Fallback

To avoid VRAM thrashing/OOM:

    private mirrors are created only up to a computed budget

    when budget is exceeded, the backend falls back to shared mmap (slow but correct) and logs a warning

Configurable:

export GGML_METAL_VRAM_RESERVE_MB=2048    # default reserve margin
export GGML_METAL_VRAM_BUDGET_MB=32000    # optional hard cap override

✅ Stability on AMD dGPU

On discrete AMD GPUs, some concurrency paths can cause non-deterministic outputs (garbled text).
For stability:

export GGML_METAL_CONCURRENCY_DISABLE=1

Build
Clone

git clone https://github.com/<your-org-or-user>/<your-repo>.git
cd <your-repo>

Configure + build (CMake)

cmake -S . -B build -DGGML_METAL=ON
cmake --build build -j

Quick Validation
1) Verify GPU selection

export GGML_METAL_DEVICE_INDEX=2
./build/bin/llama-cli -ngl 99 -c 128 --jinja --no-mmap \
  -m ~/Models/Qwen3-30B-A3B-Q4_K_M.gguf -p "test"

Expected log snippet:

    using device index X from GGML_METAL_DEVICE_INDEX

    GPU name: AMD Radeon PRO W6800X Duo (or your selected GPU)

2) Fast + correct generation (recommended defaults for AMD dGPU)

export GGML_METAL_DEVICE_INDEX=2
export GGML_METAL_CONCURRENCY_DISABLE=1
unset GGML_METAL_MMAP_PRIVATE_DISABLE

./build/bin/llama-cli -ngl 99 -c 4096 --jinja \
  -m ~/Models/Qwen3-0.6B-F32.gguf \
  -p "Explain TensorFlow in French."

Performance Notes
mmap vs no-mmap

On discrete GPUs, --no-mmap is often fast because weights are copied into GPU-friendly buffers.
This release aims to make mmap fast too by enabling a private VRAM mirror.

    Default: mmap + private mirror (fast on dGPU)

    Debug fallback (very slow on dGPU, reads over PCIe):

    export GGML_METAL_MMAP_PRIVATE_DISABLE=1

VRAM Tuning

Your device logs recommendedMaxWorkingSetSize. Budget is derived from it:

    budget = recommendedMaxWorkingSetSize - VRAM_RESERVE_MB

Typical values:

    VRAM_RESERVE_MB=2048 (safe)

    VRAM_RESERVE_MB=1024 (aggressive / max perf, less safety margin)

Benchmark
llama-bench

export GGML_METAL_DEVICE_INDEX=2
export GGML_METAL_CONCURRENCY_DISABLE=1

./build/bin/llama-bench -ngl 99 \
  -m ~/Models/Qwen3-30B-A3B-Q4_K_M.gguf \
  --mmap 1

Server & Long Context (KV cache)
Long-context slowdown is expected

As context grows, attention cost increases because each new token attends over an ever larger KV cache. This can drop token/s significantly at very large ctx sizes.
Flash-Attn + KV cache types (observations)

On this hardware/stack:

    --flash-attn ON with --cache-type-v q8_0 can become very slow and may fall back to CPU

    --flash-attn ON with --cache-type-v f16 is often faster and stays on GPU

Example server command:

export GGML_METAL_DEVICE_INDEX=2
export GGML_METAL_CONCURRENCY_DISABLE=1
export GGML_METAL_VRAM_RESERVE_MB=2048

./build/bin/llama-server -ngl 99 -sm layer --port 8085 --jinja \
  --ctx-size 62144 \
  --flash-attn \
  --cache-type-k f16 --cache-type-v f16 \
  -m ~/Models/Qwen3-Coder-30B-A3B-Q4_0.gguf

Environment Variables Reference
Variable	Purpose	Suggested
GGML_METAL_DEVICE_INDEX	Select GPU device index on macOS	set to your W6800X
GGML_METAL_CONCURRENCY_DISABLE	Disable concurrency for stability on discrete AMD GPUs	1
GGML_METAL_ORDERED_COMMIT	Ordered command buffer commits (debug)	optional
GGML_METAL_MMAP_PRIVATE_DISABLE	Disable mmap→private VRAM mirror (debug, slow on dGPU)	unset
GGML_METAL_VRAM_RESERVE_MB	VRAM safety margin	2048
GGML_METAL_VRAM_BUDGET_MB	Override total VRAM budget	optional
Known Limitations

    Metal simdgroup matrix multiply ops are not supported on these Intel + AMD dGPU Macs.

    Very large context sizes will still slow down generation due to KV cache bandwidth.

    Some concurrency paths on discrete GPUs can cause non-deterministic outputs; this repo defaults to stable behavior via recommended env vars.

License

This repository is based on upstream llama.cpp (ggml). Please refer to upstream licensing terms and preserve notices in derived works.
