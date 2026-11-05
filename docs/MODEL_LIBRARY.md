# Curated MoE model library

Particle 0.6.0 replaces open-ended model support with the exact collection requested on 5 November 2026. Search filters these entries and accepts only approved source/quantization variants; it no longer searches arbitrary Hugging Face models. Graph availability cannot add models to the product catalog.

## Models

| Model | Preferred source |
| --- | --- |
| Qwen3.6-35B-A3B | [mlx-community/Qwen3.6-35B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit) |
| Qwen3-Coder-30B-A3B | [mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit) |
| GLM-4.7-Flash | [mlx-community/GLM-4.7-Flash-4bit](https://huggingface.co/mlx-community/GLM-4.7-Flash-4bit) |
| NVIDIA Nemotron 3.5 Lightning 30B-A3B | [mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit](https://huggingface.co/mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit) |
| Qwen3-30B-A3B-Thinking-2507 | [mlx-community/Qwen3-30B-A3B-Thinking-2507-4bit](https://huggingface.co/mlx-community/Qwen3-30B-A3B-Thinking-2507-4bit) |
| Qwen3-30B-A3B-Instruct-2507 | [mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit](https://huggingface.co/mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit) |
| Gemma 4 26B-A4B | [mlx-community/gemma-4-26b-a4b-it-4bit](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit) |
| Qwen3-30B-A3B | [mlx-community/Qwen3-30B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3-30B-A3B-4bit) |
| GPT-OSS-20B | [mlx-community/gpt-oss-20b-MXFP4-Q8](https://huggingface.co/mlx-community/gpt-oss-20b-MXFP4-Q8) |
| DeepSeek-Coder-V2-Lite-Instruct | [mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit](https://huggingface.co/mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit) |

## Fast

| Model | Preferred source |
| --- | --- |
| Qwen1.5-MoE-A2.7B-Chat | [mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit](https://huggingface.co/mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit) |
| DeepSeekMoE-16B-Chat | [BlueMoonlight/deepseek-moe-16b-chat-mlx-4Bit](https://huggingface.co/BlueMoonlight/deepseek-moe-16b-chat-mlx-4Bit) |
| MiniCPM-MoE-8x2B | [openbmb/MiniCPM-MoE-8x2B](https://huggingface.co/openbmb/MiniCPM-MoE-8x2B) |

Fast is a collection label, not a measured throughput guarantee. MiniCPM's original checkpoint is BF16 and substantially larger than a 4-bit export; native byte-preserving conversion does not silently requantize it. Memory and disk requirements still apply.

## Conversion and execution

The existing graph compiler remains an internal implementation mechanism. Catalog admission checks repository identity and configuration dimensions, followed by full header, graph, tokenizer/template and quantization inspection. Related dense, vision-specific, altered or unrelated checkpoints are excluded even when their names partially match. Local imports retain local provenance and must match a supported structural family; tensor metadata alone cannot establish training identity.

Added native Qwen2-MoE, DeepSeek-MoE and MiniCPM-MoE graphs preserve their respective shared branches, route normalization, attention biases and MiniCPM embedding/residual/output scaling. Nemotron's named block schedule is normalized into the bundled hybrid graph's schedule. Repository `auto_map` pointers are ignored only where a native implementation exists; downloaded repository code is never executed.

Unused vision, MTP and backend-redundant source tensors are retained in a checksummed `source-only.bin` archive. They remain part of the exact conversion inventory and original source hash, but are not loaded into inference memory. Gemma uses the text graph and retains its native attention schedule. Original quantization, source files and model licenses are preserved.

MiniCPM's original PyTorch ZIP64 state dictionary can be inspected and ingested by a bounded native data-only reader. It recognizes tensor metadata/storage references, checks bounds, endianness and contiguous strides, and copies original tensor bytes. It never invokes pickle functions, imports Python modules, or loads arbitrary objects. Unsupported archive encodings receive an error. Remote storage headers use a bounded queue; no whole checkpoint is loaded for inspection.

Each installed supported model has its own expert count, cache, read workers, prompt chunk and context profile. Switching to a shorter-context model also bounds its response allowance. Native expert selection is the reference; reducing experts remains explicitly experimental. New graph paths are early-access Chat; model names containing Coder do not by themselves qualify coding-tool execution.

## Existing installations

Sarvam/Churo, Chronos, Mixtral and other models outside the collection are no longer offered or selectable. Their files, registry entries and conversations are retained; Installed explains the change and still allows removal. First launch opens the curated library instead of automatically downloading Sarvam. No old model files or chats are deleted during the catalog change.

The existing ParticleDesign PNG grain/wash materials, green palette, PT Serif/Poppins/Space Mono and dark background remain in use. All thirteen entries are visible immediately; Install appears only after inspection passes. Cached compatible revisions avoid repeating tensor-header inspection during ordinary browsing.

## Evidence and limits

Metadata inspection is distinct from a complete model inference qualification. Focused synthetic byte/parity checks exercise the new graph and archive paths; no complete pretrained checkpoints are downloaded just to test this update. There is no universal speed, memory-fit, long-context quality or tool-call guarantee.

Implementation references: [MLX LM Qwen2-MoE](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen2_moe.py), [MLX LM DeepSeek](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/deepseek.py), [OpenBMB MiniCPM model](https://huggingface.co/openbmb/MiniCPM-MoE-8x2B/blob/main/modeling_minicpm.py), [PyTorch serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html). These references inform bundled native code; model repositories are not executed.
