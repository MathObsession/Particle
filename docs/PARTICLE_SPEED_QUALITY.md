# Particle speed/quality experiments

25 October 2026. The user accepts approximately 20–25% measured quality loss if the result reaches at least 4 generated tokens/s. That is a target, not a current qualification. Native Sarvam weights, official tokenizer/chat formatting, full attention, FP32 KV, 12 RAM slots and the approved SSD tier remain available. No approximate mode becomes the default.

## Implemented first: explicit routed-expert budget

`--active-experts 2...6` defaults to **6**, preserving native selection and accumulation. Values below six are approximate: rank experts using the original selection-only bias, retain the first k, normalize their unbiased sigmoid scores and scale the mixture by 2.5. The shared branch, leading dense layer, attention, output head and original quantized weight bytes are unchanged. This reduces demanded expert reads and routed GPU work. It does not promise a particular speed or quality loss.

The same budget applies to scalar decode and grouped prompt processing. The prompt planner binds only real selected ranks, and reduction retains native rank order. Changing a budget requires a KV reset; earlier KV from a different model profile cannot be reused. The compact GPU router is native-six-only and falls back to CPU selection for reduced budgets. Timing JSON records `active_experts` and `approximate_execution`.

To try a three-expert candidate after its build/validation:

```sh
cd /Users/lakshyaprajapati/Repositories/Particle
Scripts/particle run \
  --model /Volumes/Particle/Models/sarvam-30b/sarvam-30b-q4_k_m.particle \
  --storage-profile sd-hybrid --ssd-cache-policy mirror --ssd-cache-mib 18432 \
  --expert-cache-slots 12 --ssd-read-workers 2 \
  --attention reference --projection vector --deferred-decode --grouped-experts --ffn-vector \
  --active-experts 3 --mirror-read-hint nocache --pipelined-verification --cooperative-quant \
  --interactive --max-new 768 --max-context 2048
```

Use `--active-experts 6` or omit the option to return to native routing. Close the old chat before launching another profile. The original SD pack/GGUF and SSD mirror are not converted or replaced by this experiment.

## Reproducible pilot

```sh
Scripts/particle build -c release --product ParticleCLI
Scripts/particle-speed-quality-check \
  --profiles 6,3 --parity-profiles 6,3 \
  --mirror-read-hint nocache --pipelined-verification --cooperative-quant
```

The helper refuses to contend with an active model/test process. It checks each budget's scalar/batched/continuation parity **against the same approximate budget**, then launches one runner to compare the requested profiles (six/four/three/two by default). Kernels and storage are held fixed; each profile/question clears KV, and profile order reverses on alternate rounds. RAM and OS caches are retained. Exact commands, configuration, source fingerprints, power, swap and memory state accompany the results.

The raw decode diagnostic processes 64 forwards per pass by default, including stop IDs; it is not a natural-completion speed test. A balanced twenty-question four-choice screen spans arithmetic, Python/JSON, science and Hindi/Tamil/Telugu/Bengali. It scores next-label likelihood rather than generating reasoning. Reports include accuracy, four-choice log loss, category labels, a 95% Wilson interval, and accuracy relative to the six-expert baseline. A pilot quality flag requires at least 75% of native accuracy and a native score of at least 50%; it is **not** a certification of a model-wide 25% loss ceiling. The small sample has substantial uncertainty and cannot measure long-answer instruction following or code execution quality. Reports always retain `qualified: false`.

Four tokens/s must ultimately be measured on varied natural chat/code/multilingual completions, at least 256 actual post-prefill decode steps per tested workload, longer histories, and a sustained run without growing swap. Power mode and AC/battery state must be disclosed. Repetition, capped reasoning, or continued stop IDs cannot qualify useful output. Native/approximate quality comparisons must include held-out task checks, generated-code/JSON validity, instruction following and long-output degeneration. No numerical parity with native-six output is expected for a reduced graph.

## Research and choice of first experiment

- [Don't Skip the Baseline (20 October 2026)](https://arxiv.org/html/2610.28369v1) compares static expert-count reduction with dynamic methods on Gemma/Qwen families. Static reductions are a strong baseline; this is a reason to measure a simple budget first, not evidence for Sarvam's loss or M2 speed.
- [MC-MoE](https://arxiv.org/html/2410.06270v2) combines expert-specific mixed precision with pruning and token protection. Its reported quality results do not transfer to a previously quantized Sarvam checkpoint.
- [Automated Fine-Grained MoE Quantization](https://aclanthology.org/2025.findings-acl.1386/) reports differing projection sensitivity, including higher precision needs in down-projections. A mixed gate/up/down precision pilot is preferable to assuming all matrices tolerate the same bit count.
- [ACE](https://arxiv.org/html/2609.05228v1) estimates expert contributions using checkpoint statistics and router-conditioned responses. Any such selector must beat static budgets at the same measured expert/read cost on held-out Sarvam tasks.
- [GGML quantization reference](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c) is the independent reference for a later Q3_K/Q2_K converter and Metal implementation. Current runtime support remains native Q4_K/Q5_K/Q6_K; lower-bit packing is not shipped by the expert-count option.

If no budget meets both speed and quality, the next stage is a separately identified mixed-precision expert derivative with bounded conversion, explicit quantization lineage and held-out evaluation. Re-quantizing the existing Q4_K_M adds irreversible error; it must not be described as conversion from FP16. Cache/I/O verification overlap and exact-kernel work remain compatible ways to reduce how aggressively the graph must be approximated.

## Validation artifacts

Current build/tests and pilot artifacts are kept under `docs/validation/particle-speed-quality-20261025/`. These distinguish implemented controls and component checks from whole-model qualification. The workspace is a source snapshot without Git metadata, so benchmark manifests record source hashes.

## First pilot result and exact I/O follow-up

The [first pilot](validation/particle-speed-quality-20261025/pilot-1/README.md) did not reach four tokens/s. Three experts retained 14/20 versus native 15/20 but reached median 2.617 tokens/s. Two experts reached 3.063 tokens/s and fell to 6/20, including 0/5 code questions, so it is rejected. Swap use and timing drift grew substantially. Raw arithmetic letter scoring was weak even for native mode; generated-answer reasoning remains unqualified.

Two additional opt-in controls target this bottleneck without changing the selected graph:

- `--mirror-read-hint nocache` applies `F_NOCACHE` to demand-loaded expert file descriptors only. Resident common weights retain their normal mapped-file policy. The default remains `default`.
- `--pipelined-verification` uses two bounded verifier workers while readers fetch other required experts into their reserved Metal slots. The transaction waits for all verifiers/fallbacks before new experts can reach the GPU or buffers can be reused. SHA-256 remains mandatory; damaged data uses the existing verified SD fallback. The default remains inline verification.

Both require mirror policy. They add no model copy and no second set of expert buffers. Their component tests include cancelled transactions, corrupted mirror bytes, intentionally corrupted unverified buffers and subsequent slot reuse. The [repeat native/three-expert pilot](validation/particle-speed-quality-20261025/pilot-pipeline-nocache/README.md) measured median 2.382/3.531 tokens/s respectively, with identical screen predictions and log losses to the earlier run. Three experts improved from 2.617 to 3.531 tokens/s on that diagnostic, but neither met four; these controls remain opt-in.

## Measured selective cooperative candidate

`--cooperative-quant` shares block scale/minimum metadata across SIMD lanes for Q4_K/Q5_K projections and fused expert paths. Native FP32 weight values, lane accumulation and six-rank reduction conventions are preserved; Q6_K retains the original vector path because its cooperative experiment was slower. Use it with `--projection vector --ffn-vector`; it defaults off. Independent CPU unpacking and production-shaped scalar/batched/fused/grouped GPU comparisons pass, including exact FP32 bits.

The [new natural-code check](validation/particle-speed-quality-20261025/natural-cooperative/summary.json) recorded **4.198 and 4.298 generated decode tokens/s**, 347 actual post-prefill forwards in each response, both stopping naturally. Weighted rate: **4.247 tokens/s over 694 forwards**. The final `is_even` function bodies pass 11 integer cases each under a restricted AST grader that executes no generated Python code. The responses still include reasoning and Markdown; they do not satisfy a strict code-only formatting requirement.

This is an experimental **three-expert graph**, not a new native-six speed claim. The prior small choice screen was 14/20 versus native 15/20, and exact I/O tuning preserved its predictions/log losses. Neither that screen nor two repeated code answers certifies a global 25% quality-loss ceiling or four tokens/s for all tasks/long contexts. Native-six remains the default. Mixed English/Hindi conversation and longer-context/sustained tests are reported separately.

## Mixed conversation result: remaining limits

The [English/Hindi multi-turn check](validation/particle-speed-quality-20261025/natural-mixed/README.md) measured **3.711 and 3.563 tokens/s** (weighted 3.635 over 1534 decode forwards). Both reached the 768-token limit during reasoning; neither delivered its requested two-sentence final answer. These are failed useful-output checks. The candidate has therefore **not met four tokens/s across workloads**.

On the second turn only 20 KV tokens matched; official formatting of the earlier capped assistant turn required replay of 772 historical tokens, plus 25 new tokens. Prefill took 58.79 seconds. Tokenization reuse is recorded separately and does not eliminate this evaluation. No historical context was dropped. Queue-inclusive first-token timing from pre-submitted harness input is not a typing-to-answer latency measurement.

The next qualification work must address reasoning/format compliance and historical suffix replay alongside held-out quality tests. Any later mixed-precision derivative should preserve the down-projection initially, record its already-quantized source lineage, and be compared against the current three-expert baseline at matched measured read cost. No global 25% quality-loss or all-workload four-token/s claim is currently justified.
