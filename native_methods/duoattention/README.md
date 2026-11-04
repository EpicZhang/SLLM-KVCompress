# Scheme A — Functional physical DuoAttention port in the locked TEAM environment

**Status:** prototype. CPU cache unit tests passed on the authoring machine. GPU model integration, accuracy, speed, and memory **have not yet been verified on the user's RunPod A100**. Never report this as the original authors' kernel or as an achieved paper result until independently validated.

This add-only package implements retrieval heads with full persistent KV and streaming heads with physical **sink + recent** KV. It uses a custom attention forward on top of Transformers **5.2.0 LlamaAttention**, Torch SDPA, and the team's fixed trained head pattern. Chunked prefill updates physical cache after every chunk; decode appends without retaining pruned KV.

This is a functional *port of the algorithm*, NOT a copy or validated drop-in of MIT's optimized FlashAttention/FlashInfer implementation. Performance may be **worse** than full attention because of two attention calls, head selection, mask construction and chunked processing. Physical KV bytes should be less; end-to-end peak memory and speed must be measured separately.

No file in `scripts/`, `baselines/`, `splits/`, `configs/`, `requirements.lock.txt` or `README.md` is modified. This method does **not** auto-push to GitHub. Results are separate under `results/*_duoattention_native_*`.

## On RunPod A100 — install

1. Upload `duoattention-native-team-cuda13.zip` to `/workspace/kvcompress/KVCompress-survey/` and unpack there:

   ```bash
   cd /workspace/kvcompress/KVCompress-survey
   unzip -o duoattention-native-team-cuda13.zip
   source .venv/bin/activate
   export HF_HOME=/workspace/kvcompress/hf
   ```

2. Inspect the unchanged environment and official head pattern:

   ```bash
   python -m native_methods.duoattention.preflight
   python -m unittest discover -s tests/native_duoattention -v
   ```

3. After the tiny Transformer integration test passes, run small **two-mode** physical cache smoke tests:

   ```bash
   python -m native_methods.duoattention.runner --mode full --smoke --chunk-size 128
   python -m native_methods.duoattention.runner --mode native --smoke --chunk-size 128
   ```

   `full` is a *paired, same-runner* baseline: **same torch SDPA, chunked prefill and prompting**, but all KV heads are retrieval and no physical cache is pruned. Compare `kv_cache_bytes`, `full_context_kv_bytes`, `physical_compression_ratio` and output. For numerical consistency, the tiny test checks that `full` matches ordinary Transformers forward for a random Llama.

4. Only after success, use the same pinned team RULER split and scoring, **one sample/task first**:

   ```bash
   python -m native_methods.duoattention.runner --mode full \
       --context-length 32768 --per-task 1 --chunk-size 512
   python -m native_methods.duoattention.runner --mode native \
       --context-length 32768 --per-task 1 --compression-ratio 0.5 --chunk-size 512
   ```

   This is a costly test: starts ~13 complete 32K runs. To test only one actual long-context task first, use `--smoke` or change a *copy* of the dataset script; do NOT alter common team scripts. After one-sample outputs are correct, extend both to `--per-task 50` for the team's 650 samples; runs resume from the same results directory.

   ```bash
   python -m native_methods.duoattention.runner --mode full --context-length 32768 --per-task 50 --chunk-size 512
   python -m native_methods.duoattention.runner --mode native --context-length 32768 --per-task 50 --compression-ratio 0.5 --chunk-size 512
   ```

## Controls and limitations

- **Version lock:** must be exact Transformers 5.2.0; other releases fail fast. Same `requirements.lock.txt` as H2O/SnapKV.
- **Model:** Llama-3.1-8B-Instruct only with provided **32 × 8** pretrained KV-head pattern. No head training is performed.
- **Batch:** one sample, no padding, BF16, greedy decode; single GPU.
- **Chunked prefill:** physical retained state is pruned **after each chunk**, not after full context. This is not the author's fused implementation.
- **Prompt/scoring:** invokes the exact installed `kvpress` pipeline's `preprocess()` for chat template and token IDs, and imports team's `eval_ruler.py` `score()` and `load_split()` with pinned dataset revision.
- **Decode:** uses manually supplied absolute `cache_position` and `use_cache=False` so HF does not allocate a second DynamicCache; native physical KV is kept solely in this method's cache. Must not call `model.generate()` with this patch.
- **Full baseline:** use `--mode full`, not the historical GCP baseline, when comparing TTFT/TPOT and overhead. For accuracy, compare both to the historical team baseline and flag any differences.
- **Metrics:** `physical_compression_ratio` = 1 - *stored physical KV tensor bytes* / theoretical full context KV bytes. **Not** CUDA peak memory reduction. `peak_memory_bytes` is measured separately. This method's SDPA temporary buffers may make peak memory unexpectedly high.
- **No automatic Git upload:** results only saved on `/workspace` network volume; no git push/PR.
- **No formal correctness/performance claim:** do not submit results until `full` output is close to team baseline, native logical behavior is checked, memory is actually reduced at 4K then 32K, and all 650 samples use exactly matching seeds and data.

If the tiny model test fails, stop there and share the full stack trace; do not run 32K and do not modify team lock versions.

References: MIT Han Lab DuoAttention paper (2025); [original implementation](https://github.com/mit-han-lab/duo-attention); pinned Transformers [v5.2.0 LlamaAttention](https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/llama/modeling_llama.py).
