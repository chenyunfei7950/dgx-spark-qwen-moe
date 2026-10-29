# dgx-spark-qwen-moe

Run **Qwen3.6-35B-A3B** or **Sage-A1.5** on a DGX Spark (GB10) with image input and MTP drafting,
plus the retrieval pair (Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B) that usually sits next to them.

## Effect

The stock TensorFold Python engine serves the `qwen3_5_moe` family on CUDA but has no image path for it;
vLLM serves images but is 2.5× slower per stream. This repo is the TensorFold engine with the vision patch
applied: single-stream generation goes from 44 tok/s (vLLM FP8) to 112 tok/s with MTP drafting, an agentic RAG
loop drops from 90 s to 49 s per question, and images are read correctly (text in a picture answered in 2.2 s).
At eight concurrent requests the box sustains ~450 tok/s aggregate for Qwen3.6 and ~400 tok/s for Sage-A1.5.
Prefill is linear at roughly 0.19 ms per prompt token — a 55K-token prompt reaches its first token in ~10.7 s,
and the two models prefill alike because they share the engine.

![single-stream generation](assets/single-stream.png)

![throughput vs concurrency](assets/concurrency.png)

![prefill latency vs prompt length](assets/prefill.png)

## Quality (end-to-end, 2026-10-24)

Accuracy of an agentic RAG pipeline (AnGIneer) that uses this box as its LLM backend — same
retrieval stack, same engine, same judge, same day. Sage-A1.5 is an agentic finetune of the
same Qwen3.6-35B-A3B base that this repo also serves, so the deltas below are what the finetune
buys and costs in a real RAG product.

| Suite | Sage-A1.5 | Qwen3.6-35B-A3B | Note |
|---|---|---|---|
| OpenRAGBench subset, 1040 q | **90.2%** | 88.5% | Qwen: mean of 8 nightly runs (range 87.7–89.2) |
| FinanceBench, 150 q | **64.7%** | 49.5% | Qwen: mean of 7 runs (45.3–53.3); largest stable gain |
| Refusal recognition, 33 q | 69.7% | **78.4%** | Qwen: mean of 7 runs; the finetune answers more, refuses less |
| Port & waterway exam, 50 q | **54.0%** | 40.0% | same-day paired single runs; Qwen's own day-to-day spread is 40–60% |
| Professional QA, 30 q | 73.3% | **76.7%** | same-day paired single runs; 3 flips, net 1 question |

The two small suites are single paired runs — expect ±3–4 questions of noise per run, so treat
them as "no meaningful difference" rather than a ranking. The stable signal across repeated
runs: Sage-A1.5 is materially better at answer-quality suites (finance +15 pp), slightly better
on the main board, and gives back refusal recognition (−9 pp) — the known trade-off of an
agentic finetune. L3 multi-step calculation questions fail under both models.

![quality comparison](assets/quality.png)

## Models

| Model | Weights | Size |
|---|---|---|
| Qwen3.6-35B-A3B (4-bit, ships the MTP head) | https://huggingface.co/TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP | 20.9 GB |
| Sage-A1.5 (4-bit) | https://huggingface.co/Duckets/Sage-A1.5-MLX-4bit | 20.4 GB |
| Qwen3-Embedding-0.6B | https://huggingface.co/Qwen/Qwen3-Embedding-0.6B | 1.2 GB |
| Qwen3-Reranker-0.6B (as classifier head) | https://huggingface.co/Qwen/Qwen3-Reranker-0.6B | 1.2 GB |

The MTP drafter is not a separate download: it is a 475 MB `mtp-4bit.safetensors` that ships inside the
Qwen3.6 repository (step 3 downloads that single file, 475 MB — no need for the full 20.9 GB checkpoint). The
two heads correlate 0.98–0.99 after dequantization, which is why the Qwen3.6 file drafts for Sage-A1.5 as well.

## Usage

```bash
# 1. image (the engine ships in src/tensorfold)
docker pull nvcr.io/nvidia/pytorch:26.07-py3
cat > Dockerfile <<'DOCKER'
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE}
COPY src/tensorfold /usr/local/lib/python3.12/dist-packages/tensorfold
RUN pip install --no-cache-dir "transformers>=5.3,<6" huggingface_hub anyio click filelock fsspec
ENV TENSORFOLD_NO_UPDATE_CHECK=1
DOCKER
docker build -t dgx-spark-tensorfold:0.6.2-moe-vision .
# custom base: docker build --build-arg BASE_IMAGE=<image:tag> -t dgx-spark-tensorfold:0.6.2-moe-vision .

# 2. weights (mainland China: export HF_ENDPOINT=https://hf-mirror.com)
hf download Duckets/Sage-A1.5-MLX-4bit --local-dir /models/Sage-A1.5-MLX-4bit
hf download TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP --local-dir /models/Qwen3.6-35B-A3B-MLX-4bit-MTP

# 3. two fixes the conversions need: stop tokens + the MTP drafter
python3 - <<'PY'
import json; p="/models/Sage-A1.5-MLX-4bit/generation_config.json"
c=json.load(open(p)); c.update(bos_token_id=248044, eos_token_id=[248046,248044], pad_token_id=248044)
json.dump(c, open(p,"w"), indent=2)
PY
# the MTP drafter: one 475 MB file, taken from the Qwen3.6 repository
curl -L -o /models/Sage-A1.5-MLX-4bit/mtp-4bit.safetensors https://huggingface.co/TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP/resolve/main/mtp-4bit.safetensors
# mainland China mirror:
curl -L -o /models/Sage-A1.5-MLX-4bit/mtp-4bit.safetensors https://hf-mirror.com/TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP/resolve/main/mtp-4bit.safetensors

# 4. serve
docker run -d --name sage --gpus all --network host --ipc host \
  -v /models:/models:ro -v /cache:/root/.cache \
  -e TENSORFOLD_NO_UPDATE_CHECK=1 dgx-spark-tensorfold:0.6.2-moe-vision \
  tensorfold serve /models/Sage-A1.5-MLX-4bit --name sage-a1.5 --host 0.0.0.0 --port 18892 \
    --parallel 8 --kv-dtype bf16 --no-thinking --vision --vision-max-images 4 \
    --context 126976 --prompt-cache-gib 8
```

Verify: `curl -s localhost:18892/v1/models`, then a chat completion; images go in as
`{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`.

Notes: tested on DGX Spark GB10 (121 GB unified memory, driver 580.173.02). 126976 is the context that fits
with a second 35B model resident — if startup refuses it, take the value suggested in the
`estimated largest fitting ... window` message instead. `--no-drafts` runs without the MTP head. The NGC base
image ships Pillow/Jinja2/numpy but not transformers: the pip line above adds it (5.18 tested) — without it
text serving works but the first image request dies with ModuleNotFoundError. A `Driver Release 610.43` ERROR
banner at startup is cosmetic; the real failure mode is the context-fit line. The native 1.1.2 build measured
66 tok/s single-stream and 334 tok/s at eight concurrent requests on an idle box (no effective MTP drafting)
and rejects image input, so it does not replace this path yet.

## Retrieval stack (embedding + reranker)

Both are stock vLLM pooling models — no custom image. The recipe pins `vllm/vllm-openai:v0.32.0`, the same
version the author's production runs, so vectors and scores are reproducible. Cross-version acceptance against
a 0.28.0 deployment measured cosine 1.000000 on embeddings and identical rerank scores to 4 decimals.

```bash
hf download Qwen/Qwen3-Embedding-0.6B --local-dir /models/Qwen3-Embedding-0.6B
hf download Qwen/Qwen3-Reranker-0.6B --local-dir /models/Qwen3-Reranker-0.6B

# the reranker ships as a causal LM; vLLM pooling/classify needs a 2-way-softmax classifier head:
# symlink the checkpoint files and patch five config fields
python3 - <<'PY'
import json, os
base, dst = "/models/Qwen3-Reranker-0.6B", "/models/Qwen3-Reranker-0.6B-seqcls"
os.makedirs(dst, exist_ok=True)
cfg = json.load(open(f"{base}/config.json"))
cfg.update({"architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
            "method": "from_2_way_softmax", "num_labels": 1})
json.dump(cfg, open(f"{dst}/config.json", "w"), indent=2)
for name in os.listdir(base):
    if name != "config.json" and not os.path.exists(f"{dst}/{name}"):
        os.symlink(f"../{base}/{name}", f"{dst}/{name}")  # relative: survives container mounts
PY

docker run -d --name embed --gpus all --network host --ipc host \
  -v /models:/models:ro --restart unless-stopped vllm/vllm-openai:v0.32.0 \
  /models/Qwen3-Embedding-0.6B --served-model-name qwen3-embedding \
    --convert embed --pooler-config '{"pooling_type":"LAST"}' \
    --max-model-len 16384 --max-num-seqs 10 --gpu-memory-utilization 0.04

docker run -d --name rerank --gpus all --network host --ipc host \
  -v /models:/models:ro --restart unless-stopped vllm/vllm-openai:v0.32.0 \
  /models/Qwen3-Reranker-0.6B-seqcls --served-model-name qwen3-reranker \
    --runner pooling --convert classify --dtype float16 --enforce-eager \
    --max-model-len 4096 --max-num-seqs 32 --gpu-memory-utilization 0.085 \
    --chat-template /models/Qwen3-Reranker-0.6B-seqcls/chat_template.jinja
```

Verify: `bash deploy/check-retrieval.sh` (embedding dims + rerank scores for a fixed pair).

Notes: the image's entrypoint is already `vllm serve` — the docker run commands pass the model path
directly, do not prepend `serve` (a double `serve serve` fails instantly). Reranker needs `--enforce-eager`:
v0.32 enables PIECEWISE cudagraphs for pooling models and the capture OOMs inside the small memory budget.
The `gpu-memory-utilization`
caps are tuned for coexistence with the two 35B engines (~53 GB total engine footprint on a 121 GB box) —
raise them when running the retrieval pair alone. The 4B variants of both models exist upstream; the 0.6B
pair keeps retrieval latency low at ~5 GB combined. The author's production puts a rate-limiting proxy in
front of this pair; that glue is intentionally not part of this repo.

All four models in one shot: `bash deploy/bootstrap.sh && docker compose -f deploy/docker-compose.yml up -d`.

## Credits

Based on **TensorFold v0.6.5** (Python engine, frozen upstream at 0.6.6), Apache-2.0. Changes in `src/tensorfold`:
the `qwen3_5_moe` vision patch on CUDA — encoder/processing wiring plus `--vision` in `serve_options.py` and the
`families/qwen3_5_moe` decode/graph paths — and the deployment recipe that goes with it (checkpoint fixes,
context budgeting, two-model layout). Unofficial; not endorsed by the TensorFold authors.

## License

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE).
