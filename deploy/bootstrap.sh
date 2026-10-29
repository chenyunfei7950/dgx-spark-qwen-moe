#!/usr/bin/env bash
# One-time prep for the four-model stack: download weights (~44 GB), apply the two
# checkpoint fixes (Sage stop tokens, MTP drafter) and build the reranker classifier
# head, then build the TensorFold image. Afterwards:
#   docker compose -f deploy/docker-compose.yml up -d
# Mainland China: hf-mirror is the default endpoint; set HF_ENDPOINT to override.
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
mkdir -p models

echo "== 1/4 weights: LLM engines =="
hf download TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP --local-dir models/Qwen3.6-35B-A3B-MLX-4bit-MTP
hf download Duckets/Sage-A1.5-MLX-4bit --local-dir models/Sage-A1.5-MLX-4bit
# the MTP drafter ships inside the Qwen3.6 repository — nothing extra to download

echo "== 2/4 fix: Sage stop tokens (missing eos/bos/pad make the engine never stop) =="
python3 - <<'PY'
import json
p = "models/Sage-A1.5-MLX-4bit/generation_config.json"
c = json.load(open(p))
c.update(bos_token_id=248044, eos_token_id=[248046, 248044], pad_token_id=248044)
json.dump(c, open(p, "w"), indent=2)
print("patched", p)
PY

echo "== 3/4 weights + head: retrieval pair =="
hf download Qwen/Qwen3-Embedding-0.6B --local-dir models/Qwen3-Embedding-0.6B
hf download Qwen/Qwen3-Reranker-0.6B --local-dir models/Qwen3-Reranker-0.6B
# Qwen3-Reranker ships as a causal LM; vLLM pooling/classify needs a sequence-classification
# head: keep the checkpoint files (symlinks) and patch five config fields. This is the
# documented from_2_way_softmax recipe (no/yes token logits folded into one score).
python3 - <<'PY'
import json, os
base, dst = "models/Qwen3-Reranker-0.6B", "models/Qwen3-Reranker-0.6B-seqcls"
os.makedirs(dst, exist_ok=True)
cfg = json.load(open(os.path.join(base, "config.json")))
cfg.update({
    "architectures": ["Qwen3ForSequenceClassification"],
    "classifier_from_token": ["no", "yes"],
    "is_original_qwen3_reranker": True,
    "method": "from_2_way_softmax",
    "num_labels": 1,
})
json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
for name in os.listdir(base):
    if name == "config.json":
        continue
    link = os.path.join(dst, name)
    if not os.path.exists(link):
        # relative link: survives container mounts that relocate the models dir
        os.symlink(os.path.join("..", base, name), link)
print("classifier head ready:", dst)
PY

echo "== 4/4 build the TensorFold image =="
docker build -t dgx-spark-tensorfold:0.6.2-moe-vision .

echo
echo "done. start the stack:"
echo "  docker compose -f deploy/docker-compose.yml up -d"
echo "  bash deploy/check-retrieval.sh"
