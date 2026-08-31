#!/usr/bin/env bash
# Pre-flight for the m1-LODO run. Everything here is cheap; nothing after it is.
#
#   sbatch -p gpu --gres=gpu:h200:1 -c 16 --mem=96G -t 01:00:00 \
#     -o training_methods/m1_lodo/smoke_%j.log \
#     --wrap="bash training_methods/m1_lodo/smoke.sh"
#
# 1. GPU visible to the training stack
# 2. base model loads and generates
# 3. a 3-step LoRA train produces a loadable adapter (loss must be finite)
# 4. the trained adapter runs 2 real episodes through the teacherless agent loop
#
# Any failure exits non-zero, so this can gate the orchestrator.
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=${HF_HOME:-/shared/al-maamari/DeKIS/teacher-guidence/.hf_cache}
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HUB_CACHE"
OUT=${OUT:-training_methods/m1_lodo/smoke}
MODEL=${MODEL:-ibm-granite/granite-4.1-3b}
FOLD=${FOLD:-hotpotqa}
mkdir -p "$OUT"
echo "=== node $(hostname) | $(date -u +%FT%TZ) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo; echo "=== 1. training stack sees the GPU ==="
.venv_train/bin/python -c "
import torch; assert torch.cuda.is_available(), 'no CUDA'
print('torch', torch.__version__, '| gpus', torch.cuda.device_count(),
      '|', torch.cuda.get_device_name(0))"

echo; echo "=== 2. base model loads and generates ==="
.venv_train/bin/python - <<PYEOF
import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer
t0=time.time(); m="$MODEL"
tok=AutoTokenizer.from_pretrained(m)
mod=AutoModelForCausalLM.from_pretrained(m, dtype=torch.bfloat16, device_map="cuda:0")
msgs=[{"role":"user","content":'Reply with ONLY this JSON: {"ok": 1}'}]
# transformers 5.x returns a BatchEncoding here, not a bare tensor.
enc=tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                            return_dict=True).to("cuda:0")
n_in=enc["input_ids"].shape[1]
out=mod.generate(**enc, max_new_tokens=24, do_sample=False)
print("generated:", repr(tok.decode(out[0][n_in:], skip_special_tokens=True)[:60]))
print(f"loaded+generated in {time.time()-t0:.1f}s | weights {mod.get_memory_footprint()/2**30:.1f} GiB")
PYEOF

echo; echo "=== 3. 3-step LoRA train ==="
.venv_train/bin/python training_methods/m1_sft/train.py \
    --model "$MODEL" \
    --train-file "training_methods/m1_lodo/data/fold_${FOLD}/train.jsonl" \
    --dev-file  "training_methods/m1_lodo/data/fold_${FOLD}/dev.jsonl" \
    --out-base "$OUT/runs" --tag "smoke_${FOLD}" --smoke
ADAPTER=$(ls -td "$OUT"/runs/*_smoke_${FOLD}*/adapter 2>/dev/null | head -1)
test -d "$ADAPTER" || { echo "SMOKE FAIL: no adapter produced"; exit 1; }
echo "adapter: $ADAPTER"

echo; echo "=== 4. 2 real episodes with the trained adapter ==="
.venv_train/bin/python training_methods/common/eval_agent.py \
    --model "$MODEL" --adapter "$ADAPTER" \
    --questions "training_methods/m1_lodo/data/tests/heldin_${FOLD}_questions.jsonl" \
    --corpus "data/datasets/tg_v1/hotpotqa/hotpotqa_train_corpus.jsonl" \
    --out "$OUT/eval" --tag "smoke_${FOLD}" --limit 2 --budget 3 --hidden-budget \
    --temperature 0 --device cuda:0
M=$(ls -td "$OUT"/eval/*_smoke_${FOLD}*/metrics.json 2>/dev/null | head -1)
test -f "$M" || { echo "SMOKE FAIL: no eval metrics"; exit 1; }
.venv_train/bin/python -c "
import json,sys; m=json.load(open('$M'))
print('eval metrics:', {k:m.get(k) for k in ('n','em','f1','cover_match','doc_recall','mean_steps','tokens_per_episode','invalid_action_steps')})
assert m.get('n',0) >= 2, 'fewer than 2 episodes completed'"
echo; echo "=== SMOKE PASSED ==="
