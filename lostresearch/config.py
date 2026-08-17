"""Configuration for InfoDyn full experiment on Qwen3-8B."""
import os

# ============ 模型 ============
# 默认路径，可通过环境变量 INFODYN_MODEL_PATH 覆盖
import os as _os
MODEL_PATH = _os.environ.get("INFODYN_MODEL_PATH", "/data2/css2025/models/Qwen/Qwen3-8B")
MODEL_NAME = "Qwen3-8B"
# 服务器 GPU 映射 (CUDA ordinal):
#   0 = A40 (46GB)  ← 选这个
#   1-4 = RTX 3090 (24GB)
DEVICE = "cuda:0"
DTYPE = "bfloat16"

# ============ Thinking mode ============
ENABLE_THINKING = False

# ============ 数据 ============
# 正式规模: 7 任务覆盖知识/多跳/数学/常识/科学/长文本/世界知识
DATASETS = [
    {"name": "mandarjoshi/trivia_qa", "config": "unfiltered.nocontext",
     "split": "validation", "n": 500, "label": "triviaqa"},
    {"name": "hotpotqa/hotpot_qa", "config": "distractor",
     "split": "validation", "n": 250, "label": "hotpotqa"},
    {"name": "openai/gsm8k", "config": "main",
     "split": "test", "n": 250, "label": "gsm8k"},
    {"name": "tau/commonsense_qa", "config": None,
     "split": "validation", "n": 250, "label": "commonsenseqa"},
    {"name": "allenai/ai2_arc", "config": "ARC-Challenge",
     "split": "test", "n": 250, "label": "arc_challenge"},
    {"name": "emozilla/quality", "config": None,
     "split": "validation", "n": 200, "label": "quality"},
    {"name": "cais/mmlu", "config": None,
     "split": "test", "n": 200, "label": "mmlu"},
]
MAX_PROMPT_LEN = 512

# ============ 生成 ============
MAX_NEW_TOKENS = 32
DO_SAMPLE = False

# ============ 答案处理 ============
MAX_ANSWER_TOKENS = 10

# ============ 信号丢失判据 ============
# 真正的 "信号丢失": 中间层正确答案有竞争力, 但最终层被压过
RANK_COMPETITIVE = 5    # 中间层 rank <= 5 算 "有竞争力"
RANK_LOST = 10          # 最终层 rank > 10 算 "丢失"
PROB_EMERGENCE = 0.05  # 中间层 prob > 0.05 算 "信号出现"
PROB_DECAY = 0.01       # 最终层 prob < 0.01 算 "信号衰减"

# ============ 预测任务 ============
PREDICTION_T0 = 0.5  # 在 50% 深度处预测
PREDICTION_FEATURES = ["logprob_at_t0", "slope_before_t0", "rank_at_t0",
                        "max_logprob_before_t0", "transitions_before_t0"]

# ============ 输出 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")

for d in [OUTPUT_DIR, FIGURE_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)
