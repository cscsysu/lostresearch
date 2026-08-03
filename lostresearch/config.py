"""Configuration for InfoDyn pilot experiment."""
import os

# ============ 模型 ============
MODEL_PATH = "/data2/css2025/models/Qwen/Qwen3-8B"
MODEL_NAME = "Qwen3-8B"
# GPU 选择: 服务器上 nvidia-smi 显示 5 张卡:
#   index 0 = A40 (46GB)  ← 选这个
#   index 1-4 = RTX 3090 (24GB)
# 用 CUDA_VISIBLE_DEVICES=0 选 A40, 此时在程序内它就是 cuda:0
# run_pilot.sh 会设置 CUDA_VISIBLE_DEVICES, 这里 device 用 cuda:0
DEVICE = "cuda:0"
DTYPE = "bfloat16"

# ============ Thinking mode ============
# Qwen3 默认开 thinking, 必须显式关闭
ENABLE_THINKING = False

# ============ 数据 ============
DATASET_NAME = "mandarjoshi/trivia_qa"  # HuggingFace 要求 namespace/name
DATASET_CONFIG = "unfiltered.nocontext"
NUM_SAMPLES = 10  # 先跑 10 题验证 token 对齐, 通过后再扩展到 50+
MAX_PROMPT_LEN = 512  # 截断超长 prompt

# ============ 生成 ============
MAX_NEW_TOKENS = 32
DO_SAMPLE = False  # greedy, 可复现

# ============ 读出方式 ============
# 三种读出方式, pilot 阶段先做两种
USE_RAW_LOGIT_LENS = True      # 直接用最终 LM head 解码中间层
USE_TUNED_LENS = False          # 需要训练 translator, pilot 阶段先不做
USE_CANDIDATE_SCORING = True    # 只算答案 token 的 logit

# ============ 答案处理 ============
MAX_ANSWER_TOKENS = 10  # 多 token 答案最多算前 10 个 token
USE_TEACHER_FORCING = True  # 用 teacher forcing 算多 token 答案序列概率

# ============ 输出 ============
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
