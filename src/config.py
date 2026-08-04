import os
import torch

# 1. Random Seed for Reproducibility
ROLL_NUMBER = 55 
torch.manual_seed(ROLL_NUMBER)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(ROLL_NUMBER)

# 2. Hardware Device Selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Directory Paths (Relative to project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "fra.txt")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 4. Dataset Filtering Rules (Constants used by dataset.py)
MAX_LENGTH = 10
ENG_PREFIXES = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

# 5. Training Hyperparameters
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 1
DROPOUT = 0.1
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 8