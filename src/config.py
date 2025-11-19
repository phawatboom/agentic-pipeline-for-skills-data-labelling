import os
from pathlib import Path

from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Paths
DATA_DIR = BASE_DIR / "data"
INPUT_SKILLS_CSV = DATA_DIR / "skills_input.csv"

# Fixed checkpoint file used for resumable semantic evaluation
CHECKPOINT_SKILLS_CSV = DATA_DIR / "skills_validated_checkpoint.csv"

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Default
# OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini-2024-11-20")

# Development
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini-2024-11-20")

if OPENAI_API_KEY is None:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Create a .env file with OPENAI_API_KEY=..."
    )

# Semantic evaluation settings
# Batch size controls how many rows are sent to the model per call.
# 10 to 20 is usually a good trade off between context size and efficiency.
SEMANTIC_BATCH_SIZE = 20

# If you want to limit how many rows are semantically reviewed during testing,
# set MAX_SEMANTIC_ROWS to an integer. For full dataset, set it to None.
MAX_SEMANTIC_ROWS = 2000  # e.g. 500 for testing, None for all rows

# How many semantic API batches to run concurrently.
# 1 = sequential behaviour, >1 = parallel calls.
SEMANTIC_CONCURRENCY = int(os.getenv("SEMANTIC_CONCURRENCY", "4"))


# small test
# MAX_SEMANTIC_ROWS = 30          # review only first 30 rows semantically
# SEMANTIC_BATCH_SIZE = 10        # 10 rows per model call

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
OUTPUT_SKILLS_CSV = DATA_DIR / f"skills_validated_{timestamp}.csv"
