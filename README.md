# Agentic Pipeline for ESCO Skills Data Labelling

This project runs an **agentic enrichment pipeline** over ESCO-based skills data.

It:

- Loads raw skills from CSV.
- Applies **structural validation** and flags invalid rows.
- Uses an **LLM (OpenAI)** to semantically enrich and normalise metrics such as:
	- `aiAutomationRisk_suggested`
	- `skillDemandScore_suggested`
	- `skillTrend_suggested`
	- `difficulty_level_suggested`
	- `career_level_suggested`
	- `learning_time_estimate_hours_suggested`
	- `learning_resources_suggested`
	- `verification_method_suggested`
	- `assessment_type_suggested`
- Writes a **checkpoint CSV** so long runs can be resumed.
- Saves a timestamped final `skills_validated_YYYYMMDD_HHMM.csv`.

---

## Project Structure

```text
.
├─ data/
│  ├─ skills_input.csv                 # Raw input skills
│  ├─ skills_validated_checkpoint.csv  # Checkpoint for resumable runs
│  └─ skills_validated_*.csv           # Timestamped outputs
├─ src/
│  ├─ config.py                        # Paths and config (batch sizes, model, etc.)
│  ├─ run_validate_skills.py           # Main entry point
│  ├─ skills_tools.py                  # CSV I/O + structural validation
│  └─ semantic_agent.py                # LLM-based semantic enrichment
├─ .env.example                        # Example environment configuration
├─ requirements.txt                    # Python dependencies
└─ README.md
```

---

## Setup

1. **Create and activate a virtualenv (optional but recommended)**

```cmd
cd esco-skill-validator-workflow
python -m venv .venv
.venv\Scripts\activate
```

2. **Install dependencies**

```cmd
pip install -r requirements.txt
```

3. **Configure environment variables**

Create a `.env` file at the project root (alongside `.env.example`):

```bash
cp .env.example .env
```

Then edit `.env` and set at least:

```env
OPENAI_API_KEY=sk-...
# Optional: override defaults
# OPENAI_MODEL_NAME=gpt-4o-mini-2024-11-20
# SEMANTIC_BATCH_SIZE=60
# MAX_SEMANTIC_ROWS=240
# SEMANTIC_CONCURRENCY=2
```

Never commit your real `.env` – it should be in `.gitignore`.

---

## Running the Pipeline

From the project root:

```cmd
python -m src.run_validate_skills
```

The pipeline will:

1. If `data/skills_validated_checkpoint.csv` exists, **resume** from it.  
	 Otherwise, load `data/skills_input.csv` and run structural checks.
2. Connect to OpenAI and run semantic enrichment in **batches**, with optional **concurrency**.
3. After each semantic batch, write an updated checkpoint to
	 `data/skills_validated_checkpoint.csv`.
4. At the end, write a final file like:

```text
data/skills_validated_YYYYMMDD_HHMM.csv
```

and print basic summary stats (e.g. semantic status counts, average risk/demand).

---

## Configuration

All core knobs live in `src/config.py` and can be overridden via environment variables:

- `INPUT_SKILLS_CSV`  
	Path to input skills CSV (default: `data/skills_input.csv`).

- `CHECKPOINT_SKILLS_CSV`  
	Resumable checkpoint file (default: `data/skills_validated_checkpoint.csv`).

- `OUTPUT_SKILLS_CSV`  
	Timestamped output path (auto-generated per run).

- `SEMANTIC_BATCH_SIZE`  
	Number of rows per LLM call (e.g. `60`).

- `MAX_SEMANTIC_ROWS`  
	Cap on number of rows to semantically process (e.g. `240` for testing, `None` for all).

- `SEMANTIC_CONCURRENCY`  
	Number of batches in flight at once:
	- `1` = sequential (easiest to debug).
	- `2–4` = parallel API calls for speed.

---

## Checkpointing & Resuming

- A long run can be interrupted safely (Ctrl+C).
- Next run will resume from `data/skills_validated_checkpoint.csv`, **skipping rows**
	that already have a non-empty `semantic_status`.
- If you want to **restart from scratch**, delete the checkpoint:

```cmd
del data\skills_validated_checkpoint.csv
```

(or remove it via your file explorer) and run the pipeline again.

---

## Notes on Cost and Batching

- You are billed per **token**, not per request.
- Larger `SEMANTIC_BATCH_SIZE` values are slightly more **cost-efficient per row**
	(less repeated prompt overhead), but each call is heavier and slower.
- Smaller batches are marginally more expensive overall but often **more reliable**
	and easier to debug.
- Adjust `SEMANTIC_CONCURRENCY` based on your rate limits and how aggressively you
	want to parallelise API calls.

---

## Development Tips

- Keep an eye on `stdout` logs (`[semantic] ...`, `[semantic-debug] ...`) to
	understand batching and model behaviour.
- When testing:
	- Set `MAX_SEMANTIC_ROWS` to a small number (e.g. `30`).
	- Use `SEMANTIC_BATCH_SIZE` around `10–20`.
	- Start with `SEMANTIC_CONCURRENCY=1`, then increase once stable.
