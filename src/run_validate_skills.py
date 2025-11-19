from __future__ import annotations

import pandas as pd
from openai import OpenAI

from .config import INPUT_SKILLS_CSV, OUTPUT_SKILLS_CSV, CHECKPOINT_SKILLS_CSV
from .skills_tools import load_skills_dataframe, apply_structural_rules, save_skills_dataframe
from .semantic_agent import apply_semantic_evaluation


def main() -> None:
    # 1. Load from checkpoint if present, else start from raw input.
    if CHECKPOINT_SKILLS_CSV.exists():
        print(f"Resuming from checkpoint {CHECKPOINT_SKILLS_CSV} ...")
        df = load_skills_dataframe(CHECKPOINT_SKILLS_CSV)
    else:
        print(f"Loading skills from {INPUT_SKILLS_CSV} ...")
        df = load_skills_dataframe(INPUT_SKILLS_CSV)
        print(f"Loaded {len(df)} skills")

        print("Applying structural validation rules ...")
        df = apply_structural_rules(df)

        num_invalid = (df["structural_valid"] == False).sum()  # noqa: E712
        print(f"Structural invalid rows: {num_invalid}")
        print("Initial structural validation done.")

    print("Connecting to OpenAI for semantic evaluation ...")
    client = OpenAI()

    # 2. Checkpoint function, called after each semantic batch.
    def checkpoint(df_checkpoint: pd.DataFrame) -> None:
        print("Saving checkpoint ...")
        save_skills_dataframe(df_checkpoint, CHECKPOINT_SKILLS_CSV)

    df = apply_semantic_evaluation(df, client, checkpoint_fn=checkpoint)

    # Simple summary
    semantic_counts = df["semantic_status"].value_counts(dropna=False)
    print("Semantic status counts:")
    print(semantic_counts)

    print(f"Saving validated skills to {OUTPUT_SKILLS_CSV} ...")
    save_skills_dataframe(df, OUTPUT_SKILLS_CSV)
    print("Done.")

    # Remove checkpoint once the final file is safely written.
    # if CHECKPOINT_SKILLS_CSV.exists():
    #     CHECKPOINT_SKILLS_CSV.unlink()

    avg_ai_risk = pd.to_numeric(df["aiAutomationRisk_suggested"], errors="coerce").mean()
    avg_demand = pd.to_numeric(df["skillDemandScore_suggested"], errors="coerce").mean()
    print(f"Average suggested AI risk: {avg_ai_risk:.2f}")
    print(f"Average suggested demand score: {avg_demand:.1f}")


if __name__ == "__main__":
    main()
