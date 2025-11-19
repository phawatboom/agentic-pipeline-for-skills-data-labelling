from __future__ import annotations

import json
from typing import List, Dict, Any, Optional, Callable

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI

from .config import (
    OPENAI_MODEL_NAME,
    SEMANTIC_BATCH_SIZE,
    MAX_SEMANTIC_ROWS,
    SEMANTIC_CONCURRENCY,
)


def _apply_results_to_dataframe(df: pd.DataFrame, results: List[Dict[str, Any]]) -> None:
    """Apply a list of semantic results to the DataFrame in-place.

    Each result dict is expected to contain a ``row_index`` key used to
    match back to the correct row in ``df``.
    """

    for res in results:
        row_index = res.get("row_index")
        if row_index is None or row_index not in df.index:
            continue

        df.at[row_index, "semantic_status"] = res.get("semantic_status", "")
        df.at[row_index, "semantic_reason"] = res.get("semantic_reason", "")

        ai_suggested = res.get("aiAutomationRisk_suggested")
        if ai_suggested is not None:
            df.at[row_index, "aiAutomationRisk_suggested"] = ai_suggested

        demand_suggested = res.get("skillDemandScore_suggested")
        if demand_suggested is not None:
            df.at[row_index, "skillDemandScore_suggested"] = demand_suggested

        trend_suggested = res.get("skillTrend_suggested")
        if trend_suggested is not None:
            df.at[row_index, "skillTrend_suggested"] = trend_suggested

        difficulty_suggested = res.get("difficulty_level_suggested")
        if difficulty_suggested is not None:
            df.at[row_index, "difficulty_level_suggested"] = difficulty_suggested

        career_level_suggested = res.get("career_level_suggested")
        if career_level_suggested is not None:
            df.at[row_index, "career_level_suggested"] = career_level_suggested

        hours_suggested = res.get("learning_time_estimate_hours_suggested")
        if hours_suggested is not None:
            df.at[row_index, "learning_time_estimate_hours_suggested"] = hours_suggested

        resources_suggested = res.get("learning_resources_suggested")
        if resources_suggested is not None:
            df.at[row_index, "learning_resources_suggested"] = resources_suggested

        verification_suggested = res.get("verification_method_suggested")
        if verification_suggested is not None:
            df.at[row_index, "verification_method_suggested"] = verification_suggested

        assessment_suggested = res.get("assessment_type_suggested")
        if assessment_suggested is not None:
            df.at[row_index, "assessment_type_suggested"] = assessment_suggested


def build_semantic_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Build row dicts only for skills that still need semantic review.

    If a row already has a non-empty ``semantic_status``, it is treated as
    completed from a previous run and skipped. Each remaining row dict
    includes a ``row_index`` for matching back to the DataFrame.
    """
    rows: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        existing_status = str(row.get("semantic_status", "")).strip()
        if existing_status:
            # Already processed in a previous run
            continue

        rows.append(
            {
                "row_index": int(idx),
                "preferredLabel": str(row.get("preferredLabel", "")),
                "altLabels": str(row.get("altLabels", "")),
                "description": str(row.get("description", "")),
                "reuseLevel": str(row.get("reuseLevel", "")),
                "skillType_master": str(row.get("skillType_master", "")),
                "aiAutomationRisk": str(row.get("aiAutomationRisk", "")),
                "skillDemandScore": str(row.get("skillDemandScore", "")),
                "skillTrend": str(row.get("skillTrend", "")),
                "difficulty_level": str(row.get("difficulty_level", "")),
                "career_level": str(row.get("career_level", "")),
                "learning_time_estimate_hours": str(row.get("learning_time_estimate_hours", "")),
                "learning_resources": str(row.get("learning_resources", "")),
                "verification_method": str(row.get("verification_method", "")),
                "assessment_type": str(row.get("assessment_type", "")),
            }
        )
    return rows


def maybe_limit_rows_for_semantic(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Optionally limit the number of rows for semantic evaluation based on config.
    """
    if MAX_SEMANTIC_ROWS is not None and MAX_SEMANTIC_ROWS < len(rows):
        return rows[:MAX_SEMANTIC_ROWS]
    return rows


def chunk_rows(rows: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    """
    Split rows into batches for API calls.
    """
    return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]


def semantic_review_batch(
    client: OpenAI,
    batch_rows: List[Dict[str, Any]],
    model_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Send one batch of rows to the model and get semantic scores and suggestions.

    Expects the model to return a JSON object:
      {
        "results": [
          {
            "row_index": int,
            "semantic_status": "ok" | "questionable" | "inconsistent",
            "semantic_reason": "short text",
            "aiAutomationRisk_suggested": float | null,
            "skillDemandScore_suggested": int | null,
            "skillTrend_suggested": "growing" | "declining" | "mixed" | "stable" | null
            'learning_time_estimate_hours_suggested': float,
            'learning_resources_suggested': str,
            'verification_method_suggested':
                'SELF_REPORTED' | 'EVIDENCE_UPLOADED' | 'SUPERVISOR_OR_INSTITUTION_REF' | 'PLATFORM_VERIFIED',
            'assessment_type_suggested':
                'QUIZ_TEST' | 'PRACTICAL_TASK' | 'PROJECT_PORTFOLIO' | 'WORK_SAMPLE' | 'INTERVIEW_ASSESSMENT',
            'difficulty_level_suggested': str,
            'career_level_suggested': str
          },
          ...
        ]
      }
    """
    if model_name is None:
        model_name = OPENAI_MODEL_NAME

    # system_message = (
    #     "You are a data quality and enrichment agent for ESCO skills.\n"
    #     "You must validate, refine, and output consistent field values for each skill.\n\n"
    #     "### Fields to evaluate and adjust\n"
    #     "- aiAutomationRisk: float ∈ [0.0, 1.0]; higher = more likely automated.\n"
    #     "- skillDemandScore: integer ∈ [0, 100]; higher = more market demand.\n"
    #     "- skillTrend: one of {growing, stable, declining, mixed}.\n"
    #     "- difficulty_level: one of {FOUNDATION, WORKING, ADVANCED, EXPERT}.\n"
    #     "- career_level: one of {ENTRY, MID, SENIOR}.\n"
    #     "- learning_time_estimate_hours: approximate float/int hours to reach WORKING level.\n"
    #     "- learning_resources: short comma-separated descriptors (e.g. 'MOOC, project, mentorship').\n"
    #     "- verification_method: one of {SELF_REPORTED, EVIDENCE_UPLOADED, "
    #     "SUPERVISOR_OR_INSTITUTION_REF, PLATFORM_VERIFIED}.\n"
    #     "- assessment_type: one of {QUIZ_TEST, PRACTICAL_TASK, PROJECT_PORTFOLIO, "
    #     "WORK_SAMPLE, INTERVIEW_ASSESSMENT}.\n\n"
    #     "### What to do\n"
    #     "1. Review the provided fields and decide if they are reasonable.\n"
    #     "2. Always output *_suggested values, even if unchanged (copy current value if fine).\n"
    #     "3. Assign semantic_status as:\n"
    #     "   - 'ok' (values acceptable)\n"
    #     "   - 'adjusted' (you changed at least one)\n"
    #     "   - 'implausible' (clearly inconsistent; needs human review)\n"
    #     "4. Provide a short semantic_reason explaining your decision.\n\n"
    #     "### Output format (strict JSON)\n"
    #     "{ 'results': [ { 'row_index': int, 'semantic_status': str, 'semantic_reason': str, "
    #     "'aiAutomationRisk_suggested': float, 'skillDemandScore_suggested': int, "
    #     "'skillTrend_suggested': str, 'difficulty_level_suggested': str, "
    #     "'career_level_suggested': str, 'learning_time_estimate_hours_suggested': float, "
    #     "'learning_resources_suggested': str, 'verification_method_suggested': str, "
    #     "'assessment_type_suggested': str } ] }\n"
    # )

    system_message = """
        You are an ESCO skill enrichment and data quality agent.

        Your task is to review each skill row and fill in missing or implausible fields.
        Your outputs must be realistic, well calibrated, and reflect real world labour-market patterns.

        Base your reasoning on:
        - preferredLabel, altLabels, and description
        - skillType and reuseLevel (very important)
        - domain semantics inferred from the text

        If a description is missing or vague, rely more on preferredLabel, altLabels, skillType, and reuseLevel, and default to conservative estimates.

        Existing numeric values should be treated as priors. Keep them unless they clearly conflict with guidelines or are implausible.

        You may receive one or many skills at once. When multiple skills appear in a batch, apply batch-level distribution and variation rules.

        ----------------------------------------------------------------------
        ### Fields to evaluate and adjust
        ----------------------------------------------------------------------
        - aiAutomationRisk: float in [0.0, 1.0]
        - skillDemandScore: integer in [0, 100]
        - skillTrend: {growing, stable, declining, mixed}
        - difficulty_level: {FOUNDATION, WORKING, ADVANCED, EXPERT}
        - career_level: {ENTRY, MID, SENIOR}
        - learning_time_estimate_hours: integer hours to reach WORKING proficiency
        - learning_resources: comma separated descriptors (e.g. "MOOC, coding project, workshop")
        - verification_method: {SELF_REPORTED, EVIDENCE_UPLOADED, SUPERVISOR_OR_INSTITUTION_REF, PLATFORM_VERIFIED}
        - assessment_type: {QUIZ_TEST, PRACTICAL_TASK, PROJECT_PORTFOLIO, WORK_SAMPLE, INTERVIEW_ASSESSMENT}

        ----------------------------------------------------------------------
        ### Calibration guidelines (use as strong priors)
        ----------------------------------------------------------------------

        [Automation risk]
        Use these domain patterns:
        - Managerial, interpersonal, creative: **0.05–0.30**
        - Scientific, analytical, specialised: **0.25–0.50**
        - Routine lab, operational, inspection: **0.40–0.70**
        - Simple repetitive tasks: **0.60–0.90**
        Avoid repeating identical values across skills unless strongly justified.

        [Skill demand]
        Use skillType + reuseLevel explicitly:
        - Transversal skills (high reuseLevel): **60–90**
        - Broad sector-specific: **50–75**
        - Occupation-specific: **25–55**
        - Very niche / rare (low reuseLevel): **10–45**

        Distribution expectations (over large samples, not rigid per batch):
        - ~10–20 percent above **80**
        - ~10 percent below **20**
        Avoid clustering everything around **50–60**.

        [Skill trend]
        - Technology, sustainability, digital, healthcare: **growing**
        - Mature regulated domains (railway, correctional, compliance): **stable** or **mixed**
        - Obsolete or shrinking practices: **declining**

        [Difficulty level]
        - FOUNDATION: simple behaviours or basic concepts
        - WORKING: procedural, hands-on skills
        - ADVANCED: specialised multi-step expertise
        - EXPERT: scientific, regulated, or highly complex domains

        [Career level]
        - ENTRY: basic transversal or routine skills
        - MID: standard professional or technical work
        - SENIOR: supervisory, regulatory, specialised responsibilities

        [Learning time estimate (hours)]
        Use realistic ranges for WORKING proficiency:
        - FOUNDATION: **5–20h**
        - WORKING: **20–60h**
        - ADVANCED: **60–200h**
        - EXPERT: **200–500h**

        Numeric formatting:
        - aiAutomationRisk: up to 2 decimal places
        - Demand + learning_time: integers only

        Do not reuse the same values (e.g. always 40h or always 100h) unless genuinely appropriate.

        [Learning resources]
        Choose 2–3 based on domain:
        - Creative/performance: mentorship, rehearsal, workshop
        - Scientific/lab: university course, lab practice, certification
        - Compliance/regulatory: policy manual, field training, workshop
        - Technical/programming: MOOC, textbook, coding project
        - Social services: mentorship, supervised practice, applied workshop

        [Verification method]
        - SELF_REPORTED → soft transversal
        - SUPERVISOR_OR_INSTITUTION_REF → workplace/procedural
        - PLATFORM_VERIFIED → technical or certification linked
        - EVIDENCE_UPLOADED → portfolio/project-based

        [Assessment type]
        - QUIZ_TEST → conceptual knowledge
        - PRACTICAL_TASK → hands-on or inspection
        - PROJECT_PORTFOLIO → creative or technical outputs
        - INTERVIEW_ASSESSMENT → behavioural/interpersonal
        - WORK_SAMPLE → real or simulated workplace tasks

        ----------------------------------------------------------------------
        ### Consistency rules (very important)
        ----------------------------------------------------------------------
        Ensure the following relationships:

        - FOUNDATION difficulty should not have SENIOR career level.
        - EXPERT difficulty must not have ENTRY career level.
        - FOUNDATION difficulty must not have learning_time > 60.
        - EXPERT difficulty must not have learning_time < 150.
        - High automation risk (>0.60) usually pairs with simpler or routine tasks.
        - Very low automation risk (<0.20) often pairs with interpersonal or supervisory roles.
        - If current values violate these, adjust the smallest amount needed.

        ----------------------------------------------------------------------
        ### Batch variation (avoid clustering)
        ----------------------------------------------------------------------
        When multiple skills are processed in a single call:
        - Avoid identical scores for risk, demand, or hours unless skills are extremely similar.
        - Introduce small natural variations for realism (e.g. 0.34 vs 0.38).
        - The batch should resemble a plausible labour-market distribution while still matching each skill’s content.

        Single-row calls do not need batch-level distribution handling.

        ----------------------------------------------------------------------
        ### What to do for each skill row
        ----------------------------------------------------------------------
        1. Examine all text and current values.
        2. Treat existing values as priors; adjust only if implausible or inconsistent.
        3. Make domain-specific, non-generic decisions.
        4. Assign:
        - semantic_status = "ok", "adjusted", or "implausible"
        - semantic_reason = one-sentence explanation
        5. Always populate **all** *_suggested fields. No nulls.

        ----------------------------------------------------------------------
        ### Output format (STRICT JSON only)
        ----------------------------------------------------------------------
        Return **only**:

        {
        "results": [
            {
            "row_index": int,
            "semantic_status": "ok" | "adjusted" | "implausible",
            "semantic_reason": "short explanation",
            "aiAutomationRisk_suggested": float,
            "skillDemandScore_suggested": int,
            "skillTrend_suggested": "growing" | "stable" | "declining" | "mixed",
            "difficulty_level_suggested": "FOUNDATION" | "WORKING" | "ADVANCED" | "EXPERT",
            "career_level_suggested": "ENTRY" | "MID" | "SENIOR",
            "learning_time_estimate_hours_suggested": int,
            "learning_resources_suggested": "comma-separated string",
            "verification_method_suggested": "SELF_REPORTED" | "EVIDENCE_UPLOADED" | "SUPERVISOR_OR_INSTITUTION_REF" | "PLATFORM_VERIFIED",
            "assessment_type_suggested": "QUIZ_TEST" | "PRACTICAL_TASK" | "PROJECT_PORTFOLIO" | "WORK_SAMPLE" | "INTERVIEW_ASSESSMENT"
            }
        ]
        }
        """


    user_payload = {
        "instructions": (
            "Use the rows below. For each row, always output your final recommended values "
            "in the *_suggested fields. Do NOT use null; if you think a current value is fine, "
            "copy it into the suggested field. Follow the schema exactly."
        ),
        "rows": batch_rows,
    }

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": json.dumps(user_payload)},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("Model returned empty content for semantic batch")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse model JSON: {exc}") from exc

    results = parsed.get("results")
    if not isinstance(results, list):
        raise RuntimeError("Model response did not contain a 'results' list")

    # Lightweight debug of model behaviour for this batch
    sent_count = len(batch_rows)
    result_count = len(results)
    non_empty_status = sum(
        1 for r in results if str(r.get("semantic_status", "")).strip()
    )
    print(
        f"[semantic-debug] sent={sent_count}, results={result_count}, "
        f"non_empty_semantic_status={non_empty_status}"
    )
        # Show a small sample of the model response for inspection
    if results:
        sample = results[:1]
        print(f"[semantic-debug] sample results: {sample}")

    return results


def apply_semantic_evaluation(
    df: pd.DataFrame,
    client: OpenAI,
    checkpoint_fn: Optional[Callable[[pd.DataFrame], None]] = None,
) -> pd.DataFrame:
    """Run semantic evaluation over the DataFrame and add suggestion columns.

    This function is *resumable*: if a row already has a non-empty
    ``semantic_status``, it is skipped and preserved as-is. A checkpoint
    callback can be provided and will be invoked after each batch.
    """

    df = df.copy()

    # Initialise semantic columns only if missing, do not wipe existing work.
    semantic_cols = [
        "semantic_status",
        "semantic_reason",
        "aiAutomationRisk_suggested",
        "skillDemandScore_suggested",
        "skillTrend_suggested",
        "difficulty_level_suggested",
        "career_level_suggested",
        "learning_time_estimate_hours_suggested",
        "learning_resources_suggested",
        "verification_method_suggested",
        "assessment_type_suggested",
    ]
    for col in semantic_cols:
        if col not in df.columns:
            df[col] = ""

    all_rows = build_semantic_rows(df)
    # Debug: how many rows still need semantic processing before limiting
    print(f"[semantic] pending rows before MAX_SEMANTIC_ROWS limit: {len(all_rows)}")

    all_rows = maybe_limit_rows_for_semantic(all_rows)
    # Debug: which row indices are in scope after applying MAX_SEMANTIC_ROWS
    if all_rows:
        all_indices = [r["row_index"] for r in all_rows]
        print(
            "[semantic] rows after MAX_SEMANTIC_ROWS limit: "
            f"{len(all_rows)} indices={all_indices}"
        )
    if not all_rows:
        # Nothing left to process.
        return df

    batches = chunk_rows(all_rows, SEMANTIC_BATCH_SIZE)

    if not batches:
        return df

    concurrency = max(1, SEMANTIC_CONCURRENCY)
    print(
        f"[semantic] running {len(batches)} batches "
        f"with concurrency={concurrency} (batch_size={SEMANTIC_BATCH_SIZE})"
    )

    # Submit all batches to a thread pool. Each worker only calls the model
    # and returns its results; all DataFrame writes and checkpointing are done
    # in the main thread below.
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_batch_index = {}
        for batch_index, batch in enumerate(batches, start=1):
            batch_row_indices = [r["row_index"] for r in batch]
            print(
                f"[semantic] submit batch {batch_index}: size={len(batch)} "
                f"indices={batch_row_indices}"
            )
            future = executor.submit(semantic_review_batch, client, batch)
            future_to_batch_index[future] = batch_index

        completed = 0
        total = len(future_to_batch_index)
        for future in as_completed(future_to_batch_index):
            batch_index = future_to_batch_index[future]
            try:
                results = future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"[semantic] batch {batch_index} failed: {exc}")
                continue

            _apply_results_to_dataframe(df, results)

            if checkpoint_fn is not None:
                checkpoint_fn(df)

            completed += 1
            print(f"[semantic] completed {completed}/{total} batches")

    return df
