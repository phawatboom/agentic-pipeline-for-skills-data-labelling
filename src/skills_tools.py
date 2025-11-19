from __future__ import annotations

from datetime import datetime
from typing import List

import pandas as pd


REQUIRED_FIELDS = [
    # "conceptUri",
    "skillType",
    "reuseLevel",
    "preferredLabel",
    # "description",   # main text field in your file
]

ALLOWED_SKILL_TRENDS = {"growing", "declining", "mixed", "stable"}


def load_skills_dataframe(csv_path) -> pd.DataFrame:
    """
    Load the skills CSV into a pandas DataFrame and normalise basic types.
    """
    df = pd.read_csv(csv_path, dtype=str).fillna("")

    # Strip whitespace from all string cells
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    optional_fields = [
        "difficulty_level",
        "career_level",
        "learning_time_estimate_hours",
        "learning_resources",
        "verification_method",
        "assessment_type",
    ]
    for field in optional_fields:
        if field not in df.columns:
            df[field] = ""

    # Ensure the key columns exist
    for field in REQUIRED_FIELDS + [
        "status",
        "modifiedDate",
        "aiAutomationRisk",
        "skillDemandScore",
        "skillTrend",
    ]:
        if field not in df.columns:
            raise ValueError(f"Expected column '{field}' not found in input CSV")

    return df


def _parse_float_safe(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_int_safe(value: str) -> int | None:
    if value == "":
        return None
    try:
        # Allow "55.0" etc
        return int(float(value))
    except ValueError:
        return None


def apply_structural_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply structural validation rules to each skill row.

    Adds:
      - structural_valid: bool
      - structural_issues: list of messages (as a JSON-like string)
    """
    structural_valid_list: List[bool] = []
    structural_issues_list: List[str] = []

    for _, row in df.iterrows():
        issues: List[str] = []

        # Required non empty fields
        for field in REQUIRED_FIELDS:
            if not str(row.get(field, "")).strip():
                issues.append(f"missing {field}")

        # conceptType
        concept_type = str(row.get("conceptType", "")).strip()
        if concept_type and concept_type != "KnowledgeSkillCompetence":
            issues.append(f"unexpected conceptType '{concept_type}'")

        # URIs
        # for uri_field in ["conceptUri", "skillUri"]:
        #     uri_val = str(row.get(uri_field, "")).strip()
        #     if uri_val and not uri_val.startswith("http://data.europa.eu/esco/skill/"):
        #         issues.append(f"{uri_field} has unexpected prefix")

        # status
        status_val = str(row.get("status", "")).strip()
        if status_val and status_val not in {"released"}:
            issues.append(f"unexpected status '{status_val}'")

        # modifiedDate format
        modified_val = str(row.get("modifiedDate", "")).strip()
        if modified_val:
            try:
                # ESCO uses ISO with Z
                datetime.fromisoformat(modified_val.replace("Z", "+00:00"))
            except ValueError:
                issues.append("modifiedDate invalid ISO format")

        # aiAutomationRisk
        ai_risk_raw = str(row.get("aiAutomationRisk", "")).strip()
        if ai_risk_raw:
            ai_risk = _parse_float_safe(ai_risk_raw)
            if ai_risk is None:
                issues.append("aiAutomationRisk not a float")
            else:
                if not (0.0 <= ai_risk <= 1.0):
                    issues.append("aiAutomationRisk out of [0,1]")

        # skillDemandScore
        demand_raw = str(row.get("skillDemandScore", "")).strip()
        if demand_raw:
            demand = _parse_int_safe(demand_raw)
            if demand is None:
                issues.append("skillDemandScore not an integer")
            else:
                if not (0 <= demand <= 100):
                    issues.append("skillDemandScore out of [0,100]")


        # skillTrend
        trend_raw = str(row.get("skillTrend", "")).strip()
        trend = trend_raw.lower()
        if trend and trend not in ALLOWED_SKILL_TRENDS:
            issues.append(f"invalid skillTrend '{trend_raw}'")

        # learning_time_estimate_hours
        hours_raw = str(row.get("learning_time_estimate_hours", "")).strip()
        if hours_raw:
            hours = _parse_float_safe(hours_raw)
            if hours is None or not (0 <= hours <= 5000):
                issues.append("learning_time_estimate_hours invalid or out of range [0,5000]")

        # verification_method
        verification_raw = str(row.get("verification_method", "")).strip().upper()
        if verification_raw and verification_raw not in {
            "SELF_REPORTED",
            "EVIDENCE_UPLOADED",
            "SUPERVISOR_OR_INSTITUTION_REF",
            "PLATFORM_VERIFIED",
        }:
            issues.append(f"invalid verification_method '{verification_raw}'")

        # assessment_type
        assessment_raw = str(row.get("assessment_type", "")).strip().upper()
        if assessment_raw and assessment_raw not in {
            "QUIZ_TEST",
            "PRACTICAL_TASK",
            "PROJECT_PORTFOLIO",
            "WORK_SAMPLE",
            "INTERVIEW_ASSESSMENT",
        }:
            issues.append(f"invalid assessment_type '{assessment_raw}'")

        structural_valid = len(issues) == 0
        structural_valid_list.append(structural_valid)

        # Store issues as a simple semicolon separated string for CSV
        structural_issues_list.append("; ".join(issues))

    df = df.copy()
    df["structural_valid"] = structural_valid_list
    df["structural_issues"] = structural_issues_list

    return df


def save_skills_dataframe(df: pd.DataFrame, output_path) -> None:
    """
    Save the DataFrame to CSV.
    """
    df_out = df.copy()

    # Treat empty strings as missing so "all empty" detection works
    df_out = df_out.replace("", pd.NA)

    # Drop any column where all values are NA / empty
    df_out = df_out.dropna(axis=1, how="all")

    df_out.to_csv(output_path, index=False, encoding="utf-8")
