# Evaluation and Reliability Plan

This repository enriches ESCO skills data with LLM-produced suggestions. Suggestions are **not ground truth**. An effective portfolio demonstration should measure quality, reproducibility, resilience and cost.

## Evaluation protocol

1. Sample rows from multiple skill families and ambiguous/rare categories.
2. Define human annotation guidelines and a fixed validation set.
3. Evaluate each output field with an appropriate metric. Classification fields can use accuracy and macro F1, ordinal ratings can use agreement/error metrics, and numeric estimates should report absolute error with clear units.
4. Report disagreement between annotators and examine failure patterns, rather than reporting one accuracy number without its definition.
5. Track cost per enriched row, error rate, throughput and the model/prompt version used.

## Resilience checks

| Case | Required behavior |
| --- | --- |
| Interruption after one completed batch | Restart should retain completed records |
| Malformed model JSON or invalid values | Mark errors and preserve raw context without silently treating output as valid |
| Rate limit or timeout | Retry with bounded backoff and an attempt limit |
| Partial checkpoint write | Preserve a recoverable prior checkpoint, ideally via temporary-file plus atomic replacement |
| Duplicate input identifiers | Define stable row identity so retries cannot corrupt or skip records |
| Budget exhaustion | Stop safely rather than continuing unbounded API calls |

These behaviors need implementation and tests before they can be claimed as guarantees. The current pipeline uses checkpoint CSV files; this document does not assert that writes are atomic.

## Suggested test cases

Use a mocked LLM response so tests never require paid API calls. Cover valid output, malformed output, missing values, retryable errors, already-completed rows and restart behavior. Include a small, non-sensitive fixture dataset.

## Reporting template

| Measure | Value |
| --- | --- |
| Rows evaluated | Not measured |
| Annotation agreement | Not measured |
| Classification accuracy/F1 | Not measured |
| Cost per enriched record | Not measured |
| Failed/retried batches | Not measured |

Replace placeholders only with reproducible results. Never infer them from anecdotal examples.
