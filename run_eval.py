import os
import sys
import json
import time
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.models import GeminiModel
from dotenv import load_dotenv
load_dotenv()


# ===== SETTINGS: change these =====

# Which metrics to run. Remove the ones you don't want.
# Options: "answer_relevancy", "faithfulness", "contextual_precision", "contextual_recall"
METRICS_TO_RUN = [
    "answer_relevancy",
    "faithfulness",
    "contextual_precision",
    "contextual_recall",
]

CASES_PER_METRIC = 2     # how many test cases to score per metric (None = all 10)
RETRIES = 3              # how many times to try again if the API fails (e.g. 503)
WAIT_SECONDS = 30        # seconds to wait between tries
RESULTS_FILE = "results.json"

# Optional command line overrides:
#   python run_eval.py                      uses the settings above
#   python run_eval.py faithfulness         runs just that one metric
#   python run_eval.py faithfulness 5       runs that metric on 5 cases
if len(sys.argv) > 1:
    METRICS_TO_RUN = [sys.argv[1]]
if len(sys.argv) > 2:
    CASES_PER_METRIC = int(sys.argv[2])


# The judge model (same as in class)

model = GeminiModel(
    model="gemini-3.5-flash-lite",
    temperature=0,
    api_key=os.getenv("GEMINI_API_KEY"),
)


# Load the test cases that build_cases.py saved

with open("cases.json", encoding="utf-8") as f:
    all_cases = json.load(f)


# Every metric: name -> (metric, which set of questions to use)
ALL_METRICS = {
    "answer_relevancy": AnswerRelevancyMetric(threshold=0.7, model=model, include_reason=True, async_mode=False),
    "faithfulness": FaithfulnessMetric(threshold=0.7, model=model, include_reason=True, async_mode=False),
    "contextual_precision": ContextualPrecisionMetric(threshold=0.7, model=model, include_reason=True, async_mode=False),
    "contextual_recall": ContextualRecallMetric(threshold=0.7, model=model, include_reason=True, async_mode=False),
}


def score_with_retry(metric, test_case):
    """Score one test case. If the API fails, wait and try again."""
    for attempt in range(1, RETRIES + 1):
        try:
            metric.measure(test_case)
            return
        except Exception as e:
            print(f"   attempt {attempt}/{RETRIES} failed: {str(e)[:100]}")
            if attempt == RETRIES:
                raise
            time.sleep(WAIT_SECONDS)


# ===== RUN: keep this part at the very bottom =====

results = []

for name in METRICS_TO_RUN:
    metric = ALL_METRICS[name]
    cases = all_cases[name][:CASES_PER_METRIC]
    print(f"\n===== {name} ({len(cases)} cases) =====")

    for number, c in enumerate(cases, start=1):
        print(f"[{name} {number}/{len(cases)}] {c['input']}")

        test_case = LLMTestCase(
            input=c["input"],
            actual_output=c["actual_output"],
            expected_output=c["expected_output"] or None,
            retrieval_context=c["retrieval_context"],
        )

        row = {"metric": name, "input": c["input"], "actual_output": c["actual_output"]}
        try:
            score_with_retry(metric, test_case)
            row.update(score=metric.score, passed=metric.is_successful(), reason=metric.reason)
            print(f"   score: {metric.score}  passed: {metric.is_successful()}")
        except Exception as e:
            row.update(score=None, passed=False, reason=f"ERROR: {e}")
            print("   gave up on this case")
        results.append(row)

# Save everything at the end
with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(results)} results to {RESULTS_FILE}")
