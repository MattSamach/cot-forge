DEFAULT_VERIFICATION_PROMPT = """You are an answer judge.
Verify if the provided answer below is equivalent to the ground truth answer.
They do not need to be identical, but they should convey the same meaning.
Answer with "yes" or "no" and provide a detailed (a few sentences) explanation.
Make sure to include the reasoning behind your decision.

Provided answer: {final_answer}
Ground truth answer: {ground_truth_answer}
"""

VERIFICATION_FORMAT_PROMPT = """Strictly follow the JSON structure below.

```json
{
"verification": {
    "explanation": <INSERT_DETAILED_EXPLANATION_HERE>,
    "result": <INSERT_YES_OR_NO_HERE>
}
}
```"""