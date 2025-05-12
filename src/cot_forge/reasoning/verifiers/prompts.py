DEFAULT_VERIFICATION_PROMPT = """You are an answer judge.
You are tasked with verifying the correctness of an answer to a question.
Verify if the provided answer successfully matches the ground truth answer.
They do not need to be identical, but they should convey the same meaning. (e.g., "the answer is leukemia" and "leukemia" are equivalent).
Answer with "yes" or "no" and provide a detailed (a few sentences) explanation.
Make sure to include the reasoning behind your decision.

Question that was asked: {question}
Provided answer: {final_answer}
Ground truth answer: {ground_truth_answer}
"""

STRICT_VERIFICATION_PROMPT = """You are a strict answer judge.
Verify if the provided answer below is equivalent to the ground truth answer.
While answers don't need to be word-for-word identical, they must:
1. Contain the same key facts and information
2. Maintain the same level of detail
3. Reach the same conclusions
4. Not include any contradicting statements

Be strict in your assessment. If there are any meaningful differences, mark it as not equivalent.

Answer with "yes" or "no" and provide a detailed (a few sentences) explanation.
Make sure to include the reasoning behind your decision.

Question that was asked: {question}
Provided answer: {final_answer}
Ground truth answer: {ground_truth_answer}
"""

VERIFICATION_FORMAT_PROMPT = """Strictly follow the JSON structure below.
Ensure a valid json response. Any newlines or special characters (like quotes) within this string must be properly escaped (e.g., use \\n for newlines, \\" for quotes).

{
"verification": {
    "explanation": <INSERT_DETAILED_EXPLANATION_HERE>,
    "result": <INSERT_YES_OR_NO_HERE>
}
}
```"""
