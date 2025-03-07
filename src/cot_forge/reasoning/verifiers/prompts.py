DEFAULT_VERIFICATION_PROMPT = """You are an answer judge.
Verify if the provided answer below is equivalent to the ground truth answer.
Answer simply with "yes" or "no".

Provided answer: {final_answer}
Ground truth answer: {ground_truth_answer}
"""