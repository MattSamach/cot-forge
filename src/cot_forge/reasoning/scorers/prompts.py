class ScorerPromptTemplate:
    """Base class for modular prompt construction"""
    
    @staticmethod
    def base_prompt(question: str, answer: str) -> str:
        return f"""You are a helpful assistant that scores chains of thought (CoT) based on their quality and correctness.
You are given a question, a ground truth answer, and a list responses to that question.
Your task is to score each response based on how well it answers the question and how closely it matches the ground truth answer.

<question>
{question}
</question>

<ground_truth_answer>
{answer}
</ground_truth_answer>
"""

    @staticmethod
    def format_prompt() -> str:
        return """Strictly follow the JSON structure below.
Ensure a valid json response. Any newlines or special characters (like quotes) within this string must be properly escaped (e.g., use \\n for newlines, \\" for quotes).
The output should be a JSON object with a "scoring" key, which contains another JSON object.
The inner JSON object should have keys corresponding to the names of the options used to generate the responses.
The values should be the scores for each response.
```json
{
"scoring": {
    "option_name_1": <INSERT_NUMERIC_SCORE_HERE>,
    "option_name_2": <INSERT_NUMERIC_SCORE_HERE>
}
}
```"""

    @staticmethod
    def scorer_instruction_prompt(template: str, **kwargs) -> str:
        """Format the prompt with the given template and additional arguments."""
        return template.format(**kwargs)
    
    @staticmethod
    def build_prompt(question: str, 
                     answer: str,
                     instruction_prompt: str,
                     **kwargs) -> str:
        """Build the complete prompt by combining the base prompt and format prompt."""
        base = ScorerPromptTemplate.base_prompt(question, answer)
        format_prompt = ScorerPromptTemplate.format_prompt()
        instruction_prompt = ScorerPromptTemplate.scorer_instruction_prompt(instruction_prompt, **kwargs)
        # Combine all parts of the prompt
        return f"{base}\n\n{instruction_prompt}\n\n{format_prompt}"

PROBABILITY_FINAL_ANSWER_PROMPT = """Below is a list of final answers that different chains of thought (CoT) have produced.
Your task is to score each final answer based on how likely it is to be correct. Score each final answer on a scale from 0 to 1, 
where 0 means "very unlikely to be correct" and 1 means "very likely to be correct". Use up to two decimals (e.g. 0.75).

<FINAL_ANSWERS>
{final_answers}
</FINAL_ANSWERS>
"""