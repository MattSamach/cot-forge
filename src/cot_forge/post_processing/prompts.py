

NATURAL_LANGUAGE_COT_PROMPT = """<Thought Process>
{Thought_Process}
</Thought Process>

<Question>
{Question}
</Question>

The <Thought Process> above reflects a large language model's reasoning based on the <Question>. Your task is to
rewrite the <Thought Process> to resemble a more human-like, intuitive natural thinking process. The new
version should:
1. Be presented as step-by-step reasoning, with each thought on a new line separated by a line break.
2. Avoid structured titles or formatting, focusing on natural transitions. Use casual and natural language for
transitions or validations, such as "hmm," "oh," "also," "actually," or "wait."
3. Expand the content, making the reasoning richer, more detailed, and logically clear while still being
conversational and intuitive.
Return directly the revised natural thinking in JSON format as follows:

```json
{
"NaturalReasoning": "<INSERT_CONTENT_HERE>"
}
```
"""

def build_natural_language_cot_prompt(
    question: str,
    cot: dict,
) -> str:
    """
    Build the natural language chain of thought prompt.

    Args:
        question (str): The question to be answered.
        cot (dict): The chain of thought to be reformatted.

    Returns:
        str: The formatted prompt.
    """
    return NATURAL_LANGUAGE_COT_PROMPT.format(
        Question=question,
        Thought_Process=str(cot),
    )
    
FORMAL_RESPONSE_PROMPT = """<Thinking>
{natural_reasoning}
</Thinking>

<Question>
{question}
</Question>

The <Thinking> tags represents your internal thoughts about the <Question>. Based on this, generate
a rich and high-quality final response to the user. If there is a clear answer, provide it first. Ensure your
final response closely follows the <Question>. Output only your final response, without any additional content.
"""

def build_formal_response_prompt(
    question: str,
    natural_reasoning: str,
) -> str:
    """
    Build the formal response prompt.

    Args:
        question (str): The question to be answered.
        natural_reasoning (str): The natural reasoning process.

    Returns:
        str: The formatted prompt.
    """
    return FORMAL_RESPONSE_PROMPT.format(
        question=question,
        natural_reasoning=natural_reasoning,
    )