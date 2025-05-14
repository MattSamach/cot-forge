class StrategyPromptTemplate:
  """Base class for modular prompt construction"""

  @staticmethod
  def create_header(question: str) -> str:
    return f"<question>\n{question}\n</question>\n\n"

  @staticmethod
  def create_previous_reasoning(previous_cot: str) -> str:
    return f"<previous_reasoning>\n{previous_cot}\n</previous_reasoning>\n\n"

  @staticmethod
  def create_response_requirements() -> str:
    return """<response_requirements>
Your response must include the following steps, each composed of three types of actions: **"Review"**, **"Inner Thinking"** and **"Final Conclusion"**:

1. **Review**: Review the previous reasoning and identify any flaws or errors. This should be a brief summary of the previous reasoning.
2. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
3. **Final Conclusion**: Summarize the correct reasoning from all previous 'Inner Thinking' steps and provide the final answer. No title is needed for this section.

You may skip the **Review** step if you are starting a new reasoning chain.
</response_requirements>
\n\n"""

  @staticmethod
  def create_initial_instruction() -> str:
    return "Please respond to the above question <question> using the Chain of Thought (CoT) reasoning method.\n"

  @staticmethod
  def create_new_instruction(strategy_description: str) -> str:
    return f"""<question> represents the question to be answered, and <previous_reasoning> contains your prior reasoning.
<new_instruction>Your task is to continue from the last ’Final Conclusion’ step. We have assessed the reasoning and
determined that the current **Final Conclusion** is false. 

## Create a new chain of thought by implementing this strategy: {strategy_description}. It is VITAL that you employ THIS strategy in your reasoning!!!

Then construct a new Final Conclusion.</new_instruction>\n\n"""

  @staticmethod
  def create_json_format() -> str:
    return """### Output Format
Strictly follow the JSON structure below. You do not need to repeat your previous reasoning. Begin directly from the last 'Final Conclusion' stage.
Ensure a valid json response. Any newlines or special characters (like quotes) within this string must be properly escaped (e.g., use \\n for newlines, \\" for quotes).

```json
{
"CoT": [
    {"action": "Review", "content": "<INSERT_CONTENT_HERE>"},
    {"action": "Inner Thinking", "title": "<INSERT_TITLE_HERE>", "content": "<INSERT_CONTENT_HERE>"},
    {"action": "Inner Thinking", "title": "<INSERT_TITLE_HERE>", "content": "<INSERT_CONTENT_HERE>"},
    ...,
    {"action": "Final Conclusion", "content": "<INSERT_CONTENT_HERE>"}
]
}
```"""


# Default strategy prompts
initialize_cot_prompt = "Initilization: Respond to the above question <question> using the Chain of Thought (CoT) reasoning method. Because this is the initial reasoning, do not start with the `Review` step. Instead, begin with the `Inner Thinking` step and then conclude with the `Final Conclusion` step."
backtrack_strategy_prompt = "Backtrack: Revise the reasoning by **backtracking to an earlier point** in your analysis. Identify a specific earlier step where the reasoning began to go off track, return to that point, and develop a new chain of thought from there rather than continuing from your latest conclusion."
explore_new_paths_strategy_prompt = "Explore New Path: Expand the reasoning by **exploring new approaches** to solving this problem, proposing new ideas and paths that haven't yet been covered."
correction_strategy_prompt = "Correction: Refine the reasoning by examining making precise **corrections** to address prior flaws."
validation_strategy_prompt = "Validation: Improve the reasoning by conducting a thorough **validation** process that explicitly checks each assumption, verifies logical connections between statements, and ensures that conclusions directly follow from the evidence presented."

# Additional strategy prompts
perspective_shift_strategy_prompt = "Perspective Shift: Refine the reasoning by **re-examining your prior conclusions** from multiple perspectives (expert, novice, critic). Incorporate these diverse viewpoints to identify blind spots in your original analysis and develop a more robust solution."
analogical_reasoning_prompt = "Analogical Reasoning: Enhance your reasoning by using **analogies or metaphors** to reinterpret challenging aspects of your analysis. Draw parallels to more familiar concepts that can illuminate the current problem from a different angle."
decomposition_strategy_prompt = "Decomposition: Improve your reasoning by identifying any complex remaining issues and **breaking them down into smaller sub-problems**. Address these components systematically while preserving the valid portions of your previous reasoning."
counterfactual_strategy_prompt = "Counterfactual Analysis: Strengthen your reasoning by using counterfactuals to test the robustness of your analysis, exploring **what-if scenarios** for any questionable assumptions or conclusions. Use these counterfactual insights to refine your approach."
first_principles_strategy_prompt = "First Principles: Enhance your reasoning by examining which underlying **foundational principles** need reconsideration. Identify the basic truths or axioms relevant to this problem and rebuild your analysis from these fundamentals."
constraint_analysis_prompt = "Constraint Analysis: Develop better reasoning by identifying and **temporarily relaxing key constraints** that may be limiting your analysis, then reincorporating them with newfound insights to arrive at a more comprehensive solution."
evidence_reweighting_prompt = "Evidence Reweighting: Revise your reasoning by **reassessing the relative importance** of different pieces of evidence, adjusting your conclusions based on this reweighted analysis to better align with the true significance of each factor."
