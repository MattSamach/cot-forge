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
determined that the **Final Conclusion** is false. Create a new chain of thought by: {strategy_description}. Then construct a new Final Conclusion.</new_instruction>\n\n"""
    
    @staticmethod
    def create_json_format() -> str:
        return """### Output Format
Strictly follow the JSON structure below. You do not need to repeat your previous reasoning. Begin directly from the last 'Final Conclusion' stage.

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
initialize_cot_prompt = "responding to the above question <question> using the Chain of Thought (CoT) reasoning method. Because this is the initial reasoning, do not start with the `Review` step. Instead, begin with the `Inner Thinking` step and then conclude with the `Final Conclusion` step."
backtrack_strategy_prompt = "refining the reasoning using **backtracking** to revisit earlier points of reasoning."
explore_new_paths_strategy_prompt = "refining the reasoning by **exploring new approaches** to solving this problem."
correction_strategy_prompt = "refining the reasoning by making precise **corrections** to address prior flaws."
validation_strategy_prompt = "refining the reasoning by conducting a thorough **validation** process to ensure validity."

# Additional strategy prompts
perspective_shift_strategy_prompt = "refining the existing reasoning by **re-examining your prior conclusions** from multiple perspectives (expert, novice, critic)."
analogical_reasoning_prompt = "continuing from your previous chain of thought, use **analogies or metaphors** to reinterpret challenging aspects of your reasoning."
decomposition_strategy_prompt = "building on your work so far, identify any complex remaining issues and **break them down into smaller sub-problems**. Address these components while preserving the valid portions of your previous reasoning."
counterfactual_strategy_prompt = "using counterfactuals to test the robustness of your reasoning by exploring **what-if scenarios** for any questionable assumptions or conclusions. Use these counterfactuals to refine your analysis."
first_principles_strategy_prompt = "examining your existing reasoning chain to identify which underlying **foundational principles** need reconsideration. Revise those specific elements while building upon the valid aspects of your prior work."
progressive_refinement_prompt = "refining your reasoning through **progressive approximation**, keeping what works while making increasingly subtle adjustments to the problematic elements."
constraint_analysis_prompt = "building upon your existing reasoning by identifying and **temporarily relaxing key constraints** that may be limiting your analysis, then reincorporating them with newfound insights."
evidence_reweighting_prompt = "reviewing your reasoning chain and **reassess the relative importance** of different pieces of evidence, adjusting your conclusions based on this reweighted analysis."