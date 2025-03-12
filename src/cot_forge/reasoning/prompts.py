class StrategyPromptTemplate:
    """Base class for modular prompt construction"""
    
    @staticmethod
    def create_header(question: str) -> str:
        return f"<question>\n{question}\n</question>\n\n"
    
    @staticmethod
    def create_previous_reasoning(previous_cot: str) -> str:
        return f"<previous reasoning>\n{previous_cot}\n</previous reasoning>\n\n"
    
    @staticmethod
    def create_response_requirements() -> str:
        return """<response requirements>
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"** and **"Final Conclusion"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Conclusion**: Summarize the correct reasoning from all previous 'Inner Thinking' steps and provide the final answer. No title is needed for this section.
</response requirements>\n\n"""

    @staticmethod
    def create_initial_instruction() -> str:
        return "Please respond to the above question <question> using the Chain of Thought (CoT) reasoning method.\n"

    @staticmethod
    def create_new_instruction(strategy_description: str) -> str:
        return f"""<question> represents the question to be answered, and <previous reasoning> contains your prior reasoning.
<new_instruction>Your task is to continue from the last ’Verification’ step. We have manually reviewed the reasoning and
determined that the **Final Conclusion** is false. {strategy_description}. Then construct a new Final Conclusion.</new_instruction>\n\n"""
    
    @staticmethod
    def create_json_format() -> str:
        return """### Output Format
Strictly follow the JSON structure below. You do not need to repeat your previous reasoning. Begin directly from the last 'Verification' stage.

```json
{
"CoT": [
    {"action": "Inner Thinking", "title": "...", "content": "..."},
    {"action": "Inner Thinking", "title": "...", "content": "..."},
    ...,
    {"action": "Final Conclusion", "content": "..."},
]
}
```"""

# Default strategy prompts
initialize_cot_prompt = "Please respond to the above question <question> using the Chain of Thought (CoT) reasoning method."
backtrack_strategy_prompt = "Refine the reasoning using **backtracking** to revisit earlier points of reasoning."
explore_new_paths_strategy_prompt = "Refine the reasoning by **exploring new approaches** to solving this problem."
correction_strategy_prompt = "Refine the reasoning by making precise **corrections** to address prior flaws."
validation_strategy_prompt = "Refine the reasoning by conducting a thorough **validation** process to ensure validity."

# Additional strategy prompts
# Additional strategy prompts - revised for iteration emphasis
perspective_shift_strategy_prompt = "Refine the existing reasoning by **re-examining your prior conclusions** from multiple perspectives (expert, novice, critic)."
analogical_reasoning_prompt = "Continuing from your previous chain of thought, use **analogies or metaphors** to reinterpret challenging aspects of your reasoning."
decomposition_strategy_prompt = "Building on your work so far, identify any complex remaining issues and **break them down into smaller sub-problems**. Address these components while preserving the valid portions of your previous reasoning."
counterfactual_strategy_prompt = "Use counterfactuals to test the robustness of your reasoning by exploring **what-if scenarios** for any questionable assumptions or conclusions. Use these counterfactuals to refine your analysis."
first_principles_strategy_prompt = "Examine your existing reasoning chain to identify which underlying **foundational principles** need reconsideration. Revise those specific elements while building upon the valid aspects of your prior work."
progressive_refinement_prompt = "Refine your reasoning through **progressive approximation**, keeping what works while making increasingly subtle adjustments to the problematic elements."
constraint_analysis_prompt = "Build upon your existing reasoning by identifying and **temporarily relaxing key constraints** that may be limiting your analysis, then reincorporating them with newfound insights."
evidence_reweighting_prompt = "Review your reasoning chain and **reassess the relative importance** of different pieces of evidence, adjusting your conclusions based on this reweighted analysis."