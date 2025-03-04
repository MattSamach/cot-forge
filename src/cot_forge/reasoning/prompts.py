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
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Conclusion"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Conclusion**: Summarize the correct reasoning from all previous 'Inner Thinking' steps and provide the final answer. No title is needed for this section.
3. **Verification**: Verify the accuracy of the "Final Conclusion". If it holds, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.
</response requirements>\n\n"""

    @staticmethod
    def create_initial_instruction() -> str:
        return "Please respond to the above question <question> using the Chain of Thought (CoT) reasoning method.\n"

    @staticmethod
    def create_new_instruction(strategy_description: str) -> str:
        return f"""<question> represents the question to be answered, and <previous reasoning> contains your prior reasoning.
<new_instruction>Your task is to continue from the current ’Verification’ step. I have manually reviewed the reasoning and
determined that the **Final Conclusion** is false. Your ’Verification’ results must align with mine. {strategy_description}. Then construct a new Final Conclusion.</new_instruction>\n\n"""
    
    @staticmethod
    def create_json_format() -> str:
        return """### Output Format
Strictly follow the JSON structure below. You do not need to repeat your previous reasoning. Begin directly from the next 'Verification' stage.

```json
{
"CoT": [
    {"action": "Verification", "content": "..."},
    {"action": "Inner Thinking", "title": "...", "content": "..."},
    ...,
    {"action": "Final Conclusion", "content": "..."},
    {"action": "Verification", "content": "..."}
]
}
```"""

initialize_cot_prompt = "Please respond to the above question <question> using the Chain of Thought (CoT) reasoning method."
backtrack_strategy_prompt = "Refine the reasoning using **backtracking** to revisit earlier points of reasoning.",
explore_new_paths_strategy_prompt = "Refine the reasoning by exploring new approaches to solving this problem.",
correction_strategy_prompt = "Refine the reasoning by making precise **corrections** to address prior flaws.",
validation_strategy_prompt = "Refine the reasoning by conducting a thorough **validation** process to ensure validity."