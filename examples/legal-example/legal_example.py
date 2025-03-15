import os

from cot_forge.llm import GeminiLLMProvider
from cot_forge.reasoning import (CoTBuilder, default_strategy_registry,
                                 naive_linear_search, simple_beam_search)
from cot_forge.reasoning.scorers import ProbabilityFinalAnswerScorer

legal_facts_question = """
Below are the facts of a legal case. Based on these facts, please answer the following question:

<question>
What are the key legal issues presented in the case?
</question>

<facts>
"A convict named Kehoe, who was unable to pay a fine, applied for discharge under Section 1042 of the Revised Statutes after being confined in prison for thirty days solely for non-payment of the fine. The United States District Court for the District of Alaska considered whether it qualified as a court of the United States under Section 1042, given the existing laws of Alaska regarding imprisonment for non-payment of fines."
</facts>
"""

ground_truth_answer = """
The key legal issues presented in the case are:
1. "Is the District Court of Alaska a court of the United States within the meaning of section 1042 of the Revised Statutes?",
2. "Do the provisions of section 1042 of the Revised Statutes have any force in the territory of Alaska, in view of the sections of the Compiled Laws of Alaska regarding imprisonment for non-payment of fines?" 
"""

api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
llm = GeminiLLMProvider(api_key=api_key)

builder = CoTBuilder(
    llm=llm,
    search=simple_beam_search,
    strategy_reg=default_strategy_registry,
)

def main():
    search_result = builder.build(question=legal_facts_question,
                        ground_truth_answer=ground_truth_answer,
                        max_depth=3,
                        beam_width=3,
                        branching_factor=2,
                        scorer=ProbabilityFinalAnswerScorer()
                    )
    return search_result

if __name__ == "__main__":
    search_result = main()
    print(search_result)
