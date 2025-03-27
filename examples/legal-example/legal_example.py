import os

from cot_forge.llm import GeminiProvider
from cot_forge.reasoning import CoTBuilder, SimpleBeamSearch, default_strategy_registry
from cot_forge.reasoning.scorers import ProbabilityFinalAnswerScorer
from cot_forge.reasoning.verifiers import LLMJudgeVerifier

legal_facts_question = """
To help alleviate the burdens of poverty and perhaps to help cut the welfare rolls, the state legislature of Margate passed legislation establishing State Family Counseling Centers throughout the state. 
The legislature recognized that much of the \"welfare and poverty problem\" was centered on poor single-parent households headed by a woman.
Therefore, it decreed in the legislation that all counseling would be free for single mothers with an income of less than $ 20,000$ per year.
Others would have to pay fees for the counseling on a sliding scale depending upon income.
The counseling services available at the State Family Counseling Centers included classes on parenting, anti-substance abuse programs, and financial counseling.
The counseling centers were popular and other states considered copying the Margate model.
Peter's wife had died recently of a drug overdose, and he was left to care for their two children (ages two and four) on an income of approximately $ 7,000$ per year.
Peter had no idea how he could manage to care for the two children and himself on his small paychecks.
He heard about the State Family Counseling Centers and went to the closest one for financial counseling.
Peter was told that he would have to pay a $ 50$ fee for the counseling.
Peter had $ 10$ in his pocket, which he needed for bus fare home and to feed his children until his check, due in five days, arrived.
Peter became very angry when he learned that single mothers in his situation were entitled to free counseling while single fathers were not.
A public-interest law firm agreed to take Peter's case and filed suit in federal court, asking that Peter be allowed to take advantage of the free counseling services because the law establishing them discriminated against males. 

To win the case, what must be shown by whom?
"""

ground_truth_answer = """
Margate must show that favoring mothers over fathers is substantially related to an important governmental interest.
"""

api_key = os.getenv("GEMINI_API_KEY")
llm = GeminiProvider(api_key=api_key)

builder = CoTBuilder(
    search_llm=llm,
    search=SimpleBeamSearch(max_depth=3, beam_width=3, branching_factor=2),
    # search=NaiveLinearSearch(max_depth=3),
    verifier=LLMJudgeVerifier(llm),
    strategy_reg=default_strategy_registry,
    scorer=ProbabilityFinalAnswerScorer(llm)
)

def main():
    search_result = builder.build(
        question=legal_facts_question,
        ground_truth_answer=ground_truth_answer,
    )
    return search_result

if __name__ == "__main__":
    search_result = main()
    print(search_result)
