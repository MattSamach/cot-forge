
import os

from cot_forge.llm import GeminiProvider
from cot_forge.reasoning import CoTBuilder, NaiveLinearSearch, SimpleBeamSearch
from cot_forge.reasoning.scorers import ProbabilityFinalAnswerScorer
from cot_forge.reasoning.verifiers import LLMJudgeVerifier

# Setup
api_key = os.getenv("GEMINI_API_KEY")
llm = GeminiProvider(api_key=api_key)
verifier = LLMJudgeVerifier(llm_provider=llm)
dataset_name = "dummy_dataset"
search = NaiveLinearSearch()


# Set up the CoTBuilder with persistence
builder = CoTBuilder.with_persistence(
    search_llm=llm,
    search=search,
    verifier=verifier,
    dataset_name=dataset_name,
)

# Data
batch_1 = {
    "questions": [
        "What is the capital of France?",
        "What is the largest mammal?",
        "How do you make a cake?",
    ],
    "ground_truth_answers": [
        "Paris",
        "Blue Whale",
        "Mix flour, sugar, and eggs."
    ],
}

batch_2 = {
    "questions": [
        "What is the capital of Germany?",
        "What is the fastest land animal?",
        "How do you make a pizza?"
    ],
    "ground_truth_answers": [
        "Berlin",
        "Cheetah",
        "Mix flour, water, and yeast."
    ],
}

results = builder.build_batch(
    questions=batch_1["questions"],
    ground_truth_answers=batch_1["ground_truth_answers"],
    load_processed=True,
)

results = builder.build_batch(
    questions=batch_2["questions"],
    ground_truth_answers=batch_2["ground_truth_answers"],
    load_processed=True, 
)

beam_search = SimpleBeamSearch(max_depth=3, branching_factor=2)

scorer = ProbabilityFinalAnswerScorer(
    llm_provider=llm,
)

beam_builder = CoTBuilder.with_persistence(
    search_llm=llm,
    search=beam_search,
    verifier=verifier,
    dataset_name=dataset_name,
    scorer=scorer
)

beam_results = beam_builder.build_batch(
    questions=batch_1["questions"],
    ground_truth_answers=batch_1["ground_truth_answers"],
    load_processed=True,
)

big_batch = {
    "questions": [
        "What is the capital of France?",
        "What is the largest mammal?",
        "How do you make a cake?",
        "What is the capital of Germany?",
        "What is the fastest land animal?",
        "How do you make a pizza?",
        "What is the capital of Italy?",
        "What is the largest reptile?",
        "How do you make a salad?",
        "What is the capital of Spain?",
        "What is the largest fish?",
        "How do you make a sandwich?",
        "What is the capital of Canada?",
        "What is the largest bird?",
    ],
    "ground_truth_answers": [
        "Paris",
        "Blue Whale",
        "Mix flour, sugar, and eggs.",
        "Berlin",
        "Cheetah",
        "Mix flour, water, and yeast.",
        "Rome",
        "Green Anaconda",
        "Mix lettuce, tomatoes, and dressing.",
        "Madrid",
        "Whale Shark",
        "Layer bread, meat, and cheese.",
        "Ottawa",
        "Ostrich",
    ],
}

beam_results = beam_builder.build_batch(
    questions=big_batch["questions"],
    ground_truth_answers=big_batch["ground_truth_answers"],
    load_processed=False,
    multi_thread=True,
)