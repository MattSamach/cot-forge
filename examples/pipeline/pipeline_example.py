from pathlib import Path

from datasets import load_dataset

from cot_forge.llm import AnthropicProvider, GeminiProvider, OpenAIProvider
from cot_forge.reasoning import CoTBuilder, NaiveLinearSearch
from cot_forge.reasoning.verifiers import LLMJudgeVerifier

# Dataset
ds = load_dataset("FreedomIntelligence/medical-o1-verifiable-problem", split="train[:200]")
problem = ds["Open-ended Verifiable Question"]
solution = ds["Ground-True Answer"]

# Determine the directory of the current script
SCRIPT_DIR = Path(__file__).resolve().parent

# Set up the CoTBuilder
builder = CoTBuilder.with_persistence(
    search_llm=GeminiProvider(),
    search=NaiveLinearSearch(),
    verifier=LLMJudgeVerifier(llm_provider=OpenAIProvider()),
    dataset_name="openthoughts_math",
    post_processing_llm=AnthropicProvider(),
    base_dir=SCRIPT_DIR / "data",
)

# process
builder.process_batch(
  questions=problem,
  ground_truth_answers=solution,
  max_workers=4,
  multi_thread=True,
  load_processed=True,
  only_successful=True,
  limit=20
)