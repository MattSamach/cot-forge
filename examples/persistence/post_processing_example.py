import os

from cot_forge.llm import GeminiProvider
from cot_forge.post_processing import ReasoningProcessor

llm = GeminiProvider()

# Set current working directory
os.chdir(os.path.dirname(__file__))

processor = ReasoningProcessor(
    llm_provider=llm,
    dataset_name = "dummy_dataset",
    search_name="beam_search"
)

# Process results as batch
processor.process_batch(limit=20)


