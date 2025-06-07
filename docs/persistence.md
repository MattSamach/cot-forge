# Persistence

The persistence module in CoT Forge allows you to save, resume, and manage reasoning generation for large datasets.

## PersistenceManager

The `PersistenceManager` class handles saving and loading of search results:

```python
from cot_forge.persistence import PersistenceManager

# Create a persistence manager
persistence = PersistenceManager(
    dataset_name="math_problems",
    search_name="beam_search",
    base_dir="data",
)
```

### Core Features

- Save results of CoT generation to disk
- Resume processing from where you left off
- Store configuration for reproducibility
- Support for multiple search algorithms per dataset

### Directory Structure

```
base_dir/
├── dataset_name/
│   ├── search_name/
│   │   ├── config.json       # Search configuration
│   │   ├── results.jsonl     # Search results
│   │   ├── processed_results.jsonl  # Post-processed results
│   │   └── metadata.json     # Tracking information
```

## Batch Processing with Persistence

```python

builder = CoTBuilder(
    search_llm=gpt_4o, # generates reasoning steps
    search=NaiveLinearSearch(), # Naive linear search chooses random reasoning steps in a chain
    verifier=LLMJudgeVerifier(llm_provider=sonnet_3_5, strict=False), # claude sonnet to verify answers
    post_processing_llm=sonnet_3_5, # converts reasoning into natural language
    dataset_name="medical-o1-verifiable-problem", # dataset name, used for folder structure
    base_dir= "./data", # base directory to save the results
)

# Define a batch of questions
questions = [
    "What is the capital of France?",
    "What is the largest mammal?",
    # ... many more questions
]

ground_truths = [
    "Paris",
    "Blue Whale",
    # ... corresponding answers
]

# Process with multi-threading for speed
results = builder.build_batch(
    questions=questions,
    ground_truth_answers=ground_truths,
    load_processed=True,
    multi_thread=True,
    max_workers=4  # Number of parallel workers
)
```

## Loading and Managing Results

```python
# Load a persistence manager
persistence = PersistenceManager(
    dataset_name="math_problems",
    search_name="naive_linear_search",
    base_dir="data"
)

# Load all previously processed results
all_results = persistence.load_results()

# Check processing status
processed_ids = persistence.processed_ids

# Get configuration
config = persistence.load_config()
```