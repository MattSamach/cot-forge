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
    auto_resume=True
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

## Using with CoTBuilder

The `CoTBuilder.with_persistence` factory method creates a builder with persistence:

```python
from cot_forge.reasoning import CoTBuilder, SimpleBeamSearch
from cot_forge.llm import GeminiProvider
from cot_forge.reasoning.verifiers import LLMJudgeVerifier

llm = GeminiProvider(api_key="your-api-key")
verifier = LLMJudgeVerifier(llm_provider=llm)
search = SimpleBeamSearch(max_depth=3, branching_factor=2)

# Create builder with persistence
builder = CoTBuilder.with_persistence(
    search_llm=llm,
    search=search,
    verifier=verifier,
    dataset_name="my_dataset",
    search_name="beam_search",
    base_dir="data"
)

# Process batches with automatic persistence
results = builder.build_batch(
    questions=["What is 2+2?", "What is the capital of France?"],
    ground_truth_answers=["4", "Paris"],
    load_processed=True  # Skip already processed questions
)
```

## Batch Processing with Persistence

```python
# Define a large batch of questions
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
# Load all previously processed results
all_results = persistence.load_results()

# Check processing status
processed_ids = persistence.processed_ids

# Get configuration
config = persistence.load_config()
```

## Integrating with ReasoningProcessor

The persistence system integrates seamlessly with the post-processing module:

```python
from cot_forge.post_processing.reasoning_processor import ReasoningProcessor

# Create processor that uses the same dataset and search name
processor = ReasoningProcessor(
    llm_provider=llm,
    dataset_name="my_dataset",
    search_name="beam_search",
    base_dir="data"  # Same base_dir as the persistence manager
)

# Process all successful results
processed = processor.process_batch(only_successful=True)
```

## Customizing Persistence

You can customize the persistence behavior:

```python
# Create with custom paths
persistence = PersistenceManager(
    dataset_name="custom_dataset",
    search_name="custom_search",
    base_dir="/path/to/custom/directory",
    auto_resume=False  # Disable auto-resuming, will overwrite existing data
)

# Save custom configuration
persistence.save_config({
    "model": "gemini-pro",
    "temperature": 0.7,
    "strategy_registry": strategy_registry.serialize(),
    "custom_parameter": "value"
})
```

## Managing Multiple Search Strategies

You can run different search strategies on the same dataset:

```python
# Linear search
linear_builder = CoTBuilder.with_persistence(
    search_llm=llm,
    search=NaiveLinearSearch(),
    verifier=verifier,
    dataset_name="math_problems",
    search_name="linear_search"
)

# Beam search on same dataset
beam_builder = CoTBuilder.with_persistence(
    search_llm=llm,
    search=SimpleBeamSearch(),
    verifier=verifier,
    dataset_name="math_problems",
    search_name="beam_search"
)
```