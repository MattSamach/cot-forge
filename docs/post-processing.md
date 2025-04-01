# Post-Processing

The Post-Processing module in CoT Forge transforms rigid, structured reasoning into natural language suitable for model training.

## ReasoningProcessor

The `ReasoningProcessor` class is the main tool for post-processing:

```python
from cot_forge.post_processing.reasoning_processor import ReasoningProcessor
from cot_forge.llm import GeminiProvider

processor = ReasoningProcessor(
    llm_provider=llm,
    dataset_name="math_problems",
    search_name="beam_search",
    base_dir="data"
)
```

### Key Features

- Converts structured CoT into natural language reasoning
- Generates formal responses based on the reasoning
- Processes batches of results from previous CoT generation
- Saves processed results for model training

### Core Methods

#### Process Individual CoT

```python
# Process a single CoT
result, error = processor.process(
    question="What is the area of a circle with radius 5?",
    cot = [
        {"action": "Inner Thinking", "title": "Identifying formula", "content": "I need to use the formula for the area of a circle"},
        {"action": "Inner Thinking", "title": "Retrieving formula", "content": "The formula is A = π * r^2"},
        {"action": "Inner Thinking", "title": "Calculating area", "content": "Substituting r=5 into the formula gives A = π * 5^2"},
        {"action": "Inner Thinking", "title": "Final Calculation", "content": "A = π * 25 = 78.54"},
        {"action": "Final Conclusion", "content": "The area is 78.54 square units"}
    ]
)

# Access results
natural_reasoning = result["natural_reasoning"]
formal_response = result["formal_response"]
```

#### Process Batches

```python
# Process a batch of previously generated results
processed_results = processor.process_batch(
    only_successful=True,  # Only process successful CoTs
    limit=100,             # Limit number of results
    progress_bar=True      # Show progress bar
)
```

### Loading and Saving

```python
# Load previously processed results
results = processor.load_processed_results()

# Results are automatically saved during processing,
# but you can manually save a result:
processor._save_processed_result({
    "id": "problem-123",
    "question": "What is 2+2?",
    "ground_truth": "4",
    "natural_reasoning": "To find the sum...",
    "formal_response": "The answer is 4."
})
```

## Natural Language Generation

The processor uses LLMs to transform structured reasoning into natural language:

```python
# Generate natural-sounding reasoning
natural_reasoning = processor.generate_natural_reasoning(
    question = "What is the capital of France?",
    cot = [
        {"action": "Inner Thinking", "title": "Location", "content": "France is a country in Western Europe"},
        {"action": "Inner Thinking", "title": "Capital", "content": "The capital of France is Paris"},
        {"action": "Final Conclusion", "content": "Paris"}
    ]
)

# Generate a formal response based on natural reasoning
formal_response = processor.generate_formal_response(
    question="What is the capital of France?",
    natural_reasoning=natural_reasoning
)
```

## Integration with CoTBuilder

ReasoningProcessor works seamlessly with CoTBuilder results:

```python
from cot_forge.reasoning import CoTBuilder, NaiveLinearSearch
from cot_forge.post_processing.reasoning_processor import ReasoningProcessor

# Generate CoTs
builder = CoTBuilder(search_llm=llm, search=NaiveLinearSearch(), verifier=verifier)
result = builder.build(
    question="What is the sum of angles in a triangle?",
    ground_truth_answer="180 degrees"
)

# Process the best reasoning path
processor = ReasoningProcessor(
    llm_provider=llm,
    dataset_name="geometry",
    search_name="naive_linear_search"
)

successful_node = result.get_successful_terminal_nodes()[0]
processed, _ = processor.process(
    question="What is the sum of angles in a triangle?",
    cot=successful_node.get_full_cot()
)

print(processed["natural_reasoning"])
```

## Processing Large Datasets

For large datasets, use the batch processing functionality with persistence:

```python
# Initialize processor with persistence
processor = ReasoningProcessor(
    llm_provider=llm,
    dataset_name="large_dataset",
    search_name="beam_search",
    base_dir="data/large_processing"
)

# Process all successful results
all_processed = processor.process_batch(only_successful=True)
```