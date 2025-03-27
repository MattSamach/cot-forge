# Verification

Verification is a critical component of CoT Forge that ensures the reasoning paths lead to correct answers.

## Available Verifiers

### LLMJudgeVerifier

Uses an LLM to judge whether a reasoning path has arrived at the correct answer:

```python
from cot_forge.reasoning.verifiers import LLMJudgeVerifier
from cot_forge.llm import GeminiProvider

llm = GeminiProvider(api_key="your-api-key")
verifier = LLMJudgeVerifier(
    llm_provider=llm,
    strict_mode=True  # Enable strict verification
)
```

Parameters:
- `llm_provider`: LLM provider used for verification
- `strict_mode`: Whether to use stricter verification criteria (default: False)
- `system_prompt`: Custom system prompt for verification (optional)

### ExactMatchVerifier

Simple verifier that checks if the answer exactly matches the ground truth:

```python
from cot_forge.reasoning.verifiers import ExactMatchVerifier

verifier = ExactMatchVerifier()
```

## Verifier Interface

All verifiers implement the `Verifier` interface:

```python
class Verifier:
    def verify(
        self, 
        question: str, 
        answer: str, 
        ground_truth: str | None = None
    ) -> tuple[bool, float]:
        """
        Verify if the answer is correct.
        
        Returns:
            tuple[bool, float]: (is_correct, confidence_score)
        """
        pass
```

## Using with CoTBuilder

Verifiers are typically passed to the CoTBuilder to validate reasoning paths:

```python
from cot_forge.reasoning import CoTBuilder, NaiveLinearSearch
from cot_forge.reasoning.verifiers import LLMJudgeVerifier

builder = CoTBuilder(
    search_llm=llm,
    search=NaiveLinearSearch(),
    verifier=LLMJudgeVerifier(llm)
)
```

## Creating Custom Verifiers

You can create custom verifiers by implementing the `Verifier` interface:

```python
from cot_forge.reasoning.verifiers import Verifier

class MyCustomVerifier(Verifier):
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        
    def verify(
        self, 
        question: str, 
        answer: str, 
        ground_truth: str | None = None
    ) -> tuple[bool, float]:
        # Your verification logic
        # Return (is_correct, confidence_score)
        
        # Example:
        similarity = calculate_similarity(answer, ground_truth)
        is_correct = similarity > self.threshold
        return is_correct, similarity
```

## Verification without Ground Truth

Some verifiers can operate without ground truth by evaluating reasoning quality:

```python
# Use without ground truth
result = builder.build(question="What is the capital of France?")

# Use with ground truth for more reliable verification
result = builder.build(
    question="What is the capital of France?",
    ground_truth_answer="Paris"
)
```

## Handling Verification Errors

CoT Forge can handle verification failures gracefully:

```python
try:
    result = builder.build(question, ground_truth)
except VerificationError as e:
    print(f"Verification failed: {e}")
    # Handle the error
```