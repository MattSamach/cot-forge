# Scorers

Scorers are used to rank different reasoning paths by quality, helping CoT Forge select the best chains of thought (CoTs).

## Available Scorers

### ProbabilityFinalAnswerScorer

Scores paths based on the model's probability estimate for the final answer matching the ground truth:

```python
from cot_forge.reasoning.scorers import ProbabilityFinalAnswerScorer
from cot_forge.llm import LLMProvider

llm = LLMProvider(api_key="your-api-key")
scorer = ProbabilityFinalAnswerScorer(llm_provider=llm)
```

Parameters:
- `llm_provider`: Required LLM provider used for scoring
- `llm_kwargs`: Optional dictionary of additional LLM provider arguments

## Scorer Interface

All scorers implement the `BaseScorer` abstract base class:

```python
class BaseScorer:
    def __init__(
        self,
        name: str,
        description: str,
        llm_provider: LLMProvider = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs
    ):
        """Initialize the scorer."""
        pass

    def score(
        self,
        cot_list: list[dict],
        question: str,
        ground_truth_answer: str,
        **kwargs: Any
    ) -> dict[str, float]:
        """
        Score a list of reasoning paths.
        
        Args:
            cot_list: List of CoTs to be scored. Each CoT contains:
                     - strategy_name: Name of the strategy
                     - cot: Dictionary containing the chain of thought
            question: The question being answered
            ground_truth_answer: The true answer to compare against
            
        Returns:
            Dictionary mapping strategy names to float scores
        """
        pass
```

## Using with CoTBuilder

Scorers are typically passed to the CoTBuilder:

```python
from cot_forge.reasoning import CoTBuilder
from cot_forge.reasoning.scorers import ProbabilityFinalAnswerScorer

scorer = ProbabilityFinalAnswerScorer(llm_provider=llm)

builder = CoTBuilder(
    search_llm=llm,
    scorer=scorer
)
```

## Creating Custom Scorers 

You can create custom scorers by inheriting from `BaseScorer`:

```python
from cot_forge.reasoning.scorers import BaseScorer

class MyCustomScorer(BaseScorer):
    def __init__(self, weight_factor=1.0):
        super().__init__(
            name="My Custom Scorer",
            description="Scores based on custom logic"
        )
        self.weight_factor = weight_factor
        
    def score(
        self,
        cot_list: list[dict],
        question: str,
        ground_truth_answer: str,
        **kwargs
    ) -> dict[str, float]:
        scores = {}
        for cot_item in cot_list:
            strategy_name = cot_item["strategy_name"]
            cot = cot_item["cot"]
            
            # Example scoring logic:
            # Score based on number of reasoning steps
            score = len(cot) * self.weight_factor
            scores[strategy_name] = score
            
        return scores
```