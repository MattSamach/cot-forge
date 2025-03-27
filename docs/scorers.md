# Scorers

Scorers are used to rank different reasoning paths by quality, helping CoT Forge select the best chains of thought.

## Available Scorers

### ProbabilityFinalAnswerScorer

Scores paths based on the model's probability estimate for the final answer:

```python
from cot_forge.reasoning.scorers import ProbabilityFinalAnswerScorer
from cot_forge.llm import GeminiProvider

llm = GeminiProvider(api_key="your-api-key")
scorer = ProbabilityFinalAnswerScorer(llm_provider=llm)
```

Parameters:
- `llm_provider`: LLM provider used for scoring

## Scorer Interface

All scorers implement the `ReasoningScorer` interface:

```python
class ReasoningScorer:
    def score(
        self, 
        question: str, 
        node: ReasoningNode
    ) -> float:
        """
        Score a reasoning path.
        
        Args:
            question: The original question
            node: The reasoning node to score
            
        Returns:
            float: Score (higher is better)
        """
        pass
```

## Using with CoTBuilder

Scorers are typically passed to the CoTBuilder to rank reasoning paths:

```python
from cot_forge.reasoning import CoTBuilder, SimpleBeamSearch
from cot_forge.reasoning.scorers import ProbabilityFinalAnswerScorer

scorer = ProbabilityFinalAnswerScorer(llm_provider=llm)

builder = CoTBuilder(
    search_llm=llm,
    search=SimpleBeamSearch(),
    verifier=verifier,
    scorer=scorer
)
```

## Creating Custom Scorers

You can create custom scorers by implementing the `ReasoningScorer` interface:

```python
from cot_forge.reasoning.scorers import ReasoningScorer
from cot_forge.reasoning.types import ReasoningNode

class MyCustomScorer(ReasoningScorer):
    def __init__(self, weight_factor=1.0):
        self.weight_factor = weight_factor
        
    def score(self, question: str, node: ReasoningNode) -> float:
        # Your scoring logic
        # Return a float score (higher is better)
        
        # Example: Score based on reasoning length and keywords
        cot = node.get_full_cot()
        
        # Higher score for more detailed reasoning (more steps)
        step_score = len(node.thoughts) * 0.5
        
        # Higher score for mentioning certain keywords
        keyword_score = sum(1.0 for keyword in ['because', 'therefore', 'since'] 
                          if keyword in cot.lower())
                          
        return (step_score + keyword_score) * self.weight_factor
```

## Using Scorers for Selection

In search strategies that explore multiple paths, scorers help select the best nodes:

```python
# Inside a search strategy implementation
def search(self, question, llm_provider, verifier, scorer):
    # Generate multiple reasoning paths
    nodes = [generate_node(question) for _ in range(5)]
    
    if scorer:
        # Score each node
        scores = [scorer.score(question, node) for node in nodes]
        
        # Select best node based on scores
        best_node = nodes[scores.index(max(scores))]
    else:
        # Default selection if no scorer provided
        best_node = nodes[0]
```