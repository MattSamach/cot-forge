# Search Algorithms

Search algorithms in CoT Forge define how the reasoning space is explored to find valid reasoning paths. While reasoning strategies determine the approach at individual nodes, search algorithms control the overall path exploration process.

## Core Concept

A search algorithm orchestrates how reasoning nodes are generated, evaluated, and connected to form cohesive chains of thought. It determines which nodes to expand, when to terminate exploration, and how to select the best resulting reasoning path.

## Base Search Interface

All search algorithms implement the `BaseSearch` abstract class:

```python
from cot_forge.reasoning.search import BaseSearch

class MySearchAlgorithm(BaseSearch):
    def __init__(self, max_depth=3, some_param=1.0):
        self.max_depth = max_depth
        self.some_param = some_param
    
    def _search(
        self,
        question: str,
        ground_truth_answer: str,
        search_llm,
        verifier,
        scorer=None,
        strategy_registry=default_strategy_registry,
        llm_kwargs=None,
        **kwargs
    ) -> SearchResult:
        # Implement search logic here
        # Return a SearchResult object
        pass
    
    def to_dict(self) -> dict:
        return {
            "type": "my_search_algorithm",
            "max_depth": self.max_depth,
            "some_param": self.some_param
        }
    
    @classmethod
    def from_dict(cls, config: dict):
        return cls(
            max_depth=config.get("max_depth", 3),
            some_param=config.get("some_param", 1.0)
        )
```

## Built-in Search Algorithms

CoT Forge provides several search algorithms:

### NaiveLinearSearch

The simplest search strategy that generates a single reasoning path by randomly selecting strategies at each node:

```python
from cot_forge.reasoning import NaiveLinearSearch

search = NaiveLinearSearch(max_depth=3)
```

Parameters:
- `max_depth`: Maximum reasoning steps before terminating (default: 3)
- `on_error`: Error handling strategy ("continue", "raise", or "retry")
- `system_prompt`: Optional custom system prompt for the LLM

### BeamSearch

Explores multiple reasoning paths simultaneously, keeping the most promising ones:

```python
from cot_forge.reasoning import BeamSearch

search = BeamSearch(
    max_depth=3,
    beam_width=3,
    branching_factor=2
)
```

Parameters:
- `max_depth`: Maximum depth of reasoning tree (default: 3)
- `beam_width`: Number of paths to maintain at each step (default: 3)
- `branching_factor`: Number of children to generate and evaluate per node (default: 2)
- `strategy_selector`: How to select strategies at each node (default: RandomStrategySelector)

### MonteCarloTreeSearch
(Not implemented yet)
Uses Monte Carlo simulation to explore the reasoning space:

```python
from cot_forge.reasoning import MonteCarloTreeSearch

search = MonteCarloTreeSearch(
    max_iterations=10,
    exploitation_constant=1.0
)
```

Parameters:
- `max_iterations`: Maximum number of iterations (default: 10)
- `exploitation_constant`: Controls exploration vs exploitation (default: 1.0)

## Search Process

The general search process follows these steps:

1. **Initialization**: Create initial node(s) using an initial strategy
2. **Expansion**: Generate candidate child nodes using various reasoning strategies
3. **Scoring**: Evaluate nodes using a scorer (if the search selects for a subset of candidate nodes)
4. **Selection**: Choose which nodes to add to the chains
5. **Verification**: Verify if the child node has reached the success criteria
   - If yes, mark it as a successful terminal node
   - If no, continue expanding the node
6. **Termination**: Stop when success criteria are met or termination conditions reached
7. **Return**: Package the results into a SearchResult object

## Search Result

The `SearchResult` class contains the outcome of a search:

```python
# Get all terminal nodes, including possibly failed ones
terminal_nodes = search_result.terminal_nodes
# Get just successful terminal nodes
successful_nodes = search_result.get_successful_terminal_nodes()
# Get just the successful answers (strings)
successful_answers = search_result.get_successful_final_answers()
# Get all nodes in reasoning chain for one of the successful nodes
successful_chain = successful_nodes[0].get_full_node_chain()
# Get the full chain of thought for the successful node as a dictionary
successful_chain_dict = successful_nodes[0].get_full_cot()
```

SearchResult contains:
- All nodes generated during the search
- Terminal nodes (leaf nodes in the search tree)
- Success/failure status
- Metadata about the search process

## Node Creation and Verification

Search algorithms use helper methods for common operations:

```python
# Create a reasoning node
node = self.create_node(
    strategy=strategy,
    prompt=prompt,
    response=response,
    cot=extracted_cot,
    parent=parent_node
)

# Verify a node to check if it meets the success criteria
success, error = self.verify_node(
    node=node,
    question=question,
    ground_truth_answer=ground_truth_answer,
    verifier=verifier
)
```

## Using Search Algorithms

Search algorithms are typically used with the CoTBuilder:

```python
from cot_forge.reasoning import CoTBuilder, BeamSearch

builder = CoTBuilder(
    search_llm=llm,
    search=BeamSearch(max_depth=3, beam_width=2),
    verifier=verifier
)

result = builder.build(
    question="What is the capital of France?",
    ground_truth_answer="Paris"
)
```

## Comparing Search Algorithms

You can compare different algorithms on the same question:

```python
# Define different search algorithms
linear = NaiveLinearSearch(max_depth=3)
beam = BeamSearch(max_depth=3, beam_width=2, branching_factor=2)
mcts = MonteCarloTreeSearch(max_iterations=10)

# Create verifier and scorer
verifier = LLMJudgeVerifier(llm_provider = llm)
scorer = ProbabilityFinalAnswerScorer(llm_provider = llm)

# Create builders using each algorithm
builders = {
    "linear": CoTBuilder(search_llm=llm, search=linear, verifier=verifier),
    "beam": CoTBuilder(search_llm=llm, search=beam, verifier=verifier, scorer=scorer),
    "mcts": CoTBuilder(search_llm=llm, search=mcts, verifier=verifier, scorer=scorer)
}

# Compare results
results = {}
for name, builder in builders.items():
    results[name] = builder.build(question, ground_truth_answer)
    print(f"{name} success: {results[name].success}")
```

## Serialization

Search algorithms can be serialized for reproducibility:

```python
# Serialize search configuration
search_config = search.to_dict()

# Deserialize later
restored_search = BeamSearch.from_dict(search_config)
```

## Custom Search Algorithms

You can implement custom search algorithms by extending BaseSearch:

```python
class HybridSearch(BaseSearch):
    def __init__(self, max_depth=3, temperature_schedule=None):
        self.max_depth = max_depth
        self.temperature_schedule = temperature_schedule or [1.0, 0.8, 0.5]
    
    def _search(self, question, ground_truth_answer, search_llm, verifier, **kwargs):
        # Custom search implementation
        # ...
        return SearchResult(root_nodes, terminal_nodes)
    
    def to_dict(self):
        return {
            "type": "hybrid_search",
            "max_depth": self.max_depth,
            "temperature_schedule": self.temperature_schedule
        }
    
    @classmethod
    def from_dict(cls, config):
        return cls(
            max_depth=config.get("max_depth", 3),
            temperature_schedule=config.get("temperature_schedule")
        )
```