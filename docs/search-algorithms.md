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

## Built-in Search Algorithms

CoT Forge provides several search algorithms:
* [NaiveLinearSearch](./naivelinearsearch.md)
* [BeamSearch](./beamsearch.md)

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

## Serialization

Search algorithms can be serialized for reproducibility:

```python
# Serialize search configuration
search_config = search.to_dict()

# Deserialize later
restored_search = BeamSearch.from_dict(search_config)
```