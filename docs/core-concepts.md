# Core Concepts

CoT Forge is built around several key concepts that work together to generate high-quality Chain of Thought (CoT) reasoning:

## Chain of Thought (CoT)

Chain of Thought is a technique where language models generate step-by-step reasoning before arriving at an answer. This makes the reasoning process transparent and improves accuracy on complex tasks.

## Components Architecture

CoT Forge uses these primary components:

### 1. CoTBuilder

The central component that coordinates the CoT generation process:

```python
builder = CoTBuilder(
    search_llm=llm_provider,
    search=search_strategy,
    verifier=verifier,
    scorer=scorer,
    strategy_reg=strategy_registry
)
```

### 2. Search Strategies

Search strategies determine how the reasoning space is explored:

- **NaiveLinearSearch**: Generates a single reasoning path
- **BeamSearch**: Explores multiple reasoning branches simultaneously
- **MonteCarloTreeSearch**: Uses Monte Carlo simulation for exploration (not yet implemented)

### 3. Reasoning Nodes

Each node in a reasoning path represents a step in the reasoning process:

```python
# A reasoning node structure (simplified)
{
    "chain of thought (cot)": ["First, I need to...", "Next, I consider..."],
    "final conclusion": "The final answer is...",
    "children": [<child_nodes>]
    "parent": <parent_node>
}
```

### 4. Verifiers

Verifiers evaluate whether a reasoning path has led to a correct answer:

- **LLMJudgeVerifier**: Uses an LLM to judge correctness
- **ExactMatchVerifier**: Direct string matching (None yet implemented)

### 5. Scorers

Scorers rank different reasoning paths by quality:

- **ProbabilityFinalAnswerScorer**: Scores based on answer likelihood

### 6. Post-Processing

Transforms structured CoT into formats suitable for training:

- **ReasoningProcessor**: Converts rigid reasoning into natural language

## Data Flow

1. CoTBuilder receives a question and ground truth
2. Search strategies explore the reasoning space
3. Verifiers check correctness of reasoning
4. Scorers rank the reasoning paths
5. The best reasoning paths are selected
6. Post-processing converts reasoning into training data

## Example Flow

```
Question → CoTBuilder → Search Strategy → 
  → Reasoning Paths → Verification → 
    → Scoring → Best Reasoning → Post-Processing → 
      → Training Data
```