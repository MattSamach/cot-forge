# Core Concepts

## Chain of Thought (CoT)

Chain of Thought is a technique where language models generate step-by-step reasoning before arriving at an answer. This makes the reasoning process transparent and improves accuracy on complex tasks.

## Components Architecture

CoT Forge uses these primary components:

### 1. CoTBuilder

The central object that coordinates the CoT generation process:

### 2. Reasoning Strategies
Reasoning strategies implement a particular strategy at a point in the reasoning chain. For example, a reasoning strategy might be to break down a complex question into simpler sub-questions, to backtrack in the reasoning process, or to explore alternative hypotheses.


### 3. Reasoning Nodes

Each node in a reasoning path represents a step in the reasoning chain. Each node has one reasoning strategy applied to. It can have child nodes representing further reasoning steps or branches. 

See more here: [Reasoning Nodes](./reasoning-nodes.md)

### 4. Search Algorithms

Search algorithms determine how the reasoning space is explored. For more details, see [Search Algorithms](./search-algorithms.md).

- **Naive Linear Search**: Generates a single reasoning path by naively selecting reasoning strategies. For more details, see[Naive Linear Search](naivelinearsearch.md)

- **Beam Search**: Explores multiple reasoning branches simultaneously, keeping the best ones based on a scoring function. For more details, see [Beam Search](beamsearch.md)

### 4. Verifiers

Verifiers evaluate whether a reasoning path has led to a correct answer:

- **LLMJudgeVerifier**: Uses an LLM to judge correctness of an answer.
- **ExactMatchVerifier**: Direct string matching (None yet implemented)

### 5. Scorers

Scorers rank different reasoning paths by quality:

- **ProbabilityFinalAnswerScorer**: Scores based on answer likelihood according to an LLM's evaluation.

### 6. Post-Processing

Transforms structured CoT into formats suitable for training:

- **ReasoningProcessor**: Converts cot, which is structured as a list of dicts, into natural language text.

## Data Flow

1. CoTBuilder receives a question and ground truth
2. Search strategies explore the reasoning space
3. Verifiers check correctness of reasoning
4. Scorers rank the reasoning paths
5. The best reasoning paths are selected
6. Post-processing converts reasoning into training data