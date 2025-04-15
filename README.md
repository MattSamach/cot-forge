# CoT Forge ✨

A Python library for generating high-quality Chain of Thought (CoT) ⛓️ reasoning data for training and fine-tuning large language models. 🧠

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Python](https://img.shields.io/badge/python-≥3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

CoT Forge helps you create synthetic training data that includes complex reasoning chains, enabling LLMs to learn more robust and transparent reasoning capabilities. 🚀 Inspired by research like [HuatuoGPT-o1](https://github.com/FreedomIntelligence/HuatuoGPT-o1), this library implements a flexible framework for:

* Creating verifiable question/answer pairs 🎯
* Finding optimal reasoning paths through tree search algorithms 🌳
* Reformatting reasoning chains into natural language ✍️

## Installation

CoT Forge is compatible with Python 3.10 and above. You can install it via pip or clone the repository directly. ⬇️

```bash
# Install using pip
pip install cot-forge

# Or directly from the repository
pip install git+https://github.com/MattSamach/cot-forge
```

## Quick Start 🏃‍♀️

```python
from cot_forge.llm import GeminiProvider
from cot_forge.reasoning import CoTBuilder, NaiveLinearSearch
from cot_forge.reasoning.verifiers import LLMJudgeVerifier

# Initialize LLM provider
llm = GeminiProvider(api_key="your-api-key")

# Setup CoT builder with essential components
builder = CoTBuilder(
    search_llm=llm,
    search=NaiveLinearSearch(max_depth=3),
    verifier=LLMJudgeVerifier(llm)
)

# Generate CoT reasoning for a question
question = "What is the capital of France, and why did it become the capital?"
ground_truth = "Paris is the capital of France. It became the capital due to its central location and political importance."

search_result = builder.build(
    question=question,
    ground_truth_answer=ground_truth,
)

# Access the best reasoning paths
successful_nodes = search_result.get_successful_terminal_nodes()
# Get the full chain of thought for a successful node as a dictionary
success_node = successful_nodes[0]
print(success_node.get_full_cot())

# Process reasoning into natural language for training
from cot_forge.post_processing.reasoning_processor import ReasoningProcessor

processor = ReasoningProcessor(
    llm_provider=llm, 
    dataset_name="my_dataset",
    search_name="naive_search"
)

processed = processor.process(
    question=question,
    cot=success_node.get_full_cot()
)

print(processed)
```

## Core Features ⚙️

* **Multiple LLM Providers**: Support for OpenAI, Anthropic Claude, and Google Gemini models
* **Flexible Search Strategies**: Choose from beam search, naive linear, and more
* **Quality Verification**: Built-in verifiers to ensure reasoning is correct
* **Result Scoring**: Various scoring methods to select the best reasoning paths
* **Natural Language Reformatting**: Convert structured reasoning into natural language for training
* **Persistence**: Save and resume reasoning generation for large datasets
* **Extensibility**: Easily add custom reasoning strategies, verifiers, and more

## Documentation 📚

For detailed documentation, see the [docs folder](./docs/):

- [Core Concepts](./docs/core-concepts.md)
- [LLM Providers](./docs/llm-providers.md)
- [Strategies](./docs/strategies.md)
- [Search Algorithms](./docs/search-algorithms.md)
- [Verification](./docs/verification.md)
- [Post-Processing](./docs/post-processing.md)
- [Persistence](./docs/persistence.md)
- [Examples](./examples/README.md)

## License 📝
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Contributing 🤝

Please see [CONTRIBUTING](./CONTRIBUTING.md) for guidelines on contributing to this project.