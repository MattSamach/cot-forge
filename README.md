# CoT Forge
A Python library for generating high-quality Chain of Thought (CoT) reasoning data for training and fine-tuning large language models.

## Overview
CoT Forge helps you create synthetic training data that includes complex reasoning chains, enabling LLMs to learn more robust and transparent reasoning capabilities. Inspired by research like [HuatuoGPT-o1](https://github.com/FreedomIntelligence/HuatuoGPT-o1), this library implements a flexible framework for:

* Creating verifiable question/answer pairs
* Finding optimal reasoning paths through tree search
* Formatting natural-sounding reasoning for supervised fine-tuning

## Installation
## Quick Start
## Features
* Flexible problem creation: Convert various sources (text, multiple-choice questions, etc.) into verifiable problems
* Quality filtering: Ensure problems are appropriate for reasoning training
* Multiple reasoning strategies: Implement diverse approaches to find optimal reasoning paths
* Extensible framework: Add custom reasoning strategies, verifiers, and more

## Documentation

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Poetry
Poetry is used for dependency management and packaging in Python. To install Poetry, follow the instructions [here](https://python-poetry.org/docs/#installation).

To add a dependency:
```bash
poetry add <package-name>
```

To install all dependencies:
```bash
poetry install
```



## Linting
Using Ruff as linter. Installation instructions can be found [here.](https://github.com/astral-sh/ruff?tab=readme-ov-file#getting-started)

To run Ruff as a linter, try any of the following:
```bash
ruff check                          # Lint all files in the current directory (and any subdirectories).
ruff check path/to/code/            # Lint all files in `/path/to/code` (and any subdirectories).
ruff check path/to/code/*.py        # Lint all `.py` files in `/path/to/code`.
ruff check path/to/code/to/file.py  # Lint `file.py`.
ruff check @arguments.txt           # Lint using an input file, treating its contents as newline-delimited command-line arguments.
```

Or, to run Ruff as a formatter:
```bash
ruff format                          # Format all files in the current directory (and any subdirectories).
ruff format path/to/code/            # Format all files in `/path/to/code` (and any subdirectories).
ruff format path/to/code/*.py        # Format all `.py` files in `/path/to/code`.
ruff format path/to/code/to/file.py  # Format `file.py`.
ruff format @arguments.txt           # Format using an input file, treating its contents as newline-delimited command-line arguments.
```