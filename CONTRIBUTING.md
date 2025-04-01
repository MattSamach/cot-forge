### Commit Message Format
We encourage you to follow the commit message schema below when contributing to cot-forge. This helps maintain a clean and understandable project history.
```
<type>(<scope>): <short summary>

<optional body>

<optional footer>
```
### Types
* **feat**: A new feature
* **fix**: A bug fix
* **docs**: Documentation only changes
* **style**: Changes that don't affect code functionality (formatting, etc)
* **refactor**: Code changes that neither fix bugs nor add features
* **perf**: Performance improvements
* **test**: Adding or correcting tests
* **build**: Changes to build system or dependencies
* **ci**: Changes to CI configuration
* **chore**: Other changes that don't modify src or test files

### Scope
The scope should be the component or module being modified:

* **llm**: Changes to LLM providers
* **reasoning**: Changes to reasoning module
* **problems**: Changes to problem generation module
* **docs**: Documentation changes
* **utils**: Utility functions
* **config**: Configuration-related changes
* **deps**: Dependency management


### Examples
```
feat(llm): add Claude provider implementation

Implements Claude LLM provider with standard retries and error handling.
```

```
fix(llm): correct token limit handling in Gemini provider
```

```
refactor(reasoning): simplify prompt construction logic
```

### Rules to Follow
1. Keep the summary line under 50 characters
2. Use imperative mood ("add" not "added" or "adds")
3. Don't end the summary line with a period
4. Separate subject from body with a blank line
5. Use the body to explain what and why, not how
6. Reference issues and pull requests in the footer

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
This project uses Ruff as linter.
To create a pull request, run the linter to ensure that your code adheres to the project's style guidelines. 
Installation instructions can be found [here.](https://github.com/astral-sh/ruff?tab=readme-ov-file#getting-started)

To run Ruff as a linter, try any of the following:
```bash
poetry run ruff check                          # Lint all files in the current directory (and any subdirectories).
poetry run ruff check path/to/code/            # Lint all files in `/path/to/code` (and any subdirectories).
poetry run ruff check path/to/code/*.py        # Lint all `.py` files in `/path/to/code`.
poetry run ruff check path/to/code/to/file.py  # Lint `file.py`.
poetry run ruff check @arguments.txt           # Lint using an input file, treating its contents as newline-delimited command-line arguments.
poetry run ruff check --fix path/to/code/      # Lint and automatically fix all files in `/path/to/code` (and any subdirectories).
```

## Testing
This project uses pytest for testing.
To create a pull request, run all tests to ensure that the code does not break existing functionality.
Using pytest for testing. Installation instructions can be found [here.](https://docs.pytest.org/en/latest/getting-started.html)

To run the tests, you can use the following commands:
```bash
poetry run pytest # To run all tests
poetry run pytest tests/test_file.py # To run a specific test file
poetry run pytest tests/test_file.py::test_function # To run a specific test function
```
