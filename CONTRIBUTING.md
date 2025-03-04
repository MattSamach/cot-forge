# Git Commit Message Schema for cot-forge
We encourage you to follow the commit message schema below when contributing to cot-forge. This helps maintain a clean and understandable project history.

### Commit Message Format
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
