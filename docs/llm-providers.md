# LLM Providers

CoT Forge supports multiple LLM providers through a unified interface.
This allows you to use different models for generating reasoning chains.
You can easily switch between providers without changing the core logic of your application.
You can use the same provider for both the reasoning generation, verification, and scoring, or you can choose different providers for each step.

## Available Providers

### GeminiProvider

Interface for Google's Gemini models:

```python
from cot_forge.llm import GeminiProvider

gemini = GeminiProvider(
    api_key="your-google-api-key",
    model="gemini-2.0-flash"
)

response = gemini.generate("What is the capital of France?")
```

### OpenAIProvider

Interface for OpenAI models (GPT-3.5-Turbo, GPT-4, etc.):

```python
from cot_forge.llm import OpenAIProvider

openai = OpenAIProvider(
    api_key="your-openai-api-key",
    model="gpt-4"  # Specify model
)

response = openai.generate(
    "Explain the concept of gravity.",
    temperature=0.7
)
```

### AnthropicProvider

Interface for Anthropic Claude models:

```python
from cot_forge.llm import AnthropicProvider

claude = AnthropicProvider(
    api_key="your-anthropic-api-key",
    model="claude-3-opus-20240229"  # Specify Claude model
)

response = claude.generate(
    "Summarize the key points of quantum mechanics.",
    max_tokens=1000
)
```

## Common Interface

All providers implement the `LLMProvider` interface with these core methods:

```python
class LLMProvider:
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        pass
        
    def get_token_count(self, text: str) -> int:
        """Count tokens in the given text."""
        pass
        
    def get_token_limit(self) -> int:
        """Get the maximum token limit for the model."""
        pass
```

## Using with CoTBuilder

LLM providers are typically passed to the CoTBuilder for generating reasoning:

```python
from cot_forge.reasoning import CoTBuilder, NaiveLinearSearch
from cot_forge.llm import GeminiProvider

llm = GeminiProvider(api_key="your-api-key")

builder = CoTBuilder(
    search_llm=llm,  # Used for generating reasoning
    search=NaiveLinearSearch(),
    verifier=LLMJudgeVerifier(llm)  # Can use same or different model
)
```

## Provider-Specific Parameters

Each provider accepts model-specific parameters on initialization and when calling `generate()`:

```python
# OpenAI-specific parameters
response = openai.generate(
    prompt,
    temperature=0.7,
    top_p=1.0
)

# Gemini-specific parameters
response = gemini.generate(
    prompt,
    temperature=0.9,
    top_k=40,
)

# Claude-specific parameters
response = claude.generate(
    prompt,
    temperature=0.7,
    max_tokens=2000
)
```

## Creating Custom Providers

You can create custom providers by implementing the `LLMProvider` interface:

```python
from cot_forge.llm import LLMProvider

class MyCustomProvider(LLMProvider):
    def __init__(self, **kwargs):
        # Your initialization code
        pass
        
    def generate_completion(self, prompt: str, **kwargs) -> str:
        # Your generation logic
        return generated_text
        
    def estimate_input_tokens(self, text: str) -> int:
        # Your token counting logic
        return token_count
```