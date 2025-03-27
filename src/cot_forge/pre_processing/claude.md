# Structuring the Questions Module
I'll outline a high-level design for the questions (or problems) module that aligns with your needs for flexible premise/VGT pair creation from diverse data sources.

## Module Structure
```
cot_forge/pre_processing/
├── __init__.py
├── creator.py        # Core problem creation functionality
├── filters.py        # Filtering mechanisms
├── formatters.py     # Output formatting
├── loaders/          # Data source loaders
│   ├── __init__.py
│   ├── text_loader.py
│   ├── dataset_loader.py
│   ├── json_loader.py
│   └── csv_loader.py
├── schema.py         # Data schemas and validation
└── types.py          # Type definitions
```

## Core API Design

### 1. Problem Creation
The `ProblemCreator` class would be the main entry point:
```python
class ProblemCreator:
    def __init__(self, 
                 llm_provider,
                 extraction_template=None,
                 filter_template=None,
                 enable_filtering=True,
                 schema=None):
        """Initialize problem creator.
        
        Args:
            llm_provider: LLM provider for extraction and filtering
            extraction_template: Custom template for premise/VGT extraction
            filter_template: Custom template for filtering
            enable_filtering: Whether to apply filtering
            schema: Optional schema for validation
        """
        pass
    
    def from_text(self, 
                  texts, 
                  difficulty="medium",
                  max_problems=None,
                  metadata=None):
        """Create problems from raw text documents."""
        pass
    
    def from_json(self, 
                  json_data, 
                  text_field,
                  metadata_fields=None):
        """Create problems from JSON data."""
        pass
    
    def from_dataset(self, 
                    dataset, 
                    text_column,
                    metadata_columns=None):
        """Create problems from a HuggingFace dataset."""
        pass
    
    def from_csv(self, 
                 csv_path, 
                 text_column,
                 metadata_columns=None):
        """Create problems from a CSV file."""
        pass
    
    def from_multiple_choice(self,
                            questions,
                            answers=None,
                            difficulty="medium"):
        """Create open-ended problems from multiple-choice questions."""
        pass
```

### 2. Schema Validation
Using Pydantic for schema validation:
```python
# In schema.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

class VerifiableProblem(BaseModel):
    """Base class for verifiable problems."""
    id: str
    premise: str
    ground_truth: Any
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def create_schema(cls, **field_definitions):
        """Create a custom schema with specific field definitions."""
        return type("CustomVerifiableProblem", (cls,), field_definitions)

# Example custom schemas for legal domains
class LegalProblem(VerifiableProblem):
    """Schema for legal problems."""
    ground_truth: Dict[str, Any] = Field(
        ..., 
        description="A dictionary containing legal conclusion and reasoning"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "id": "case-1234",
                "premise": "Facts of the legal case...",
                "ground_truth": {
                    "legal_issues": ["Was there a breach of contract?", "..."],
                    "conclusion": "The court ruled in favor of the plaintiff.",
                    "reasons": ["The defendant failed to perform...", "..."]
                },
                "metadata": {
                    "source": "Supreme Court of X",
                    "date": "2020-01-01",
                    "jurisdiction": "Federal"
                }
            }
        }
```

### 3. Filtering API
```python
# In filters.py
class ProblemFilter:
    """Base class for problem filters."""
    def __init__(self, llm_provider, filter_template=None):
        """Initialize filter.
        
        Args:
            llm_provider: LLM provider for filtering
            filter_template: Custom template for filtering
        """
        pass
    
    def filter(self, text, metadata=None):
        """Filter a text based on criteria.
        
        Returns:
            tuple: (passed, reason)
        """
        pass
    
    def batch_filter(self, texts, metadata=None, parallel=True):
        """Filter multiple texts in batch."""
        pass

# Specific filters
class ReasoningDepthFilter(ProblemFilter):
    """Filter for ensuring sufficient reasoning depth."""
    pass

class AmbiguityFilter(ProblemFilter):
    """Filter for ensuring unambiguous ground truth."""
    pass

class MinimumDifficultyFilter(ProblemFilter):
    """Filter out problems that are too easy."""
    
    def __init__(self, llm_provider, verifier, threshold=0.8):
        """Initialize with a verifier to check if problems can be solved without CoT."""
        pass
```

### 4. Custom Extraction Templates
```python
# Usage example
from cot_forge.llm.prompts import PromptTemplate
from cot_forge.problems import ProblemCreator, schema

# Define a custom extraction template
legal_extraction_template = PromptTemplate("""
[Your legal extraction prompt here]
""")

# Define a custom schema for the outputs
LegalCase = schema.VerifiableProblem.create_schema(
    ground_truth=Dict[str, List[str]],
    parties=Dict[str, str]
)

# Create the problem creator
creator = ProblemCreator(
    llm_provider=my_llm,
    extraction_template=legal_extraction_template,
    schema=LegalCase
)

# Create problems
problems = creator.from_text(
    legal_texts,
    difficulty="hard",
    metadata={"domain": "contract_law"}
)
```

### Implementation Notes
1. Flexible Target Definition:
    * The schema system allows defining exactly what structure the premise and ground truth should have
    * Users can define custom schemas or use built-in ones

2. Primitive Type Support:
    * Pydantic integration handles validation of different types (string, list, dict, etc.)
    * The VerifiableProblem base class can be extended for domain-specific needs

3. Filtering Capabilities:
    * Modular filter system with composable filters
    * Pre-filtering and post-filtering options
    * Default filters for common scenarios
    * Custom filters via templates

4. Input Format Flexibility:
    * Modular loader system for different data sources
    * Consistent API regardless of the input format

5. Integration Points:
    * The system would integrate with the verifier module for quality control
    * Integration with the reasoning module to ensure problems are suitable for CoT training

This design provides a flexible, extensible framework that meets your requirements while maintaining a clean, intuitive API. The separation of concerns (loading, filtering, extraction, validation) makes it easy to adapt to different domains and use cases.