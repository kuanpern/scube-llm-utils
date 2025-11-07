# scube-llm-utils

A collection of utility functions designed to simplify interactions with Large Language Models (LLMs). This library provides tools for prompt engineering, structured data extraction, text manipulation, and more.

**Please note:** This project is developed for personal use. There is **no guarantee** of stability, support, or suitability for any particular purpose. Use at your own risk.

## Key Features

*   **Structured LLM Processor:**  A generic class (`StructuredLLMProcessor`) for simplifying interactions with LLMs, including prompt templating with Jinja2, LLM invocation, and post-processing of results.
*   **File Loading:**  Recursively load files with specified extensions into a nested dictionary. This is helpful for organizing prompts, configurations, and other data. Includes optional parsing with custom functions for YAML, Jinja2, JSON, etc. (`load_files`).
*   **Function Argument Inspection:** Retrieve detailed information about function arguments, including names, types, default values, and parameter kinds (`get_function_arguments`).
*   **Nested Data Access:**  Convenient functions for accessing values from nested dictionaries or object attributes using key paths (`get_nested_value`, `get_nested_attribute`).
*   **Text Truncation:**  Safely truncate text to fit within a token limit, preserving meaningful content and adding an ellipsis (`truncate_text_to_token_limit`). Uses Hugging Face tokenizers.
*   **YAML Extraction:** Robustly extract YAML segments from text, even when embedded in Markdown code blocks.  Includes functions for parsing and filtering YAML data (`extract_yaml_segments`, `YAMLExtractor`, `LLMYAMLExtractor`).

## Contents

The library is organized into the following modules:

*   `scube_llm_utils/__init__.py`:  Exports the `StructuredLLMProcessor` class.
*   `scube_llm_utils/common.py`: Contains general-purpose utility functions for file loading, argument inspection, data access, and text manipulation.
*   `scube_llm_utils/types.py`: Defines the `StructuredLLMProcessor` class.
*   `scube_llm_utils/langgraph/`: Contains helper functions related to LangGraph.
    *   `scube_llm_utils/langgraph/helpers.py`: Defines the `generic_node_factory` function.
*   `scube_llm_utils/extractors/`: Contains classes and functions for extracting structured data.
    *   `scube_llm_utils/extractors/yaml.py`: Implements YAML extraction and parsing functionality.

## Usage Examples

**1. Loading files:**

```python
from scube_llm_utils.common import load_files
import yaml

data = load_files(
    file_extensions=[".yaml"], parser=yaml.safe_load, root_dir="./my_configs"  # Optional root directory
)

print(data)
```

```python
# 2. Truncating text:

from scube_llm_utils.common import truncate_text_to_token_limit
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "This is a long sentence that needs to be truncated because it exceeds the token limit."
max_tokens = 10

truncated_text = truncate_text_to_token_limit(text, tokenizer, max_tokens)
print(truncated_text)
```

3. Extracting YAML:

```python
from scube_llm_utils.extractors.yaml import YAMLExtractor

text = """
Here is some text with a YAML block:

```yaml
name: John Doe
age: 30
city: New York
```

And another YAML block:

```yaml
profession: Software Engineer
company: Example Inc.
extractor = YAMLExtractor()
yaml_data = extractor.extract_from_text(text)
print(yaml_data)
```

**4. Structured LLM Processor:**

```python
from scube_llm_utils.types import StructuredLLMProcessor
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

# Example setup (replace with your actual LLM and keys)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7) # Or use OpenAI(model_name="text-davinci-003", temperature=0.7)

instruction = "Summarize the following text: {{text}}"
description = "You are a helpful summarization assistant."

processor = StructuredLLMProcessor(
    name="text_summarizer",
    llm=llm,
    instruction=instruction,
    description=description
)

state = {"text": "This is a very long and complex text that needs to be summarized."}
output = processor(state)
print(output) # Expected output: {'output': 'A concise summary of the provided text.'}
```

## Installation
```bash
pip install scube-llm-utils
```

## Dependencies
* python >= 3.8
* transformers
* langchain
* PyYAML
* Jinja2
* tenacity
* markdown

## License
This project is for personal use and does not have a formal license. Use with caution and at your own risk.

## Contributing
As this is a personal project, contributions are not currently being accepted.