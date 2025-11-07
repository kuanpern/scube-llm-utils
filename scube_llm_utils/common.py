import os
import glob
import inspect
from typing import Dict, Any, Callable, Optional, Union


def load_files(
    file_extensions: list[str],
    parser: Optional[Callable[[str], Any]] = None,
    root_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Recursively loads files with given extensions into a nested dictionary.
    Optionally parses file content using a custom parser (e.g., YAML, Jinja2, JSON).

    Args:
        file_extensions: List of file extensions to search for (e.g., [".yaml", ".jinja2"]).
        parser: Function to parse file content (e.g., `yaml.safe_load` for YAML).
                If `None`, reads files as plain text.
        root_dir: Root directory to search. Defaults to the script's directory.

    Returns:
        Nested dictionary where keys are path segments and values are parsed file contents.
    """
    # Set root directory (default: script's directory)
    if root_dir is None:
        root_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

    # Find all matching files
    files = []
    for ext in file_extensions:
        files.extend(glob.glob(os.path.join(root_dir, "**", f"*{ext}"), recursive=True))

    # Initialize output dictionary
    file_registry = {}

    for file in files:
        # Get relative path and remove extension
        relpath = os.path.relpath(file, root_dir)
        relpath_without_ext = os.path.splitext(relpath)[0]
        parts = relpath_without_ext.split(os.sep)

        # Traverse/create nested dictionary
        current_level = file_registry
        for part in parts[:-1]:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]

        # Read and optionally parse the file
        with open(file, "r") as f:
            content = f.read()
            current_level[parts[-1]] = parser(content) if parser else content

    return file_registry
# end def



def get_function_arguments(
    func: Callable
) -> Dict[str, Dict[str, Union[type, Any, None]]]:
    """
    Retrieve the valid arguments of a target function, including their names,
    default values, and type annotations (if available).

    Args:
        func (Callable): The target function to inspect.

    Returns:
        Dict[str, Dict[str, Union[type, Any, None]]]:
            A dictionary where keys are argument names, and values are dictionaries
            containing:
            - 'type': The type annotation (if available), otherwise `None`.
            - 'default': The default value (if available), otherwise `inspect.Parameter.empty`.
            - 'kind': The kind of parameter (e.g., positional, keyword, var-positional, var-keyword).

    Example:
        >>> def example(a: int, b: str = "hello", *args, **kwargs): pass
        >>> get_function_arguments(example)
        {
            'a': {'type': <class 'int'>, 'default': <empty>, 'kind': 'POSITIONAL_OR_KEYWORD'},
            'b': {'type': <class 'str'>, 'default': 'hello', 'kind': 'POSITIONAL_OR_KEYWORD'},
            'args': {'type': None, 'default': <empty>, 'kind': 'VAR_POSITIONAL'},
            'kwargs': {'type': None, 'default': <empty>, 'kind': 'VAR_KEYWORD'}
        }
    """
    signature = inspect.signature(func)
    arguments = {}

    for name, param in signature.parameters.items():
        arguments[name] = {
            'type': param.annotation if param.annotation != inspect.Parameter.empty else None,
            'default': param.default if param.default != inspect.Parameter.empty else None,
            'kind': param.kind.name.lower()
        }

    return arguments
# end def

def get_nested_value(data, keys, default_value=None):
    """Get value from nested dict
    
    Args:
        data (dict): The input dictionary.
        keys (list): A list of keys to access the nested value.
        default_value (any): The default value to return if the key is not found.

    Returns:
        any: The nested value or the default value.
    """
    for key in keys:
        if key in data:
            data = data[key]
        else:
            return default_value
        # end if
    # end for
    return data
# end def

def get_nested_attribute(obj, attribute_path, default_value=None):
    """Get value from nested object attribute    """
    parts = attribute_path.split('.')
    current_obj = obj
    for part in parts:
        if hasattr(current_obj, part):
            current_obj = getattr(current_obj, part)
        else:
            return default_value
    return current_obj
# end def


import logging
from transformers import PreTrainedTokenizer

# --- Text Truncation Function ---
def truncate_text_to_token_limit(
    text: str,
    tokenizer: PreTrainedTokenizer,
    max_tokens: int,
    ellipsis: str = " ...",
    verbose: bool = False,
    logger: logging.Logger = None

) -> str:
    """
    Truncates input text to fit within a specified token limit using a tokenizer.
    Preserves as much meaningful content as possible while respecting the token constraint.

    Args:
        text: Input text to truncate.
        tokenizer: HuggingFace tokenizer to measure token count.
        max_tokens: Maximum allowed tokens in the output.
        ellipsis: String appended to indicate truncation (default: " ...").
        verbose: If True, prints truncation progress (default: False).
        logger: logger

    Returns:
        Truncated text with ellipsis appended, guaranteed to be <= max_tokens.

    Raises:
        ValueError: If max_tokens is non-positive or text is empty after processing.
    """

    if logging is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
    # end if

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    
    stripped_text = text.strip()
    if not stripped_text:
        return stripped_text

    # Initial token count to check if truncation is needed
    try:
        initial_tokens = tokenizer(stripped_text, return_tensors="pt")['input_ids']
    except Exception as e:
        logger.error(f"Tokenizer error: {e}")
        return stripped_text # Return original if tokenizer fails

    if initial_tokens.shape[-1] <= max_tokens:
        return stripped_text

    # Reduce text proportionally based on token count
    words = stripped_text.split()
    if not words:
        return stripped_text # Should not happen if stripped_text is not empty

    # Estimate words to keep, ensuring at least one word remains.
    # Calculate target word count based on ratio of max_tokens to initial_tokens.
    # Ensure target_word_count is at least 1 to prevent empty output.
    estimated_target_words = max(1, int(len(words) * max_tokens / initial_tokens.shape[-1]))
    words_to_keep = words[:estimated_target_words]

    # Iteratively refine by removing words until token limit is met
    current_text = " ".join(words_to_keep)
    while len(tokenizer(current_text, return_tensors="pt")['input_ids']) > max_tokens:
        if len(words_to_keep) <= 1: # Cannot truncate further
            break
        words_to_keep.pop() # Remove the last word
        current_text = " ".join(words_to_keep)
        if verbose:
            logger.info(f"Truncating... Remaining words: {len(words_to_keep)}")

    # Add ellipsis if truncation occurred and there's space
    final_text_with_ellipsis = current_text
    if len(words) > len(words_to_keep): # If actual truncation happened
        try:
            # Check if ellipsis fits
            ellipsis_tokens = len(tokenizer(ellipsis, return_tensors="pt")['input_ids'])
            if len(tokenizer(current_text, return_tensors="pt")['input_ids']) + ellipsis_tokens <= max_tokens:
                final_text_with_ellipsis += ellipsis
            else:
                # If ellipsis doesn't fit, try to truncate current_text more to make space
                while len(tokenizer(final_text_with_ellipsis, return_tensors="pt")['input_ids']) + ellipsis_tokens > max_tokens:
                    if len(words_to_keep) <= 1:
                        break # Cannot make space for ellipsis
                    words_to_keep.pop()
                    final_text_with_ellipsis = " ".join(words_to_keep)
                if len(final_text_with_ellipsis) > 0: # Only add if there's text left
                    final_text_with_ellipsis += ellipsis

        except Exception as e:
            logger.error(f"Tokenizer error while adding ellipsis: {e}")
            # If ellipsis addition fails, return the truncated text without it.

    # Ensure final output adheres to max_tokens. This is a safeguard.
    # If the logic above somehow fails, this re-truncates to the absolute limit.
    final_tokens = tokenizer(final_text_with_ellipsis, return_tensors="pt")['input_ids']
    if final_tokens.shape[-1] > max_tokens:
        return tokenizer.decode(final_tokens[0, :max_tokens], skip_special_tokens=True)
        
    return final_text_with_ellipsis
# end def