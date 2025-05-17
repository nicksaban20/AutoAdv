"""
Utilities for token calculation and management.
This module provides functionality for calculating tokens in prompts and responses.
"""
import tiktoken
from transformers import AutoTokenizer

def count_tokens_openai(text, model="gpt-3.5-turbo"):
    """
    Count the number of tokens in a text string using OpenAI's tokenizer.
    
    Args:
        text (str): The text to count tokens for
        model (str): The model to use for tokenization (default: gpt-3.5-turbo)
        
    Returns:
        int: The number of tokens in the text
    """
    if not text:
        return 0
        
    try:
        # Get the appropriate encoding based on the model
        if model.startswith("gpt-4"):
            encoding = tiktoken.encoding_for_model("gpt-4")
        elif model.startswith("gpt-3.5"):
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        else:
            encoding = tiktoken.get_encoding("cl100k_base")  # Default encoding
            
        # Count tokens
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Error counting tokens: {e}")
        # Fallback counting method
        return len(text.split()) * 1.3  # Rough approximation


def count_tokens_huggingface(text, model_name="gpt2"):
    """
    Count the number of tokens in a text string using Hugging Face's tokenizer.
    
    Args:
        text (str): The text to count tokens for
        model_name (str): The model to use for tokenization (default: gpt2)
        
    Returns:
        int: The number of tokens in the text
    """
    if not text:
        return 0
        
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokens = tokenizer.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Error counting tokens with HuggingFace: {e}")
        # Fallback counting method
        return len(text.split()) * 1.3  # Rough approximation


def estimate_tokens(text):
    """
    Provides a rough estimate of tokens in text using a simple heuristic.
    Useful as a fallback when tokenizers aren't available.
    
    Args:
        text (str): The text to estimate tokens for
        
    Returns:
        int: Estimated number of tokens
    """
    if not text:
        return 0
    
    # Split by whitespace and punctuation
    words = len(text.split())
    
    # Rough heuristic: 1.3 tokens per word for English
    return int(words * 1.3)


def summarize_token_usage(prompt_tokens, completion_tokens=0):
    """
    Summarize token usage for a request.
    
    Args:
        prompt_tokens (int): Number of tokens in the prompt
        completion_tokens (int): Number of tokens in the completion (optional)
        
    Returns:
        dict: Dictionary with token usage information
    """
    total_tokens = prompt_tokens + completion_tokens
    
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    }