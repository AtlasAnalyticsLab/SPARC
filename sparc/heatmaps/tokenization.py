"""
Tokenization utilities for different CLIP-compatible tokenizers.

This module provides robust token conversion and word merging functions
that work with various tokenizer implementations.
"""

import torch
from typing import List, Sequence, Tuple, Any


def ids_to_tokens(tokenizer: Any, ids: Sequence[int]) -> List[str]:
    """
    Robustly convert numeric IDs to readable token strings for any
    CLIP/open_clip tokenizer – no Hugging Face required.
    
    Args:
        tokenizer: Any CLIP-compatible tokenizer
        ids: Sequence of token IDs
        
    Returns:
        List of token strings
    """
    # 1. The nice way (HF / many custom tokenizers)
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        return tokenizer.convert_ids_to_tokens(ids)

    # 2. open_clip's Tokenizer: expose `decoder` dict
    if hasattr(tokenizer, "decoder"):
        return [tokenizer.decoder[int(i)] for i in ids]

    # 3. tokenizer has .decode(list[int]) (openai CLIP etc.)
    if hasattr(tokenizer, "decode"):
        return [tokenizer.decode([int(i)]) for i in ids]

    # 4. worst case – just return raw ints as strings
    return [str(int(i)) for i in ids]


def tokens_to_words(tokens: Sequence[str], scores: torch.Tensor) -> Tuple[List[str], torch.Tensor]:
    """
    Merge open_clip BPE pieces into words.
    A token that ends with '</w>' finishes a word.
    Scores are averaged over the pieces that form the word.
    Special tokens and padding are discarded.
    
    Args:
        tokens: Sequence of token strings
        scores: Tensor of scores corresponding to tokens
        
    Returns:
        Tuple of (words, aggregated_scores)
    """
    words, agg_scores = [], []
    current, buf = "", []

    for t, s in zip(tokens, scores):
        # Skip padding & special tokens
        if t.startswith("<start_of_text>") or t.startswith("<end_of_text>"):
            continue
        if set(t) == {"!"}:  # padding: '!!!!!!!!!!!'
            continue

        # Strip the BPE end-of-word marker
        end_word = t.endswith("</w>")
        piece = t.replace("</w>", "")
        current += piece
        buf.append(s)

        if end_word:  # word boundary reached
            words.append(current)
            agg_scores.append(torch.stack(buf).mean())
            current, buf = "", []

    # Handle any tail fragment (shouldn't happen for well-formed input)
    if current:
        words.append(current)
        agg_scores.append(torch.stack(buf).mean())

    return words, torch.tensor(agg_scores) 