"""Generation: prompts, Transformer generator, extractive fallback and grounding."""

from .generator import ExtractiveGenerator, Generator, HFGenerator, build_generator
from .grounding import (
    Citation,
    SentenceSupport,
    ground_answer,
    should_abstain_after,
    should_abstain_before,
    split_sentences,
    support_score,
)
from .prompt import build_chat_messages, build_prompt, format_passages

__all__ = [
    "Citation",
    "ExtractiveGenerator",
    "Generator",
    "HFGenerator",
    "SentenceSupport",
    "build_chat_messages",
    "build_generator",
    "build_prompt",
    "format_passages",
    "ground_answer",
    "should_abstain_after",
    "should_abstain_before",
    "split_sentences",
    "support_score",
]
