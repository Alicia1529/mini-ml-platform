"""Deterministic toy moderation; these rules are not a real safety classifier."""

from dataclasses import dataclass
from random import Random

UNSAFE_KEYWORDS = ("scam", "password", "hack")
MESSAGE_TEMPLATES = (
    "Great game, everyone!",
    "Meet me near the blue tower.",
    "Can someone help with this level?",
    "I like your new avatar.",
    "That obstacle course was fun!",
    "This is a SCAM offer.",
    "Send me your password for a reward.",
    "Download this hack to win.",
)


@dataclass(frozen=True)
class Message:
    id: int
    text: str


@dataclass(frozen=True)
class ModerationResult:
    message_id: int
    flagged: bool
    matched_keywords: tuple[str, ...]


def generate_messages(count: int = 1_000, seed: int = 42) -> list[Message]:
    """Use a local seeded RNG so runs share the same input and stable IDs."""
    if count < 0:
        raise ValueError("count must be non-negative")
    rng = Random(seed)
    return [Message(i, rng.choice(MESSAGE_TEMPLATES)) for i in range(count)]


def moderate_message(message: Message) -> ModerationResult:
    """Flag case-insensitive substrings, returning matches in keyword order."""
    text = message.text.casefold()
    matches = tuple(keyword for keyword in UNSAFE_KEYWORDS if keyword in text)
    return ModerationResult(message.id, bool(matches), matches)
