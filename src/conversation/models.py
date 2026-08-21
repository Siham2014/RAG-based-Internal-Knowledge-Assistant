from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class ConversationMessage:
    role: MessageRole
    content: str
    created_at: datetime

    @classmethod
    def create(cls, role: MessageRole, content: str) -> "ConversationMessage":
        return cls(role=role, content=content, created_at=datetime.now(timezone.utc))


@dataclass(frozen=True)
class QueryRewriteResult:
    original_query: str
    rewritten_query: str
    changed: bool
    provider: str | None = None
    model_name: str | None = None

