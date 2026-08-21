from __future__ import annotations

from collections import OrderedDict
from threading import RLock
from typing import Protocol

from src.conversation.models import ConversationMessage


class ConversationMemory(Protocol):
    def get_history(self, conversation_id: str) -> tuple[ConversationMessage, ...]: ...

    def append_message(
        self,
        conversation_id: str,
        message: ConversationMessage,
    ) -> None: ...


class InMemoryConversationMemory:
    """Thread-safe bounded memory replaceable by Redis or PostgreSQL later."""

    def __init__(self, max_conversations: int = 1000, max_messages: int = 12) -> None:
        if max_conversations <= 0 or max_messages <= 0:
            raise ValueError("Conversation memory limits must be positive.")
        self._max_conversations = max_conversations
        self._max_messages = max_messages
        self._items: OrderedDict[str, list[ConversationMessage]] = OrderedDict()
        self._lock = RLock()

    def get_history(self, conversation_id: str) -> tuple[ConversationMessage, ...]:
        with self._lock:
            messages = self._items.get(conversation_id, [])
            if conversation_id in self._items:
                self._items.move_to_end(conversation_id)
            return tuple(messages)

    def append_message(
        self,
        conversation_id: str,
        message: ConversationMessage,
    ) -> None:
        normalized_id = str(conversation_id).strip()
        if not normalized_id:
            raise ValueError("conversation_id cannot be empty.")
        with self._lock:
            messages = self._items.setdefault(normalized_id, [])
            messages.append(message)
            del messages[:-self._max_messages]
            self._items.move_to_end(normalized_id)
            while len(self._items) > self._max_conversations:
                self._items.popitem(last=False)

