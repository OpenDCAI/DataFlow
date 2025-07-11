#!/usr/bin/env python3
"""
memory_service.py  ── Simple in-memory session & conversation store
Author  : Zhou Liu
License : MIT
Created : 2024-06-24

This module provides the `Memory` class, a lightweight container that keeps:

* full chat history (list of role/content dicts)
* caches of the latest user / assistant messages
* arbitrary per-session key-value data
* helper functions for pickling complex objects

It is completely thread-safe in CPython **only if** each request handler
works on its own event-loop thread; otherwise add locking by yourself.
"""

import hashlib
import pickle
import httpx
import requests
from typing import Any, Dict, List

class Memory:
    """
    In-memory container that keeps both conversation history and arbitrary
    per-session data.  
    A *session* is identified by a SHA-256 hash of a user-supplied key
    (`sessionKEY`); every API caller can therefore keep an independent context
    by re-using the same key.
    """

    def __init__(self) -> None:
        # Conversation history:
        # { session_id: [ {"role": str, "content": str}, ... ] }
        self.storage: Dict[str, List[Dict[str, Any]]] = {}

        # Cache only the **latest** user/assistant messages to speed up lookup
        self._last_user: Dict[str, Dict[str, Any]] = {}
        self._last_assistant: Dict[str, Dict[str, Any]] = {}

        # Arbitrary business-level data:
        # { session_id: { key: value, ... } }
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def get_session_id(self, key: str) -> str:
        """
        Convert an arbitrary `sessionKEY` to a stable SHA-256 hex digest.

        Parameters
        ----------
        key : str
            Raw session key provided by the client.

        Returns
        -------
        str
            64-char hexadecimal digest usable as dictionary key.
        """
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def add_messages(
        self, session_id: str, messages: List[Dict[str, Any]]
    ) -> None:
        """
        Append a batch of messages to the conversation history.

        Each element in *messages* must be a dict with at least
        ``{"role": ..., "content": ...}``.

        The method is usually used when the frontend sends a whole list of
        previous turns.
        """
        buf = self.storage.setdefault(session_id, [])
        for m in messages:
            buf.append({"role": m["role"], "content": m["content"]})

    def add_response(self, session_id: str, message: Dict[str, Any]) -> None:
        """
        Append one assistant message to the history (syntactic sugar)."""
        buf = self.storage.setdefault(session_id, [])
        buf.append({"role": message["role"], "content": message["content"]})

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Return the full conversation list for the session.  Empty list if none.
        """
        return self.storage.get(session_id, [])

    def add_content(self, session_id: str, role: str, message: Any) -> None:
        """
        Append **any** object as plain string to the history.
        Handy for quick logging/debugging.
        """
        buf = self.storage.setdefault(session_id, [])
        buf.append({"role": role, "content": str(message)})

    def get_last_messages(
        self, session_id: str, n: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Convenience helper: return the last *n* messages (default 2)
        from the conversation history.  If the history length is shorter than
        *n*, the entire history is returned.
        """
        history = self.get_history(session_id)
        return history[-n:] if len(history) >= n else history

    def add_last_user(self, session_id: str, message: Dict[str, Any]) -> None:
        """Cache the latest user message for quick lookup."""
        self._last_user[session_id] = message

    def add_last_assistant(
        self, session_id: str, message: Dict[str, Any]
    ) -> None:
        """Cache the latest assistant message for quick lookup."""
        self._last_assistant[session_id] = message

    def get_last_user(self, session_id: str) -> Dict[str, Any]:
        """Return the cached latest user message; empty dict if missing."""
        return self._last_user.get(session_id, {})

    def get_last_assistant(self, session_id: str) -> Dict[str, Any]:
        """Return the cached latest assistant message; empty dict if missing."""
        return self._last_assistant.get(session_id, {})

    def set_session_data(self, session_id: str, key: str, value: Any) -> None:
        """Store any Python object (not necessarily JSON-serialisable)."""
        self.sessions.setdefault(session_id, {})[key] = value

    def get_session_data(
        self, session_id: str, key: str, default: Any = None
    ) -> Any:
        """Fetch a value from the per-session store; *default* if absent."""
        return self.sessions.get(session_id, {}).get(key, default)

    def save_object(self, session_id: str, key: str, obj: Any) -> None:
        """Pickle *obj* and save it under (*session_id*, *key*)."""
        data = pickle.dumps(obj)
        self.set_session_data(session_id, key, data)

    def load_object(
        self, session_id: str, key: str, default: Any = None
    ) -> Any:
        """
        Unpickle and return an object previously stored with `save_object`.
        If the key is absent or unpickling fails, *default* is returned.
        """
        data = self.get_session_data(session_id, key)
        if data is None:
            return default
        try:
            return pickle.loads(data)
        except Exception:
            return default

    def append_session_list(
        self, session_id: str, key: str, item: Any
    ) -> None:
        """
        Treat the value at (*session_id*, *key*) as a list; append *item* to it.
        If the key does not exist, it is created as a new list first.
        """
        buf = self.get_session_data(session_id, key) or []
        buf.append(item)
        self.set_session_data(session_id, key, buf)

    def clear_history(self, session_id: str) -> None:
        """
        Remove **all conversation data** for the session:

        - `self.storage`         full dialogue list
        - `self._last_user`      cached last user turn
        - `self._last_assistant` cached last assistant turn
        """
        self.storage.pop(session_id, None)
        self._last_user.pop(session_id, None)
        self._last_assistant.pop(session_id, None)

    def clear_session(self, session_id: str) -> None:
        """
        Remove **all auxiliary business data** for the session
        (`self.sessions` subtree).
        """
        self.sessions.pop(session_id, None)

    def reset(self, session_id: str) -> None:
        """
        Completely wipe a session: conversation history **and** business data.
        """
        self.clear_history(session_id)
        self.clear_session(session_id)



class MemoryClient:
    def __init__(self, memory):
        self.memory = memory

    async def post(self,
                   url, headers, json_data,
                   session_key: str) -> str:
        session_id = self.memory.get_session_id(session_key)
        # Record the sent messages
        if "messages" in json_data:
            self.memory.add_messages(session_id, json_data["messages"])
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                headers=headers,
                json=json_data,
                timeout=360.0
            )
            resp.raise_for_status()
            result = resp.json()
        # Record the assistant message (only the first choice)
        choice = result.get("choices", [{}])[0].get("message")
        if choice:
            self.memory.add_response(session_id, choice)
        # Return the message content
        return choice["content"] if choice and "content" in choice else ""