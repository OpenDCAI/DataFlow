import httpx
from typing import List, Dict, Any, Optional

class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: int = 60,
        org_id: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.org_id = org_id
        self._sync_client = httpx.Client(timeout=self.timeout, headers=self._headers())
        self._async_client = httpx.AsyncClient(timeout=self.timeout, headers=self._headers())

    def _headers(self) -> Dict[str, str]:
        h = {"Authorization": f"Bearer {self.api_key}",
             "Content-Type": "application/json"}
        if self.org_id:
            h["OpenAI-Organization"] = self.org_id
        return h

    def _prepare_payload(self, messages: List[Dict[str, str]], **extra: Any) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            **extra,
        }

    def _parse_response(self, data: Dict[str, Any]) -> str:
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"Bad schema from API: {data}") from None

    def chat(self, messages: List[Dict[str, str]], **extra) -> str:
        payload = self._prepare_payload(messages, **extra)
        r = self._sync_client.post(self.base_url, json=payload)
        r.raise_for_status()
        return self._parse_response(r.json())

    async def async_chat(self, messages: List[Dict[str, str]], **extra: Any) -> str:
        payload = self._prepare_payload(messages, **extra)
        r = await self._async_client.post(self.base_url, json=payload)
        r.raise_for_status()
        return self._parse_response(r.json())