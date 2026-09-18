# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Authenticated client for the RPent-compatible vLLM chat service."""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import os
import secrets
import stat
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import OpenerDirector, ProxyHandler, Request, build_opener

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

_MAX_ERROR_DETAIL = 500


@dataclass(frozen=True, slots=True)
class SignedVllmConfig:
    """Validated connection and signing configuration."""

    base_url: str
    key_id: str
    private_key: Path | None = None
    signer_url: str | None = None
    signer_token: str | None = None
    expected_user_id: str | None = None
    model: str = "default"
    timeout_s: float = 900.0

    def __post_init__(self) -> None:
        """Reject ambiguous or unsafe manually constructed configurations."""
        _validate_http_url(self.base_url, name="base_url")
        if not self.key_id:
            raise ValueError("key_id must be non-empty")
        if (self.private_key is None) == (self.signer_url is None):
            raise ValueError("configure exactly one local key or loopback signer")
        if self.signer_url is not None:
            _validate_signer_url(self.signer_url)
        if not self.model:
            raise ValueError("model must be non-empty")
        _positive_timeout(self.timeout_s)

    @classmethod
    def from_env(
        cls,
        *,
        base_url_override: str | None = None,
        model_override: str | None = None,
        timeout_s: float = 900.0,
        environ: Mapping[str, str] | None = None,
    ) -> SignedVllmConfig:
        """Load configuration without retaining a global environment snapshot."""
        env = os.environ if environ is None else environ
        base_url = base_url_override or env.get("RPENT_VLLM_BASE_URL", "")
        key_id = env.get("RPENT_VLLM_KEY_ID", "")
        private_key_text = env.get("RPENT_VLLM_PRIVATE_KEY", "")
        signer_url = env.get("RPENT_VLLM_SIGNER_URL", "") or None
        if not key_id:
            raise ValueError("RPENT_VLLM_KEY_ID must be set")
        if bool(private_key_text) == bool(signer_url):
            raise ValueError(
                "set exactly one of RPENT_VLLM_PRIVATE_KEY and RPENT_VLLM_SIGNER_URL"
            )
        validated_base = _validate_http_url(base_url, name="RPENT_VLLM_BASE_URL")
        if signer_url is not None:
            signer_url = _validate_signer_url(signer_url)
        return cls(
            base_url=validated_base,
            key_id=key_id,
            private_key=Path(private_key_text).expanduser()
            if private_key_text
            else None,
            signer_url=signer_url,
            signer_token=env.get("RPENT_VLLM_SIGNER_TOKEN") or None,
            expected_user_id=env.get("RPENT_VLLM_EXPECTED_USER_ID") or None,
            model=model_override or env.get("RPENT_VLLM_MODEL", "default"),
            timeout_s=_positive_timeout(timeout_s),
        )


class VllmChatResponse:
    """Small response facade consumed by the planner."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.raw = dict(payload)
        usage = payload.get("usage")
        self.usage = dict(usage) if isinstance(usage, Mapping) else {}
        self.reasoning_content: str | None = None
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            message = first.get("message") if isinstance(first, Mapping) else None
            if isinstance(message, Mapping):
                reasoning = message.get("reasoning_content")
                if not isinstance(reasoning, str):
                    reasoning = message.get("reasoning")
                if isinstance(reasoning, str):
                    self.reasoning_content = reasoning


class SignedVllmClient:
    """Issue signed requests without consulting process proxy variables."""

    def __init__(
        self,
        config: SignedVllmConfig,
        *,
        opener: OpenerDirector | Any | None = None,
        clock: Callable[[], float] = time.time,
        nonce_factory: Callable[[], str] = lambda: secrets.token_urlsafe(18),
        signing_key: object | None = None,
    ) -> None:
        self.config = config
        self._opener = opener or build_opener(ProxyHandler({}))
        self._clock = clock
        self._nonce_factory = nonce_factory
        self._signing_key = signing_key
        if signing_key is None and config.private_key is not None:
            self._signing_key = _load_private_key(config.private_key)

    def request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Send one signed JSON request and require a JSON-object response."""
        if not path.startswith("/v1/"):
            raise ValueError("signed API path must start with /v1/")
        body = b"" if payload is None else _encode_json_body(payload)
        timestamp = str(int(self._clock()))
        nonce = self._nonce_factory()
        canonical = (
            f"{method.upper()}\n{path}\n{timestamp}\n{nonce}\n"
            f"{hashlib.sha256(body).hexdigest()}"
        ).encode()
        request_timeout = (
            self.config.timeout_s if timeout_s is None else _positive_timeout(timeout_s)
        )
        request_deadline = time.monotonic() + request_timeout
        signature = self._sign(canonical, timeout_s=request_timeout)
        if self.config.signer_url is not None:
            request_timeout = request_deadline - time.monotonic()
            if request_timeout <= 0:
                raise TimeoutError("vLLM request timed out while signing")
        headers = {
            "Accept": "application/json",
            "X-VLLM-Key-Id": self.config.key_id,
            "X-VLLM-Timestamp": timestamp,
            "X-VLLM-Nonce": nonce,
            "X-VLLM-Signature": base64.b64encode(signature).decode("ascii"),
        }
        if payload is not None:
            headers["Content-Type"] = "application/json"
        request = Request(
            _join_api_url(self.config.base_url, path),
            data=body if payload is not None else None,
            headers=headers,
            method=method.upper(),
        )
        try:
            with self._opener.open(request, timeout=request_timeout) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read(_MAX_ERROR_DETAIL).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"vLLM request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, OSError) as exc:
            raise RuntimeError(
                f"vLLM request transport failed: {_safe_detail(exc)}"
            ) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("vLLM response was not valid UTF-8 JSON") from exc
        if not isinstance(decoded, Mapping):
            raise RuntimeError("vLLM response must be a JSON object")
        return dict(decoded)

    def whoami(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        """Return the identity resolved by the deployment."""
        return self.request("GET", "/v1/auth/whoami", timeout_s=timeout_s)

    def validate_identity(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        """Fail closed when the service resolves an unexpected identity."""
        identity = self.whoami(timeout_s=timeout_s)
        if identity.get("key_id") != self.config.key_id:
            raise RuntimeError("vLLM service resolved an unexpected key_id")
        if (
            self.config.expected_user_id is not None
            and identity.get("user_id") != self.config.expected_user_id
        ):
            raise RuntimeError("vLLM service resolved an unexpected user_id")
        return identity

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str | None = None,
        extra_body: Mapping[str, Any] | None = None,
        timeout_s: float | None = None,
        **params: Any,
    ) -> VllmChatResponse:
        """Call the deployment's raw OpenAI-compatible chat mode."""
        resolved_model = model or self.config.model
        payload = _build_chat_payload(
            messages=messages,
            model=resolved_model,
            extra_body=extra_body,
            **params,
        )
        return VllmChatResponse(
            self.request("POST", "/v1/chat/completions", payload, timeout_s=timeout_s)
        )

    def _sign(self, canonical: bytes, *, timeout_s: float) -> bytes:
        if self._signing_key is not None:
            return _sign_locally(self._signing_key, canonical)
        if self.config.signer_url is None:
            raise RuntimeError("no vLLM signing method is configured")
        payload = json.dumps(
            {"canonical_b64": base64.b64encode(canonical).decode("ascii")},
            separators=(",", ":"),
        ).encode()
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.config.signer_token:
            headers["X-RPent-Signer-Token"] = self.config.signer_token
        request = Request(
            self.config.signer_url, data=payload, headers=headers, method="POST"
        )
        try:
            with self._opener.open(request, timeout=timeout_s) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except (
            HTTPError,
            URLError,
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise RuntimeError(f"vLLM signer failed: {_safe_detail(exc)}") from exc
        value = decoded.get("signature_b64") if isinstance(decoded, Mapping) else None
        if not isinstance(value, str) or not value:
            raise RuntimeError("vLLM signer returned no signature_b64")
        try:
            return base64.b64decode(value, validate=True)
        except ValueError as exc:
            raise RuntimeError("vLLM signer returned an invalid signature") from exc


def _validate_http_url(value: str, *, name: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{name} must be an absolute HTTP(S) URL")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{name} must not include a query or fragment")
    return value.rstrip("/")


def _validate_signer_url(value: str) -> str:
    validated = _validate_http_url(value, name="RPENT_VLLM_SIGNER_URL")
    hostname = urlsplit(validated).hostname
    try:
        is_loopback = (
            hostname == "localhost" or ipaddress.ip_address(hostname or "").is_loopback
        )
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise ValueError("RPENT_VLLM_SIGNER_URL must use a loopback host")
    return validated


def _join_api_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1") and path.startswith("/v1/"):
        return base + path[3:]
    return base + path


def _load_private_key(path: Path) -> object:
    try:
        info = path.lstat()
    except OSError as exc:
        raise RuntimeError("vLLM private key is not a readable file") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RuntimeError("vLLM private key must be a regular, non-symlink file")
    if info.st_mode & 0o077:
        raise RuntimeError("vLLM private key must not be accessible by group or others")
    try:
        return serialization.load_ssh_private_key(path.read_bytes(), password=None)
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError("vLLM private key could not be loaded") from exc


def _sign_locally(key: object, canonical: bytes) -> bytes:
    if isinstance(key, ed25519.Ed25519PrivateKey):
        return key.sign(canonical)
    if isinstance(key, rsa.RSAPrivateKey):
        return key.sign(
            canonical,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
    raise TypeError("vLLM signing key must be an Ed25519 or RSA private key")


def _positive_timeout(value: float) -> float:
    timeout = float(value)
    if timeout <= 0:
        raise ValueError("timeout_s must be positive")
    return timeout


def _safe_detail(exc: BaseException) -> str:
    detail = str(exc).replace("\n", " ")[:_MAX_ERROR_DETAIL]
    return detail or type(exc).__name__


def _build_chat_payload(
    *,
    messages: list[dict[str, Any]],
    model: str,
    extra_body: Mapping[str, Any] | None = None,
    **params: Any,
) -> dict[str, Any]:
    """Build the exact JSON payload used by :meth:`SignedVllmClient.chat`.

    Keeping construction pure lets the planner budget the same request that the
    signed client will serialize instead of estimating only the message list.
    """
    if not isinstance(model, str) or not model:
        raise ValueError("model must be a non-empty string")
    request_params = dict(params)
    if model != "default":
        request_params["model"] = model
    if extra_body is not None:
        if not isinstance(extra_body, Mapping):
            raise TypeError("extra_body must be a mapping")
        request_params.update(extra_body)
    return {"mode": "raw", "messages": list(messages), **request_params}


def _chat_payload_text_bytes(payload: Mapping[str, Any]) -> int:
    """Return deterministic UTF-8 JSON bytes excluding inline image payloads.

    Base64 camera data consumes transport bytes and vision tokens rather than
    ordinary text tokens.  Replace only image parts in chat-message content;
    image-shaped data in tool schemas or other request fields remains ordinary
    counted text.  The image and exact wire-body budgets are enforced
    separately by the planner.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    projected = dict(payload)
    messages = payload.get("messages")
    if isinstance(messages, list):
        projected["messages"] = _project_message_images(messages)
    return len(_encode_json_body(projected))


def _chat_payload_inline_image_bytes(payload: Mapping[str, Any]) -> int:
    """Return encoded data-URL bytes in chat-message image content only."""
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return 0
    total = 0
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            url = _inline_image_url(item)
            if url is not None:
                total += len(url.encode("utf-8"))
    return total


def _chat_payload_wire_bytes(payload: Mapping[str, Any]) -> int:
    """Return the exact JSON-body bytes sent and signed by the client."""
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    return len(_encode_json_body(payload))


def _chat_messages_chars(messages: list[dict[str, Any]]) -> int:
    """Return the legacy serialized-message character count, including images."""
    return len(_encode_json_body(messages).decode("utf-8"))


def _encode_json_body(value: Any) -> bytes:
    """Serialize JSON exactly as :meth:`SignedVllmClient.request` does."""
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _project_message_images(messages: list[object]) -> list[object]:
    projected: list[object] = []
    for message in messages:
        if not isinstance(message, Mapping):
            projected.append(message)
            continue
        copied = dict(message)
        content = message.get("content")
        if isinstance(content, list):
            copied["content"] = [_project_image_part(item) for item in content]
        projected.append(copied)
    return projected


def _project_image_part(value: object) -> object:
    url = _inline_image_url(value)
    if url is None:
        return value
    assert isinstance(value, Mapping)
    image_url = value["image_url"]
    assert isinstance(image_url, Mapping)
    projected = dict(value)
    projected_image_url = dict(image_url)
    projected_image_url["url"] = (
        "[inline image counted by separate image budget; "
        f"encoded_bytes={len(url.encode('utf-8'))}]"
    )
    projected["image_url"] = projected_image_url
    return projected


def _inline_image_url(value: object) -> str | None:
    if not isinstance(value, Mapping) or value.get("type") != "image_url":
        return None
    image_url = value.get("image_url")
    url = image_url.get("url") if isinstance(image_url, Mapping) else None
    if isinstance(url, str) and url.startswith("data:"):
        return url
    return None


__all__ = [
    "SignedVllmClient",
    "SignedVllmConfig",
    "VllmChatResponse",
]
