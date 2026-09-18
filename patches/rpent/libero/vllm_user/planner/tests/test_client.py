# Copyright 2026 The RPent Authors.

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

from rpent_vllm_user.client import (
    SignedVllmClient,
    SignedVllmConfig,
    VllmChatResponse,
    _build_chat_payload,
    _chat_messages_chars,
    _chat_payload_inline_image_bytes,
    _chat_payload_text_bytes,
    _chat_payload_wire_bytes,
)


class _Response:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return self._payload


class _Opener:
    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        return _Response(self.payloads.pop(0))


def _key_file(path: Path, mode: int = 0o600) -> tuple[Path, object]:
    key = ed25519.Ed25519PrivateKey.generate()
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.OpenSSH,
            serialization.NoEncryption(),
        )
    )
    path.chmod(mode)
    return path, key


def test_config_requires_exactly_one_signer_and_loopback_remote_signer() -> None:
    common = {
        "RPENT_VLLM_BASE_URL": "http://10.8.0.86:8100",
        "RPENT_VLLM_KEY_ID": "key",
    }
    with pytest.raises(ValueError, match="exactly one"):
        SignedVllmConfig.from_env(environ=common)
    with pytest.raises(ValueError, match="loopback"):
        SignedVllmConfig.from_env(
            environ={**common, "RPENT_VLLM_SIGNER_URL": "http://10.8.0.2:9010/sign"}
        )


def test_private_key_permissions_and_symlinks_are_rejected(tmp_path: Path) -> None:
    loose, _ = _key_file(tmp_path / "loose", 0o640)
    config = SignedVllmConfig("http://host:8100", "key", private_key=loose)
    with pytest.raises(RuntimeError, match="group or others"):
        SignedVllmClient(config)
    secure, _ = _key_file(tmp_path / "secure")
    link = tmp_path / "link"
    link.symlink_to(secure)
    with pytest.raises(RuntimeError, match="non-symlink"):
        SignedVllmClient(SignedVllmConfig("http://host:8100", "key", private_key=link))


def test_request_uses_canonical_path_and_verifiable_signature(tmp_path: Path) -> None:
    path, key = _key_file(tmp_path / "id_ed25519")
    opener = _Opener({"user_id": "zc", "key_id": "rpent"})
    config = SignedVllmConfig(
        "http://10.8.0.86:8100/v1", "rpent", private_key=path, expected_user_id="zc"
    )
    client = SignedVllmClient(
        config, opener=opener, clock=lambda: 1234, nonce_factory=lambda: "nonce"
    )

    assert client.validate_identity()["user_id"] == "zc"
    request, timeout = opener.calls[0]
    assert request.full_url == "http://10.8.0.86:8100/v1/auth/whoami"
    assert timeout == 900
    canonical = (
        "GET\n/v1/auth/whoami\n1234\nnonce\n" + hashlib.sha256(b"").hexdigest()
    ).encode()
    key.public_key().verify(
        base64.b64decode(request.headers["X-vllm-signature"]), canonical
    )


def test_identity_check_forwards_an_explicit_timeout(tmp_path: Path) -> None:
    path, _ = _key_file(tmp_path / "id_ed25519")
    opener = _Opener({"user_id": "zc", "key_id": "rpent"})
    client = SignedVllmClient(
        SignedVllmConfig(
            "http://service:8100",
            "rpent",
            private_key=path,
            expected_user_id="zc",
        ),
        opener=opener,
    )

    client.validate_identity(timeout_s=3.5)

    assert opener.calls[0][1] == 3.5


def test_chat_serializes_raw_mode_and_model_override(tmp_path: Path) -> None:
    path, _ = _key_file(tmp_path / "key")
    opener = _Opener({"choices": [], "usage": {"prompt_tokens": 2}})
    client = SignedVllmClient(
        SignedVllmConfig("http://host:8100", "key", private_key=path, model="qwen"),
        opener=opener,
    )

    response = client.chat(messages=[{"role": "user", "content": "hi"}])

    payload = json.loads(opener.calls[0][0].data)
    assert payload == {
        "mode": "raw",
        "messages": [{"role": "user", "content": "hi"}],
        "model": "qwen",
    }
    assert response.usage == {"prompt_tokens": 2}


def test_chat_payload_budgets_only_message_images_as_visual_bytes() -> None:
    message_image = "data:image/png;base64," + "a" * 50_000
    schema_image_shaped_text = "data:not-an-image-schema-field," + "b" * 20_000
    payload = _build_chat_payload(
        messages=[
            {"role": "system", "content": "rules"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "inspect"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": message_image,
                            "detail": "high",
                        },
                        "vendor_metadata": "must be counted",
                    },
                ],
            },
        ],
        model="default",
        max_tokens=100,
        extra_body={
            "tools": [
                {
                    "type": "image_url",
                    "image_url": {"url": schema_image_shaped_text},
                }
            ]
        },
    )

    measured = _chat_payload_text_bytes(payload)
    actual = len(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )

    assert 20_000 < measured < 25_000
    assert actual > measured + 49_000
    assert measured > len("rulesinspectobserve")
    assert _chat_payload_inline_image_bytes(payload) == len(message_image)
    assert _chat_payload_wire_bytes(payload) == actual
    assert _chat_messages_chars(payload["messages"]) > 50_000


def test_chat_rejects_non_json_request_values_before_transport(tmp_path: Path) -> None:
    path, _ = _key_file(tmp_path / "key")
    opener = _Opener({"choices": []})
    client = SignedVllmClient(
        SignedVllmConfig("http://host:8100", "key", private_key=path),
        opener=opener,
    )

    with pytest.raises(TypeError, match="not JSON serializable"):
        client.chat(messages=[], extra_body={"invalid": object()})

    assert opener.calls == []


def test_chat_uses_timeout_override_without_sending_it_to_model(tmp_path: Path) -> None:
    path, _ = _key_file(tmp_path / "key")
    opener = _Opener({"choices": []})
    client = SignedVllmClient(
        SignedVllmConfig("http://host:8100", "key", private_key=path),
        opener=opener,
    )

    client.chat(messages=[], timeout_s=12.5)

    assert opener.calls[0][1] <= 12.5
    assert "timeout_s" not in json.loads(opener.calls[0][0].data)


def test_response_normalizes_reasoning_field() -> None:
    response = VllmChatResponse(
        {"choices": [{"message": {"reasoning": "inspect, then act"}}]}
    )

    assert response.reasoning_content == "inspect, then act"


def test_rsa_key_uses_pss_sha256_signature() -> None:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    opener = _Opener({"user_id": "zc", "key_id": "rsa-key"})
    client = SignedVllmClient(
        SignedVllmConfig(
            "http://host:8100", "rsa-key", signer_url="http://127.0.0.1/sign"
        ),
        opener=opener,
        clock=lambda: 5,
        nonce_factory=lambda: "rsa-nonce",
        signing_key=key,
    )

    client.whoami()

    signature = base64.b64decode(opener.calls[0][0].headers["X-vllm-signature"])
    canonical = (
        "GET\n/v1/auth/whoami\n5\nrsa-nonce\n" + hashlib.sha256(b"").hexdigest()
    ).encode()
    key.public_key().verify(
        signature,
        canonical,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256(),
    )
