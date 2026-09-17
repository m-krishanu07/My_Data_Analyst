"""
AWS Bedrock provider using the Converse API.

Converse is the model-agnostic entry point on Bedrock: the same request
shape works for Anthropic, Meta and Amazon models, so switching model ids
needs no code change. It also handles system prompts and multi-turn
natively, unlike InvokeModel which requires per-vendor request bodies.

Auth uses the standard boto3 credential chain (env vars, shared config,
instance/task role), so nothing AWS-specific leaks into application code.
"""

from __future__ import annotations

import time

from src.config import ConfigError, settings
from src.llm.base import LLMError, LLMProvider, LLMResponse, Message, normalize_messages
from src.llm.pricing import estimate_cost
from src.logging_conf import get_logger

logger = get_logger(__name__)


class BedrockProvider(LLMProvider):
    name = "bedrock"

    def __init__(self, model: str | None = None, region: str | None = None):
        super().__init__(model or settings.bedrock_model)
        self.region = region or settings.aws_region
        try:
            import boto3
        except ImportError as exc:
            raise ConfigError(
                "LLM_PROVIDER=bedrock requires boto3. Run: pip install boto3"
            ) from exc

        self._client = boto3.client("bedrock-runtime", region_name=self.region)

    def complete(
        self,
        system: str,
        messages: list[Message | dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        # Converse requires content as a list of typed blocks, and strictly
        # alternating roles starting with 'user' — normalize_messages
        # guarantees both.
        converse_messages = [
            {"role": m.role, "content": [{"text": m.content}]}
            for m in normalize_messages(messages)
        ]
        if not converse_messages:
            raise LLMError("Cannot call Bedrock with an empty message list.")

        started = time.perf_counter()
        try:
            response = self._client.converse(
                modelId=self.model,
                system=[{"text": system}],
                messages=converse_messages,
                inferenceConfig={"temperature": temperature, "maxTokens": max_tokens},
            )
        except Exception as exc:
            raise LLMError(self._explain(exc)) from exc
        latency = time.perf_counter() - started

        blocks = response.get("output", {}).get("message", {}).get("content", [])
        text = "".join(block.get("text", "") for block in blocks)

        usage = response.get("usage", {})
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)

        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=latency,
            cost_usd=estimate_cost(self.model, input_tokens, output_tokens),
            raw={"stop_reason": response.get("stopReason")},
        )

    def _explain(self, exc: Exception) -> str:
        """Turn opaque boto3 errors into something the user can act on."""
        text = str(exc)
        if "AccessDenied" in text or "not authorized" in text:
            return (
                f"AWS denied access to '{self.model}' in {self.region}. Enable model "
                "access in the Bedrock console (Model access) and confirm your IAM "
                "principal allows bedrock:InvokeModel."
            )
        if "ValidationException" in text and "model identifier" in text.lower():
            return (
                f"Bedrock does not recognise model id '{self.model}' in {self.region}. "
                "Check the id and that the model is offered in this region."
            )
        if "ExpiredToken" in text or "InvalidSignature" in text:
            return "AWS credentials are invalid or expired. Refresh them and retry."
        if "could not be found" in text.lower() and "credential" in text.lower():
            return (
                "No AWS credentials found. Configure them via environment variables, "
                "`aws configure`, or an instance/task role."
            )
        return text
