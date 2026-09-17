"""
Central configuration.

All tunables live here and are driven by environment variables so the same
image can run locally, on Streamlit Cloud, or on AWS without code changes.
Validation happens up-front with actionable messages rather than crashing
deep inside an API call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _env_str(key: str, default: str) -> str:
    value = os.getenv(key)
    return value.strip() if value and value.strip() else default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid."""


@lru_cache(maxsize=1)
def _aws_credentials_available(region: str) -> bool:
    """
    Whether boto3 can resolve credentials from any source in its chain.

    This is a local resolution (env vars, shared config, IAM role) with no
    API call, so it is cheap enough to gate the provider list. Cached because
    Streamlit re-runs the whole script on every interaction, and the instance
    metadata probe in the credential chain is not free.

    It cannot tell us whether the account has Bedrock model access enabled —
    that only surfaces on the first call — but it does catch the common case
    of "boto3 is installed, so the app claimed Bedrock was ready".
    """
    try:
        import boto3

        return boto3.Session(region_name=region).get_credentials() is not None
    except Exception:  # noqa: BLE001 - any failure means "not usable"
        return False


@dataclass(frozen=True)
class SandboxSettings:
    """Limits applied to LLM-generated code execution."""

    timeout_seconds: int = field(default_factory=lambda: _env_int("SANDBOX_TIMEOUT", 30))
    memory_limit_mb: int = field(default_factory=lambda: _env_int("SANDBOX_MEMORY_MB", 2048))
    max_figure_bytes: int = field(
        default_factory=lambda: _env_int("SANDBOX_MAX_FIGURE_BYTES", 8 * 1024 * 1024)
    )
    max_output_chars: int = field(
        default_factory=lambda: _env_int("SANDBOX_MAX_OUTPUT_CHARS", 20_000)
    )
    # Worker startup is slow (pandas + sklearn + plotly imports), so the pool
    # keeps a process warm. This is how long we wait for that first boot.
    startup_timeout_seconds: int = field(
        default_factory=lambda: _env_int("SANDBOX_STARTUP_TIMEOUT", 60)
    )


@dataclass(frozen=True)
class DataSettings:
    """Guardrails on user-uploaded data."""

    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 100))
    max_rows: int = field(default_factory=lambda: _env_int("MAX_ROWS", 1_000_000))
    preview_rows: int = field(default_factory=lambda: _env_int("PREVIEW_ROWS", 5))


@dataclass(frozen=True)
class AgentSettings:
    """Agent loop behaviour."""

    temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.0))
    max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 2048))
    # Number of *extra* attempts after the first failure. 2 keeps worst-case
    # token spend at 3x a normal query.
    max_repair_attempts: int = field(default_factory=lambda: _env_int("MAX_REPAIR_ATTEMPTS", 2))
    max_api_retries: int = field(default_factory=lambda: _env_int("MAX_API_RETRIES", 3))
    retry_base_delay: float = field(default_factory=lambda: _env_float("RETRY_BASE_DELAY", 2.0))
    history_turns: int = field(default_factory=lambda: _env_int("HISTORY_TURNS", 10))


@dataclass(frozen=True)
class Settings:
    provider: str = field(default_factory=lambda: _env_str("LLM_PROVIDER", "groq").lower())

    groq_api_key: str | None = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    # Groq rotates its catalogue and retires model ids without notice — the
    # llama-3.x ids this project originally defaulted to now 404. Check
    # https://console.groq.com/docs/models if a call comes back "model does
    # not exist"; the id is the only thing that needs changing.
    groq_model: str = field(
        default_factory=lambda: _env_str("GROQ_MODEL", "openai/gpt-oss-120b")
    )

    aws_region: str = field(default_factory=lambda: _env_str("AWS_REGION", "us-east-1"))
    bedrock_model: str = field(
        default_factory=lambda: _env_str(
            "BEDROCK_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        )
    )

    log_level: str = field(default_factory=lambda: _env_str("LOG_LEVEL", "INFO").upper())

    sandbox: SandboxSettings = field(default_factory=SandboxSettings)
    data: DataSettings = field(default_factory=DataSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)

    # ── Validation ────────────────────────────────────────────

    def validate_provider(self, provider: str | None = None) -> None:
        """
        Raise ConfigError with actionable guidance if the selected provider
        cannot be used. Called at startup so failures surface immediately
        instead of mid-conversation.
        """
        name = (provider or self.provider).lower()

        if name == "groq":
            if not self.groq_api_key:
                raise ConfigError(
                    "GROQ_API_KEY is not set.\n"
                    "  Local:            add GROQ_API_KEY=... to a .env file "
                    "(see .env.example)\n"
                    "  Streamlit Cloud:  add it under App settings > Secrets\n"
                    "  Get a free key:   https://console.groq.com/keys"
                )
        elif name == "bedrock":
            try:
                import boto3  # noqa: F401
            except ImportError as exc:
                raise ConfigError(
                    "LLM_PROVIDER=bedrock requires boto3. Install it with:\n"
                    "  pip install boto3"
                ) from exc

            if not _aws_credentials_available(self.aws_region):
                raise ConfigError(
                    "No AWS credentials found for Bedrock.\n"
                    "  Local:   run `aws configure`, or set AWS_ACCESS_KEY_ID "
                    "and AWS_SECRET_ACCESS_KEY (see .env.example)\n"
                    "  AWS:     attach an IAM role with bedrock:InvokeModel\n"
                    f"  Also:    enable model access for '{self.bedrock_model}' "
                    f"in the Bedrock console for region {self.aws_region}"
                )
        else:
            raise ConfigError(
                f"Unknown LLM_PROVIDER '{name}'. Supported values: 'groq', 'bedrock'."
            )

    def available_providers(self) -> list[str]:
        """Providers that are currently usable, for UI switching."""
        available = []
        for name in ("groq", "bedrock"):
            try:
                self.validate_provider(name)
                available.append(name)
            except ConfigError:
                continue
        return available

    def demo_mode(self) -> bool:
        """
        True when no live provider is configured.

        The app falls back to replaying recorded answers rather than refusing
        to start, so a deployed link keeps working after a key is rotated or a
        free-tier quota is exhausted.
        """
        return not self.available_providers()

    def model_for(self, provider: str | None = None) -> str:
        name = (provider or self.provider).lower()
        return self.groq_model if name == "groq" else self.bedrock_model


settings = Settings()
