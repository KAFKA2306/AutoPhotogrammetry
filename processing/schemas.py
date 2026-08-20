from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MeasurementsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preflight: dict[str, Any] | None
    colmap: dict[str, Any] | None
    splat: dict[str, Any] | None


class VideoSourceModel(BaseModel):
    """Structural validation only; semantic stage policy stays in video_sources.py."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    evaluation_stage: str = Field(min_length=1)
    measurements: MeasurementsModel


class EvaluationPolicyModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    stages: dict[str, dict[str, Any]]


class VideoRegistryModel(BaseModel):
    """Pydantic boundary for the untrusted JSON registry file."""

    model_config = ConfigDict(extra="allow")

    schema_version: int
    default: str = Field(min_length=1)
    evaluation_policy: EvaluationPolicyModel
    videos: list[VideoSourceModel] = Field(min_length=1)
