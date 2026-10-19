from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Level(StrEnum):
    DEBUG = "debug"
    INFO = "info"
    NOTICE = "notice"
    WARN = "warn"
    ERROR = "error"
    CRIT = "crit"


class LogRecord(BaseModel):
    line_no: int
    timestamp: datetime
    level: Level
    message: str
    source: str = "apache"

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: object) -> object:
        if isinstance(v, str):
            return datetime.strptime(v, "%a %b %d %H:%M:%S %Y")
        return v

Severity = Literal["critical", "high", "medium", "low", "info"]
EventType = Literal["worker_error", "worker_init", "child_found", "client_denied", "other"]


class LogEvent(BaseModel):
    severity: Severity
    component: str
    event_type: EventType
    summary: str = Field(max_length=200)
    actionable: bool


@dataclass
class LoadResult:
    records: list[LogRecord] = field(default_factory=list)
    count: int = 0
    skipped: int = 0