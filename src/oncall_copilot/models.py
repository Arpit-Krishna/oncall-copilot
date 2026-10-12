from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, field_validator


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
