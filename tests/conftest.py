from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def apache_log() -> Path:
    """The real Loghub Apache sample: 2,000 lines, 1,405 notice, 595 error."""
    return DATA_DIR / "apache_2k.log"
