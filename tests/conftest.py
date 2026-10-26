import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def cp():                      # real documents + real Olist DB
    from app.service import Copilot
    return Copilot()


@pytest.fixture(scope="session")
def fx():                      # synthetic docs with ACL + prompt-injection traps
    from app.service import Copilot
    return Copilot(docs_dir=ROOT / "tests" / "fixtures" / "docs")
