import sys
from pathlib import Path

# Delegate to code/main.py so the agent can be run from repo root
sys.path.insert(0, str(Path(__file__).parent / "code"))

from main import main  # noqa: E402

if __name__ == "__main__":
    main()
