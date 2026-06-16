"""Pytest session config.

Make the test suite hermetic: do NOT load the developer's `.env` (which may select real
backends like Presidio/Ollama that aren't installed in every environment). Tests pin the
backends they need explicitly; everything else uses the offline defaults.
"""

import os

os.environ.setdefault("RMSAI_NO_DOTENV", "1")
