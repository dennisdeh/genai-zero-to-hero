"""
Configuration of the best-practices example, read from environment variables.

The settings live in one immutable dataclass that is built once at start-up (`Settings.from_env()`)
and then passed explicitly to the functions that need it. Nothing else in the code reads
`os.environ`, so the configuration is easy to find, to validate and to replace in tests
(with pytest's `monkeypatch` fixture).

Environment variables (all optional, prefixed with GZH_ to avoid clashes):
- GZH_SEED:           random seed (default: 42)
- GZH_N_SAMPLES:      number of synthetic samples (default: 2000)
- GZH_TEST_FRACTION:  fraction of samples used for testing, in (0, 1) (default: 0.25)
- GZH_LOG_LEVEL:      python logging level, e.g. DEBUG, INFO, WARNING (default: INFO)
- GZH_SAVE_OUTPUTS:   write artefacts to the outputs folder, true/false (default: true)
- GZH_OUTPUT_DIR:     folder for artefacts (default: outputs/ next to this section's code)

Secrets (API keys, passwords) would be read the same way, but never get a default value and are
never logged or committed - see the book section for the .env pattern.
"""

import logging
import os
from dataclasses import asdict, dataclass

# default output folder: outputs/ next to the section's scripts (gitignored)
SECTION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(SECTION_DIR, "outputs")

ENV_PREFIX = "GZH_"
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def parse_bool(value: str) -> bool:
    """Parse a boolean from a string such as "true", "0", "Yes" or "off".

    Raises:
        ValueError: if the string is not a recognised boolean.
    """
    normalised = value.strip().lower()
    if normalised in _TRUE_VALUES:
        return True
    if normalised in _FALSE_VALUES:
        return False
    raise ValueError(f"Cannot interpret {value!r} as a boolean")


@dataclass(frozen=True)
class Settings:
    """Immutable, validated settings of the example."""

    seed: int = 42
    n_samples: int = 2000
    test_fraction: float = 0.25
    log_level: str = "INFO"
    save_outputs: bool = True
    output_dir: str = DEFAULT_OUTPUT_DIR

    def __post_init__(self) -> None:
        """Validate the values early, so that mistakes fail at start-up and not after an hour of training."""
        if self.n_samples < 10:
            raise ValueError(f"n_samples must be at least 10, got {self.n_samples}")
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError(f"test_fraction must be in (0, 1), got {self.test_fraction}")
        if self.log_level not in logging.getLevelNamesMapping():
            raise ValueError(f"Unknown log level {self.log_level!r}")

    @classmethod
    def from_env(cls) -> "Settings":
        """Build the settings from GZH_* environment variables, falling back to the defaults."""

        def env(name: str, default: str) -> str:
            return os.getenv(ENV_PREFIX + name, default)

        return cls(
            seed=int(env("SEED", str(cls.seed))),
            n_samples=int(env("N_SAMPLES", str(cls.n_samples))),
            test_fraction=float(env("TEST_FRACTION", str(cls.test_fraction))),
            log_level=env("LOG_LEVEL", cls.log_level).upper(),
            save_outputs=parse_bool(env("SAVE_OUTPUTS", str(cls.save_outputs))),
            output_dir=env("OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the settings as a plain dictionary (e.g. to store next to the results of a run)."""
        return asdict(self)
