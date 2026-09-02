"""Environment + target resolution for blind-stats.

Holds no domain knowledge beyond the demo fixture's field domains: everything
here is about *where* to point a `BIStatsSession`, never about what the data means.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_PROXY_URL = "https://local.blindinsight.io"

REQUIRED_ENV = ("BI_EMAIL", "BI_PASSWORD", "BI_ORG")


def load_env(path: str = ".env") -> None:
    """Load a `.env` file into os.environ (no external deps).

    Existing environment variables win — this uses `setdefault`. Silently
    no-ops when the file is absent, so a wrong path does not raise; call
    `require_env()` afterwards if you need the variables to actually be set.
    """
    env_file = Path(path)
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def require_env(names: tuple[str, ...] = REQUIRED_ENV) -> None:
    """Raise if any required credential is missing from the environment."""
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing BI env vars: {', '.join(missing)}. Copy .env.example to .env and fill it in.")


def get_demo_config() -> dict[str, Any]:
    """Dataset/schema slugs and field domains for the bundled 290-record fixture.

    `field_domains` is what `BIStatsSession` needs to turn quantile/histogram
    requests into bounded binary searches over count queries: an (lo, hi) tuple
    for numeric fields, an explicit value list for categoricals.
    """
    return {
        "dataset": "stats-demo",
        "schema": "stats-fraud-290",
        "fixture": "fixtures/fraud_train_290.json",
        "schema_json": "schemas/fraud.json",
        "field_domains": {
            "risk_level": (0, 102),
        },
        "numeric_fields": ["risk_level", "year", "month", "day"],
        "categorical_fields": [
            "fraud_type",
            "is_active",
            "account_jurisdiction",
            "reporting_bank_id",
            "reporting_jurisdiction",
        ],
    }


def resolve_target(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve org/dataset/schema/proxy from the environment, falling back to the demo config.

    Resolution order for dataset and schema is `BI_STATS_DATASET` -> `BI_DATASET`
    -> the demo config, so the `BI_STATS_*` variables are optional overrides that
    win when set. Point these at your own schema to run the stats core on real data.
    """
    config = config or get_demo_config()
    require_env()
    return {
        "org": os.environ["BI_ORG"],
        "dataset": os.environ.get("BI_STATS_DATASET", os.environ.get("BI_DATASET", config["dataset"])),
        "schema": os.environ.get("BI_STATS_SCHEMA", os.environ.get("BI_SCHEMA", config["schema"])),
        "proxy_url": os.environ.get("BI_PROXY_URL", DEFAULT_PROXY_URL),
        "verify_ssl": os.environ.get("BI_VERIFY_SSL", "false").lower() == "true",
    }
