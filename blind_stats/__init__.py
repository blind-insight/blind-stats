"""blind_stats — descriptive and inferential statistics on encrypted data.

Every statistic is computed from Blind Insight `aggregate` and `count` responses.
No code path in this package requests decrypted records.
"""

from .client import BlindInsightClient, ProfilingStats, QueryTiming, profiling
from .config import (
    DEFAULT_PROXY_URL,
    get_demo_config,
    load_env,
    require_env,
    resolve_target,
)
from .stats import (
    BILinearRegression,
    BIStatsSession,
    CrosstabResult,
    DescribeResult,
    FeatureScreeningResult,
    FrequencyResult,
    GroupedStatsResult,
    HistogramResult,
    MatrixStatsResult,
    RegressionSummaryResult,
    StatResult,
)

__all__ = [
    # session + regression
    "BIStatsSession",
    "BILinearRegression",
    # result types
    "StatResult",
    "HistogramResult",
    "FrequencyResult",
    "DescribeResult",
    "GroupedStatsResult",
    "CrosstabResult",
    "MatrixStatsResult",
    "RegressionSummaryResult",
    "FeatureScreeningResult",
    # transport
    "BlindInsightClient",
    "ProfilingStats",
    "QueryTiming",
    "profiling",
    # config
    "load_env",
    "require_env",
    "get_demo_config",
    "resolve_target",
    "DEFAULT_PROXY_URL",
]
