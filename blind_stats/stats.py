"""BI-only descriptive statistics for Blind Insight aggregate queries.

The statistics core intentionally works from aggregate/count responses only.
It must not request decrypted records or depend on plaintext row access.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

FilterInput = str | Iterable[str] | None
Domain = tuple[int | float, int | float]


@dataclass
class StatResult:
    """Result for a scalar statistic computed from BI aggregate queries."""

    statistic: Any
    pvalue: float | None = None
    estimate: Any | None = None
    confidence_interval: tuple[float, float] | None = None
    n: int | dict | None = None
    method: str = ""
    exact: bool = True
    queries: int = 0
    warnings: list[str] = field(default_factory=list)

    def _repr_html_(self) -> str:
        return f"StatResult(statistic={self.statistic}, estimate={self.estimate}, n={self.n}, method={self.method}, exact={self.exact}, queries={self.queries}, warnings={self.warnings})"


@dataclass
class HistogramResult:
    """Bucket counts for a numeric field."""

    field: str
    counts: dict[str, int]
    bins: list[Any]
    n: int
    method: str = ""
    exact: bool = True
    queries: int = 0
    warnings: list[str] = field(default_factory=list)

    def _repr_html_(self) -> str:
        return f"<pre>{self!r}</pre>"


@dataclass
class FrequencyResult:
    """Category counts and proportions for a field."""

    field: str
    counts: dict[str, int]
    proportions: dict[str, float]
    n: int
    mode: str | None = None
    method: str = ""
    exact: bool = True
    queries: int = 0
    warnings: list[str] = field(default_factory=list)

    def _repr_html_(self) -> str:
        return f"<pre>{self!r}</pre>"


@dataclass
class DescribeResult:
    """Multi-statistic summary for one field."""

    field: str
    statistics: dict[str, Any]
    exact: dict[str, bool]
    n: int | None = None
    method: str = ""
    queries: int = 0
    warnings: list[str] = field(default_factory=list)

    def _repr_html_(self) -> str:
        return f"<pre>{self!r}</pre>"


@dataclass
class GroupedStatsResult:
    """Grouped numeric summaries computed from BI aggregate/count queries."""

    field: str | None
    group_field: str
    statistics: dict[str, dict[str, Any]]
    n: dict[str, int]
    method: str = ""
    exact: bool = True
    queries: int = 0
    warnings: list[str] = field(default_factory=list)

    def _repr_html_(self) -> str:
        return f"<pre>{self!r}</pre>"


@dataclass
class CrosstabResult:
    """Two-way category count table computed from BI count queries."""

    row_field: str
    col_field: str
    counts: dict[str, dict[str, int]]
    row_totals: dict[str, int]
    col_totals: dict[str, int]
    row_values: list[Any]
    col_values: list[Any]
    n: int
    method: str = ""
    exact: bool = True
    queries: int = 0
    warnings: list[str] = field(default_factory=list)

    def _repr_html_(self) -> str:
        return f"<pre>{self!r}</pre>"


@dataclass
class MatrixStatsResult:
    """Matrix-valued statistic computed from BI aggregate/count queries."""

    fields: list[str]
    matrix: dict[str, dict[str, float]]
    n: int
    method: str = ""
    exact: bool = True
    queries: int = 0
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def _repr_html_(self) -> str:
        return f"<pre>{self!r}</pre>"


@dataclass
class RegressionSummaryResult:
    """Inferential regression summary reconstructed from BI sufficient statistics."""

    coefficients: dict[str, float]
    standard_errors: dict[str, float]
    tvalues: dict[str, float]
    pvalues: dict[str, float]
    confidence_intervals: dict[str, tuple[float, float]]
    r2: float
    residual_variance: float
    df_resid: int
    n: int
    method: str = ""
    exact: bool = True
    queries: int = 0
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def intercept(self) -> float | None:
        return self.coefficients.get("intercept")

    def _repr_html_(self) -> str:
        return f"<pre>{self!r}</pre>"


@dataclass
class FeatureScreeningResult:
    """Structured feature screening or data quality report."""

    statistics: dict[str, Any]
    method: str = ""
    exact: bool = True
    queries: int = 0
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def _repr_html_(self) -> str:
        return f"<pre>{self!r}</pre>"


class BIStatsSession:
    """Run descriptive statistics through Blind Insight aggregate functions.

    Parameters mirror ``BlindInsightClient`` query identifiers. All methods use
    ``client.aggregate`` or ``client.query(..., count_only=True)`` and never ask
    the client to decrypt records.
    """

    def __init__(
        self,
        client: Any,
        org: str,
        dataset: str,
        schema: str,
        *,
        default_filters: FilterInput = None,
        field_domains: dict[str, Any] | None = None,
        min_cell_size: int = 0,
        max_workers: int = 10,
        retries: int = 3,
    ) -> None:
        self.client = client
        self.org = org
        self.dataset = dataset
        self.schema = schema
        self.default_filters = self._normalize_filters(default_filters)
        self.field_domains = field_domains or {}
        self.min_cell_size = min_cell_size
        self.max_workers = max(1, int(max_workers))
        self.retries = max(1, int(retries))
        self._cache: dict[tuple[Any, ...], Any] = {}

    @staticmethod
    def _normalize_filters(filters: FilterInput) -> list[str]:
        if filters is None:
            return []
        if isinstance(filters, str):
            return [filters]
        return [str(f) for f in filters if str(f)]

    @staticmethod
    def _format_atom(value: Any) -> str:
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, float):
            return format(value, "g")
        return str(value)

    def _merge_filters(self, *filters: FilterInput) -> list[str]:
        merged = list(self.default_filters)
        for item in filters:
            merged.extend(self._normalize_filters(item))
        return merged

    def _cache_get_or_set(self, key: tuple[Any, ...], fn) -> Any:
        if key in self._cache:
            return self._cache[key]
        value = fn()
        self._cache[key] = value
        return value

    def _with_retries(self, fn):
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                return fn()
            except Exception as exc:
                last_error = exc
                if attempt == self.retries - 1:
                    raise
                time.sleep(1.5 * (attempt + 1))
        raise last_error or RuntimeError("BI stats query failed")

    @staticmethod
    def _extract_aggregate_value(response: Any) -> float:
        # Returns NaN when the aggregate response carries no value. Count-oriented
        # callers coerce NaN to 0; min/max/avg/sum keep NaN so an empty result is
        # reported honestly instead of being silently flattened to 0.0.
        if isinstance(response, list):
            records = response
        else:
            records = response.get("records", []) if isinstance(response, dict) else []
        if not records:
            return math.nan
        rec0 = records[0]
        data = rec0.get("data", {}) if isinstance(rec0, dict) else {}
        if isinstance(data, dict) and "value" in data:
            value = data.get("value")
            return float(value) if value is not None else math.nan
        if isinstance(rec0, dict) and "value" in rec0:
            value = rec0.get("value")
            return float(value) if value is not None else math.nan
        return math.nan

    @staticmethod
    def _aggregate_count(value: float) -> int:
        """Coerce an aggregate count result to int, treating an empty result as 0."""
        return 0 if math.isnan(value) else int(value)

    def _aggregate_value(self, agg_filter: str, extra_filters: FilterInput = None) -> float:
        filters = self._merge_filters(extra_filters)
        key = ("aggregate", agg_filter, tuple(filters))

        def run() -> float:
            result = self._with_retries(
                lambda: self.client.aggregate(
                    organization=self.org,
                    dataset_slug=self.dataset,
                    schema_slug=self.schema,
                    agg_filter=agg_filter,
                    extra_filters=filters or None,
                    decrypt=False,
                )
            )
            return self._extract_aggregate_value(result)

        return float(self._cache_get_or_set(key, run))

    def _count_only(self, filters: FilterInput = None) -> int:
        merged = self._merge_filters(filters)
        if not merged:
            return self._unfiltered_count()

        key = ("count_only", tuple(merged))

        def run() -> int:
            result = self._with_retries(
                lambda: self.client.query(
                    organization=self.org,
                    dataset_slug=self.dataset,
                    schema_slug=self.schema,
                    filters=merged or None,
                    limit=1000,
                    count_only=True,
                    decrypt=False,
                )
            )
            return int(result.get("count", 0))

        return int(self._cache_get_or_set(key, run))

    def _unfiltered_count(self) -> int:
        field_name, domain = self._countable_domain()
        agg = self._domain_expr(field_name, "count", domain)
        key = ("unfiltered_count", agg)

        def run() -> int:
            return self._aggregate_count(self._aggregate_value(agg))

        return int(self._cache_get_or_set(key, run))

    def _countable_domain(self) -> tuple[str, Domain]:
        for field_name, domain in self.field_domains.items():
            if (
                isinstance(domain, tuple)
                and len(domain) == 2
                and all(isinstance(value, int | float) for value in domain)
            ):
                return field_name, domain
            if isinstance(domain, range) and domain:
                return field_name, (min(domain), max(domain))
            if isinstance(domain, list) and domain and all(isinstance(value, int | float) for value in domain):
                return field_name, (min(domain), max(domain))
        raise ValueError(
            "Unfiltered count requires at least one numeric field domain; provide field_domains={'field': (low, high)}."
        )

    def _domain_for(self, field_name: str, domain: Domain | None = None) -> Domain:
        candidate = domain if domain is not None else self.field_domains.get(field_name)
        if isinstance(candidate, tuple) and len(candidate) == 2 and all(isinstance(v, int | float) for v in candidate):
            return candidate
        raise ValueError(
            f"Numeric domain required for {field_name!r}; pass domain=(low, high) or provide field_domains[field]."
        )

    def _domain_expr(self, field_name: str, op: str, domain: Domain | None = None) -> str:
        low, high = self._domain_for(field_name, domain)
        return f"{field_name}:{op}({self._format_atom(low)}~{self._format_atom(high)})"

    def _support_for(self, field_name: str, support: Iterable[Any] | None = None) -> list[Any]:
        if support is not None:
            values = list(support)
        else:
            domain = self.field_domains.get(field_name)
            if isinstance(domain, range):
                values = list(domain)
            elif isinstance(domain, list):
                values = list(domain)
            elif isinstance(domain, tuple) and len(domain) == 2 and all(isinstance(v, int) for v in domain):
                values = list(range(int(domain[0]), int(domain[1]) + 1))
            else:
                values = []
        if not values:
            raise ValueError(
                f"Enumerated support required for {field_name!r}; pass support=... "
                "or provide field_domains[field] as a list/range/integer tuple."
            )
        return sorted(values)

    def _category_values_for(self, field_name: str, values: Iterable[Any] | None = None) -> list[Any]:
        if values is not None:
            resolved = list(values)
        else:
            domain = self.field_domains.get(field_name)
            if isinstance(domain, list | range):
                resolved = list(domain)
            elif isinstance(domain, tuple) and not (
                len(domain) == 2 and all(isinstance(v, int | float) for v in domain)
            ):
                resolved = list(domain)
            else:
                resolved = []
        if not resolved:
            raise ValueError(
                f"Category values required for {field_name!r}; pass values=... "
                "or provide field_domains[field] as a category list."
            )
        return resolved

    def _other_category_value(self, field_name: str, selected: Any, other: Any | None) -> Any:
        if other is not None:
            return other
        selected_label = self._format_atom(selected)
        candidates = [
            value for value in self._category_values_for(field_name) if self._format_atom(value) != selected_label
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Cannot infer the alternate value for {field_name!r}; provide it explicitly "
                "or configure a two-value categorical domain."
            )
        return candidates[0]

    def _parallel(self, items: list[Any], fn) -> list[Any]:
        if not items:
            return []
        if self.max_workers <= 1 or len(items) == 1:
            return [fn(item) for item in items]
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(items))) as executor:
            return list(executor.map(fn, items))

    def _min_cell_warnings(self, label: str, count: int) -> list[str]:
        if self.min_cell_size > 0 and count < self.min_cell_size:
            return [
                f"{label} count {count} is below min_cell_size={self.min_cell_size}; "
                "suppression is deferred beyond phase 1."
            ]
        return []

    @staticmethod
    def _validate_suppression_policy(policy: str) -> str:
        if policy not in {"warn", "raise", "suppress", "coarsen"}:
            raise ValueError("suppression_policy must be one of: 'warn', 'raise', 'suppress', 'coarsen'")
        return policy

    def _apply_cell_policy(self, label: str, count: int, policy: str) -> tuple[int | None, list[str], bool]:
        policy = self._validate_suppression_policy(policy)
        if self.min_cell_size <= 0 or count >= self.min_cell_size:
            return count, [], True
        message = f"{label} count {count} is below min_cell_size={self.min_cell_size}"
        if policy == "raise":
            raise ValueError(message)
        if policy == "suppress":
            return None, [f"{message}; cell suppressed"], False
        if policy == "coarsen":
            return count, [f"{message}; generic coarsening is not available for this table"], False
        return count, [message], True

    def _apply_count_policies(
        self,
        counts: dict[str, int],
        *,
        label_prefix: str,
        suppression_policy: str,
    ) -> tuple[dict[str, int | None], list[str], bool]:
        warnings: list[str] = []
        exact = True
        adjusted: dict[str, int | None] = {}
        for label, count in counts.items():
            value, cell_warnings, cell_exact = self._apply_cell_policy(
                f"{label_prefix}:{label}",
                count,
                suppression_policy,
            )
            adjusted[label] = value
            warnings.extend(cell_warnings)
            exact = exact and cell_exact
        return adjusted, warnings, exact

    def _coarsen_ordered_counts(
        self,
        counts: dict[str, int],
        ordered_labels: list[str],
    ) -> tuple[dict[str, int], list[str], bool]:
        if self.min_cell_size <= 0:
            return counts, [], True
        coarsened: dict[str, int] = {}
        bucket_labels: list[str] = []
        bucket_count = 0
        warnings: list[str] = []
        for label in ordered_labels:
            count = counts[label]
            bucket_labels.append(label)
            bucket_count += count
            if bucket_count >= self.min_cell_size:
                coarsened["|".join(bucket_labels)] = bucket_count
                if len(bucket_labels) > 1:
                    warnings.append(f"coarsened {','.join(bucket_labels)} to satisfy min_cell_size")
                bucket_labels = []
                bucket_count = 0
        if bucket_labels:
            if coarsened:
                previous = next(reversed(coarsened))
                count = coarsened.pop(previous) + bucket_count
                merged = f"{previous}|{'|'.join(bucket_labels)}"
                coarsened[merged] = count
                warnings.append(f"coarsened {merged} to satisfy min_cell_size")
            else:
                coarsened["|".join(bucket_labels)] = bucket_count
                warnings.extend(self._min_cell_warnings("|".join(bucket_labels), bucket_count))
        return coarsened, warnings, coarsened == counts

    @staticmethod
    def _bin_midpoint(bin_pair: tuple[Any, Any]) -> float:
        low, high = bin_pair
        return (float(low) + float(high)) / 2.0

    @staticmethod
    def _is_contiguous_numeric_support(values: list[Any]) -> bool:
        if not values:
            return False
        if not all(isinstance(value, int | float) for value in values):
            return False
        sorted_values = sorted(float(value) for value in values)
        return all(math.isclose(right - left, 1.0) for left, right in zip(sorted_values, sorted_values[1:]))

    @staticmethod
    def _safe_log_ratio(numerator: float, denominator: float) -> float:
        return math.log(numerator / denominator) if numerator > 0 and denominator > 0 else 0.0

    @staticmethod
    def _validate_alternative(alternative: str) -> str:
        if alternative not in {"two-sided", "less", "greater"}:
            raise ValueError("alternative must be one of: 'two-sided', 'less', 'greater'")
        return alternative

    @classmethod
    def _normal_pvalue(cls, z_statistic: float, alternative: str) -> float:
        alternative = cls._validate_alternative(alternative)
        if math.isnan(z_statistic):
            return math.nan
        if alternative == "two-sided":
            return float(2.0 * scipy_stats.norm.sf(abs(z_statistic)))
        if alternative == "less":
            return float(scipy_stats.norm.cdf(z_statistic))
        return float(scipy_stats.norm.sf(z_statistic))

    @classmethod
    def _t_pvalue(cls, t_statistic: float, df: float, alternative: str) -> float:
        alternative = cls._validate_alternative(alternative)
        if math.isnan(t_statistic) or math.isnan(df) or df <= 0:
            return math.nan
        if alternative == "two-sided":
            return float(2.0 * scipy_stats.t.sf(abs(t_statistic), df))
        if alternative == "less":
            return float(scipy_stats.t.cdf(t_statistic, df))
        return float(scipy_stats.t.sf(t_statistic, df))

    @staticmethod
    def _ratio(numerator: float, denominator: float) -> float:
        if denominator == 0:
            return math.inf if numerator > 0 else math.nan
        return numerator / denominator

    def count(self, *, where: FilterInput = None) -> StatResult:
        n = self._count_only(where)
        return StatResult(
            statistic=n,
            n=n,
            method="count_only",
            exact=True,
            queries=1,
            warnings=self._min_cell_warnings("count", n),
        )

    def sum_(self, field: str, *, domain: Domain | None = None, where: FilterInput = None) -> StatResult:
        value = self._aggregate_value(self._domain_expr(field, "sum", domain), where)
        n = self.count_range(field, *self._domain_for(field, domain), where=where).statistic
        return StatResult(
            statistic=value,
            estimate=value,
            n=n,
            method="sum",
            exact=True,
            queries=2,
            warnings=self._min_cell_warnings(field, n),
        )

    def mean(self, field: str, *, domain: Domain | None = None, where: FilterInput = None) -> StatResult:
        # Computed as exact sum/count rather than BI ``avg``: the proxy rounds the
        # ``avg`` aggregate to an integer, whereas ``sum`` and ``count`` are exact,
        # so sum/count recovers the true mean at the same query cost (two queries).
        total = self._aggregate_value(self._domain_expr(field, "sum", domain), where)
        n = self.count_range(field, *self._domain_for(field, domain), where=where).statistic
        value = total / n if n else math.nan
        return StatResult(
            statistic=value,
            estimate=value,
            n=n,
            method="sum_over_count",
            exact=True,
            queries=2,
            warnings=self._min_cell_warnings(field, n),
        )

    def min_(self, field: str, *, domain: Domain | None = None, where: FilterInput = None) -> StatResult:
        value = self._aggregate_value(self._domain_expr(field, "min", domain), where)
        return StatResult(statistic=value, estimate=value, method="min", exact=True, queries=1)

    def max_(self, field: str, *, domain: Domain | None = None, where: FilterInput = None) -> StatResult:
        value = self._aggregate_value(self._domain_expr(field, "max", domain), where)
        return StatResult(statistic=value, estimate=value, method="max", exact=True, queries=1)

    def range_(self, field: str, *, domain: Domain | None = None, where: FilterInput = None) -> StatResult:
        min_result = self.min_(field, domain=domain, where=where)
        max_result = self.max_(field, domain=domain, where=where)
        value = max_result.statistic - min_result.statistic
        return StatResult(statistic=value, estimate=value, method="max-min", exact=True, queries=2)

    def proportion(self, numerator_filter: FilterInput, denominator_filter: FilterInput = None) -> StatResult:
        numerator_filters = [
            *self._normalize_filters(denominator_filter),
            *self._normalize_filters(numerator_filter),
        ]
        denominator = self._count_only(denominator_filter)
        numerator = self._count_only(numerator_filters)
        value = numerator / denominator if denominator else math.nan
        warnings = [] if denominator else ["denominator count is zero"]
        warnings.extend(self._min_cell_warnings("numerator", numerator))
        warnings.extend(self._min_cell_warnings("denominator", denominator))
        return StatResult(
            statistic=value,
            estimate={"numerator": numerator, "denominator": denominator},
            n=denominator,
            method="count_ratio",
            exact=True,
            queries=2,
            warnings=warnings,
        )

    def rate(self, numerator_filter: FilterInput, denominator_filter: FilterInput = None) -> StatResult:
        result = self.proportion(numerator_filter, denominator_filter)
        result.method = "rate"
        return result

    def _proportion_counts(
        self,
        event_filter: FilterInput,
        denominator_filter: FilterInput = None,
    ) -> tuple[int, int, list[str]]:
        numerator_filters = [
            *self._normalize_filters(denominator_filter),
            *self._normalize_filters(event_filter),
        ]
        denominator = self._count_only(denominator_filter)
        numerator = self._count_only(numerator_filters)
        warnings = [] if denominator else ["denominator count is zero"]
        warnings.extend(self._min_cell_warnings("numerator", numerator))
        warnings.extend(self._min_cell_warnings("denominator", denominator))
        return numerator, denominator, warnings

    def count_range(
        self,
        field: str,
        low: int | float,
        high: int | float,
        *,
        where: FilterInput = None,
    ) -> StatResult:
        agg = f"{field}:count({self._format_atom(low)}~{self._format_atom(high)})"
        count = self._aggregate_count(self._aggregate_value(agg, where))
        return StatResult(
            statistic=count,
            n=count,
            method="count_range",
            exact=True,
            queries=1,
            warnings=self._min_cell_warnings(f"{field}:{self._format_atom(low)}~{self._format_atom(high)}", count),
        )

    def count_eq(self, field: str, value: Any, *, where: FilterInput = None) -> StatResult:
        count = self._count_only([*self._normalize_filters(where), f"{field}:{self._format_atom(value)}"])
        return StatResult(
            statistic=count,
            n=count,
            method="count_eq",
            exact=True,
            queries=1,
            warnings=self._min_cell_warnings(f"{field}:{self._format_atom(value)}", count),
        )

    def histogram(
        self,
        field: str,
        *,
        bins: list[Any] | None = None,
        support: Iterable[Any] | None = None,
        where: FilterInput = None,
        suppression_policy: str = "warn",
    ) -> HistogramResult:
        self._validate_suppression_policy(suppression_policy)
        if support is not None:
            support_values = list(support)

            if self._is_contiguous_numeric_support(support_values):
                ordered = sorted(support_values)

                # Single-value numeric ranges can over-count on the encrypted
                # index. Cumulative range counts are exact, so adjacent
                # differences recover each integer bucket without row access.
                def cumulative(value: Any) -> tuple[Any, int]:
                    return value, self.count_range(field, ordered[0], value, where=where).statistic

                cumulative_pairs = self._parallel(ordered, cumulative)
                previous = 0
                pairs = []
                for value, cumulative_count in cumulative_pairs:
                    pairs.append((self._format_atom(value), max(0, cumulative_count - previous)))
                    previous = cumulative_count
            else:

                def count_value(value: Any) -> tuple[str, int]:
                    return self._format_atom(value), self.count_range(field, value, value, where=where).statistic

                pairs = self._parallel(support_values, count_value)
            counts = dict(pairs)
            if suppression_policy == "coarsen":
                ordered_labels = [self._format_atom(value) for value in support_values]
                counts, warnings, exact = self._coarsen_ordered_counts(counts, ordered_labels)
            else:
                adjusted, warnings, exact = self._apply_count_policies(
                    counts,
                    label_prefix=field,
                    suppression_policy=suppression_policy,
                )
                counts = adjusted
            return HistogramResult(
                field=field,
                counts=counts,
                bins=support_values,
                n=sum(count for count in counts.values() if count is not None),
                method="support_counts",
                exact=exact,
                queries=len(support_values),
                warnings=warnings,
            )

        if bins is None:
            raise ValueError("histogram requires either bins=... or support=...")

        range_bins = self._normalize_bins(bins)

        def count_bin(bin_pair: tuple[Any, Any]) -> tuple[str, int]:
            low, high = bin_pair
            label = f"{self._format_atom(low)}~{self._format_atom(high)}"
            return label, self.count_range(field, low, high, where=where).statistic

        pairs = self._parallel(range_bins, count_bin)
        counts = dict(pairs)
        if suppression_policy == "coarsen":
            ordered_labels = [f"{self._format_atom(low)}~{self._format_atom(high)}" for low, high in range_bins]
            counts, warnings, exact = self._coarsen_ordered_counts(counts, ordered_labels)
        else:
            adjusted, warnings, exact = self._apply_count_policies(
                counts,
                label_prefix=field,
                suppression_policy=suppression_policy,
            )
            counts = adjusted
        return HistogramResult(
            field=field,
            counts=counts,
            bins=range_bins,
            n=sum(count for count in counts.values() if count is not None),
            method="range_bins",
            exact=exact,
            queries=len(range_bins),
            warnings=warnings,
        )

    @staticmethod
    def _normalize_bins(bins: list[Any]) -> list[tuple[Any, Any]]:
        if not bins:
            raise ValueError("bins must contain at least one bucket")
        if all(isinstance(item, tuple) and len(item) == 2 for item in bins):
            return list(bins)
        if len(bins) < 2:
            raise ValueError("edge-style bins require at least two values")
        # Edge-style bins are cut points interpreted as half-open intervals
        # [e_i, e_{i+1}) with the final interval closed, matching numpy.histogram.
        # BI count_range is inclusive on both ends, so pairing consecutive edges
        # directly (zip) would make adjacent ranges share a boundary value and
        # count it twice. Make each interior right edge exclusive by dropping to
        # the next-lower integer; the final bin stays closed. This requires
        # integer edges (the encrypted index is integer-valued); non-integer
        # cut points cannot be turned into non-overlapping inclusive ranges
        # unambiguously, so callers must pass explicit (low, high) tuple bins.
        if not all(isinstance(edge, int) and not isinstance(edge, bool) for edge in bins):
            raise ValueError(
                "edge-style bins must be integer cut points; pass explicit (low, high) "
                "tuple bins for non-integer fields so bin boundaries are unambiguous"
            )
        edges = list(bins)
        last = len(edges) - 2
        return [(edges[idx], edges[idx + 1] if idx == last else edges[idx + 1] - 1) for idx in range(len(edges) - 1)]

    def frequency(
        self,
        field: str,
        *,
        values: Iterable[Any] | None = None,
        where: FilterInput = None,
        suppression_policy: str = "warn",
    ) -> FrequencyResult:
        self._validate_suppression_policy(suppression_policy)
        if values is None:
            domain = self.field_domains.get(field)
            if isinstance(domain, list | tuple | range) and not (
                isinstance(domain, tuple) and len(domain) == 2 and all(isinstance(v, int | float) for v in domain)
            ):
                values = list(domain)
            else:
                raise ValueError(
                    f"Category values required for {field!r}; pass values=... "
                    "or provide field_domains[field] as a category list."
                )
        value_list = list(values)

        def count_value(value: Any) -> tuple[str, int]:
            label = self._format_atom(value)
            return label, self.count_eq(field, value, where=where).statistic

        pairs = self._parallel(value_list, count_value)
        counts = dict(pairs)
        adjusted, warnings, exact = self._apply_count_policies(
            counts,
            label_prefix=field,
            suppression_policy=suppression_policy,
        )
        n = sum(count for count in adjusted.values() if count is not None)
        proportions = {key: (count / n if count is not None and n else math.nan) for key, count in adjusted.items()}
        unsuppressed_counts = {key: count for key, count in adjusted.items() if count is not None}
        mode = max(unsuppressed_counts.items(), key=lambda item: item[1])[0] if unsuppressed_counts else None
        if suppression_policy == "coarsen" and self.min_cell_size > 0:
            warnings.extend(
                warning
                for label, count in counts.items()
                for warning in self._min_cell_warnings(f"{field}:{label}", count)
            )
        return FrequencyResult(
            field=field,
            counts=adjusted,
            proportions=proportions,
            n=n,
            mode=mode,
            method="category_counts",
            exact=exact,
            queries=len(value_list),
            warnings=warnings,
        )

    def mode(self, field: str, *, values: Iterable[Any] | None = None, where: FilterInput = None) -> StatResult:
        freq = self.frequency(field, values=values, where=where)
        mode_count = freq.counts.get(freq.mode, 0) if freq.mode is not None else 0
        # Map the winning bucket label back to its original typed value so int/bool
        # categories round-trip instead of degrading to their string label.
        resolved_values = list(values) if values is not None else self._category_values_for(field)
        label_to_value = {self._format_atom(value): value for value in resolved_values}
        typed_mode = label_to_value.get(freq.mode, freq.mode) if freq.mode is not None else None
        return StatResult(
            statistic=typed_mode,
            estimate={"count": mode_count, "proportion": freq.proportions.get(freq.mode, math.nan)},
            n=freq.n,
            method="frequency_mode",
            exact=freq.exact,
            queries=freq.queries,
            warnings=freq.warnings,
        )

    def var(
        self,
        field: str,
        *,
        support: Iterable[Any] | None = None,
        ddof: int = 1,
        where: FilterInput = None,
    ) -> StatResult:
        support_values = self._support_for(field, support)
        hist = self.histogram(field, support=support_values, where=where)
        n = hist.n
        warnings = list(hist.warnings)
        if n <= ddof:
            warnings.append("not enough observations for requested ddof")
            return StatResult(
                statistic=math.nan,
                n=n,
                method=f"support_counts_ddof_{ddof}",
                exact=True,
                queries=hist.queries,
                warnings=warnings,
            )
        sum_x = 0.0
        sum_x2 = 0.0
        for raw_value in support_values:
            count = hist.counts[self._format_atom(raw_value)]
            value = float(raw_value)
            sum_x += value * count
            sum_x2 += value * value * count
        variance = (sum_x2 - (sum_x * sum_x / n)) / (n - ddof)
        variance = max(0.0, variance)
        return StatResult(
            statistic=variance,
            estimate={"sum": sum_x, "sum_squares": sum_x2},
            n=n,
            method=f"support_counts_ddof_{ddof}",
            exact=True,
            queries=hist.queries,
            warnings=warnings,
        )

    def std(
        self,
        field: str,
        *,
        support: Iterable[Any] | None = None,
        ddof: int = 1,
        where: FilterInput = None,
    ) -> StatResult:
        variance = self.var(field, support=support, ddof=ddof, where=where)
        value = math.sqrt(variance.statistic) if not math.isnan(variance.statistic) else math.nan
        return StatResult(
            statistic=value,
            estimate=variance.estimate,
            n=variance.n,
            method=variance.method.replace("support_counts", "support_counts_std"),
            exact=variance.exact,
            queries=variance.queries,
            warnings=variance.warnings,
        )

    def quantile(
        self,
        field: str,
        q: float,
        *,
        support: Iterable[Any] | None = None,
        where: FilterInput = None,
    ) -> StatResult:
        if q < 0 or q > 1:
            raise ValueError("q must be between 0 and 1")
        support_values = self._support_for(field, support)
        hist = self.histogram(field, support=support_values, where=where)
        n = hist.n
        warnings = list(hist.warnings)
        if n == 0:
            warnings.append("no observations for quantile")
            return StatResult(
                statistic=None,
                n=0,
                method="support_cdf",
                exact=True,
                queries=hist.queries,
                warnings=warnings,
            )
        rank = max(1, math.ceil(q * n))
        cumulative = 0
        result = support_values[-1]
        for value in support_values:
            cumulative += hist.counts[self._format_atom(value)]
            if cumulative >= rank:
                result = value
                break
        return StatResult(
            statistic=result,
            estimate={"q": q, "rank": rank},
            n=n,
            method="support_cdf",
            exact=True,
            queries=hist.queries,
            warnings=warnings,
        )

    def median(self, field: str, *, support: Iterable[Any] | None = None, where: FilterInput = None) -> StatResult:
        result = self.quantile(field, 0.5, support=support, where=where)
        result.method = "support_cdf_median"
        return result

    def iqr(self, field: str, *, support: Iterable[Any] | None = None, where: FilterInput = None) -> StatResult:
        q25 = self.quantile(field, 0.25, support=support, where=where)
        q75 = self.quantile(field, 0.75, support=support, where=where)
        value = q75.statistic - q25.statistic if q25.statistic is not None and q75.statistic is not None else None
        return StatResult(
            statistic=value,
            estimate={"q25": q25.statistic, "q75": q75.statistic},
            n=q75.n,
            method="support_cdf_iqr",
            exact=q25.exact and q75.exact,
            queries=q25.queries + q75.queries,
            warnings=[*q25.warnings, *q75.warnings],
        )

    def percentile_rank(
        self,
        field: str,
        value: int | float,
        *,
        support: Iterable[Any] | None = None,
        where: FilterInput = None,
    ) -> StatResult:
        if support is not None:
            support_values = self._support_for(field, support)
            hist = self.histogram(field, support=support_values, where=where)
            n = hist.n
            le_count = sum(count for raw, count in hist.counts.items() if float(raw) <= float(value))
            queries = hist.queries
        else:
            low, high = self._domain_for(field)
            le_count = self.count_range(field, low, value, where=where).statistic
            n = self.count_range(field, low, high, where=where).statistic
            queries = 2
        rank = le_count / n if n else math.nan
        warnings = [] if n else ["no observations for percentile rank"]
        return StatResult(
            statistic=rank,
            estimate={"count_lte": le_count},
            n=n,
            method="cdf_count",
            exact=True,
            queries=queries,
            warnings=warnings,
        )

    def ecdf(self, field: str, values: Iterable[int | float], *, where: FilterInput = None) -> StatResult:
        value_list = list(values)
        low, high = self._domain_for(field)
        total = self.count_range(field, low, high, where=where).statistic

        def rank_for(value: int | float) -> tuple[str, float]:
            count = self.count_range(field, low, value, where=where).statistic
            return self._format_atom(value), count / total if total else math.nan

        pairs = self._parallel(value_list, rank_for)
        warnings = [] if total else ["no observations for ecdf"]
        return StatResult(
            statistic=dict(pairs),
            n=total,
            method="ecdf_range_counts",
            exact=True,
            queries=len(value_list) + 1,
            warnings=warnings,
        )

    def crosstab(
        self,
        row_field: str,
        col_field: str,
        *,
        row_values: Iterable[Any] | None = None,
        col_values: Iterable[Any] | None = None,
        where: FilterInput = None,
        suppression_policy: str = "warn",
    ) -> CrosstabResult:
        self._validate_suppression_policy(suppression_policy)
        rows = self._category_values_for(row_field, row_values)
        cols = self._category_values_for(col_field, col_values)
        cells = [(row_value, col_value) for row_value in rows for col_value in cols]

        def count_cell(cell: tuple[Any, Any]) -> tuple[str, str, int]:
            row_value, col_value = cell
            row_label = self._format_atom(row_value)
            col_label = self._format_atom(col_value)
            filters = [
                *self._normalize_filters(where),
                f"{row_field}:{row_label}",
                f"{col_field}:{col_label}",
            ]
            return row_label, col_label, self._count_only(filters)

        triples = self._parallel(cells, count_cell)
        counts = {self._format_atom(row_value): {} for row_value in rows}
        warnings: list[str] = []
        exact = True
        for row_label, col_label, count in triples:
            adjusted, cell_warnings, cell_exact = self._apply_cell_policy(
                f"{row_field}:{row_label},{col_field}:{col_label}",
                count,
                suppression_policy,
            )
            counts[row_label][col_label] = adjusted
            warnings.extend(cell_warnings)
            exact = exact and cell_exact

        row_totals = {
            row_label: sum(count for count in col_counts.values() if count is not None)
            for row_label, col_counts in counts.items()
        }
        col_labels = [self._format_atom(value) for value in cols]
        col_totals = {
            col_label: sum(counts[self._format_atom(row_value)].get(col_label, 0) or 0 for row_value in rows)
            for col_label in col_labels
        }
        return CrosstabResult(
            row_field=row_field,
            col_field=col_field,
            counts=counts,
            row_totals=row_totals,
            col_totals=col_totals,
            row_values=rows,
            col_values=cols,
            n=sum(row_totals.values()),
            method="count_only_crosstab",
            exact=exact,
            queries=len(cells),
            warnings=warnings,
        )

    @staticmethod
    def _observed_matrix(table: CrosstabResult) -> list[list[int]]:
        return [
            [
                table.counts[BIStatsSession._format_atom(row)][BIStatsSession._format_atom(col)]
                for col in table.col_values
            ]
            for row in table.row_values
        ]

    @staticmethod
    def _expected_dict(table: CrosstabResult, expected: list[list[float]]) -> dict[str, dict[str, float]]:
        return {
            BIStatsSession._format_atom(row): {
                BIStatsSession._format_atom(col): float(expected[row_idx][col_idx])
                for col_idx, col in enumerate(table.col_values)
            }
            for row_idx, row in enumerate(table.row_values)
        }

    def chisquare(
        self,
        field: str,
        *,
        values: Iterable[Any] | None = None,
        expected: Iterable[float] | dict[str, float] | None = None,
        where: FilterInput = None,
    ) -> StatResult:
        freq = self.frequency(field, values=values, where=where)
        labels = list(freq.counts)
        observed = [freq.counts[label] for label in labels]
        if expected is None:
            expected_values = [freq.n / len(labels) if labels else math.nan for _ in labels]
        elif isinstance(expected, dict):
            expected_values = [float(expected[label]) for label in labels]
        else:
            expected_values = [float(value) for value in expected]
        if len(expected_values) != len(observed):
            raise ValueError("expected must have the same length as observed values")
        if not observed or freq.n == 0:
            return StatResult(
                statistic=math.nan,
                pvalue=math.nan,
                estimate={"observed": dict(freq.counts), "expected": dict(zip(labels, expected_values))},
                n=freq.n,
                method="chisquare_goodness_of_fit",
                exact=True,
                queries=freq.queries,
                warnings=[*freq.warnings, "no observations for chi-square goodness of fit"],
            )

        result = scipy_stats.chisquare(f_obs=observed, f_exp=expected_values)
        return StatResult(
            statistic=float(result.statistic),
            pvalue=float(result.pvalue),
            estimate={
                "observed": dict(freq.counts),
                "expected": dict(zip(labels, expected_values)),
                "dof": len(observed) - 1,
            },
            n=freq.n,
            method="chisquare_goodness_of_fit",
            exact=True,
            queries=freq.queries,
            warnings=freq.warnings,
        )

    def chi2_independence(
        self,
        row_field: str,
        col_field: str,
        *,
        row_values: Iterable[Any] | None = None,
        col_values: Iterable[Any] | None = None,
        where: FilterInput = None,
        correction: bool = True,
    ) -> StatResult:
        table = self.crosstab(row_field, col_field, row_values=row_values, col_values=col_values, where=where)
        observed = self._observed_matrix(table)
        if table.n == 0:
            return StatResult(
                statistic=math.nan,
                pvalue=math.nan,
                estimate={"observed": table.counts, "expected": {}, "dof": 0},
                n=table.n,
                method="chi2_independence",
                exact=True,
                queries=table.queries,
                warnings=[*table.warnings, "no observations for chi-square independence"],
            )

        statistic, pvalue, dof, expected = scipy_stats.chi2_contingency(observed, correction=correction)
        return StatResult(
            statistic=float(statistic),
            pvalue=float(pvalue),
            estimate={
                "observed": table.counts,
                "expected": self._expected_dict(table, expected.tolist()),
                "dof": int(dof),
                "row_totals": table.row_totals,
                "col_totals": table.col_totals,
            },
            n=table.n,
            method="chi2_independence",
            exact=True,
            queries=table.queries,
            warnings=table.warnings,
        )

    def fisher_exact(
        self,
        row_field: str,
        col_field: str,
        *,
        row_values: Iterable[Any] | None = None,
        col_values: Iterable[Any] | None = None,
        where: FilterInput = None,
        alternative: str = "two-sided",
    ) -> StatResult:
        alternative = self._validate_alternative(alternative)
        table = self.crosstab(row_field, col_field, row_values=row_values, col_values=col_values, where=where)
        observed = self._observed_matrix(table)
        if len(observed) != 2 or any(len(row) != 2 for row in observed):
            raise ValueError("fisher_exact requires a 2x2 table")
        result = scipy_stats.fisher_exact(observed, alternative=alternative)
        return StatResult(
            statistic=float(result.statistic),
            pvalue=float(result.pvalue),
            estimate={"observed": table.counts, "alternative": alternative},
            n=table.n,
            method="fisher_exact_2x2",
            exact=True,
            queries=table.queries,
            warnings=table.warnings,
        )

    def _binary_table(
        self,
        exposure_field: str,
        outcome_field: str,
        exposed: Any,
        outcome: Any,
        unexposed: Any | None = None,
        nonoutcome: Any | None = None,
        where: FilterInput = None,
    ) -> tuple[CrosstabResult, float, float, float, float]:
        unexposed = self._other_category_value(exposure_field, exposed, unexposed)
        nonoutcome = self._other_category_value(outcome_field, outcome, nonoutcome)
        table = self.crosstab(
            exposure_field,
            outcome_field,
            row_values=[exposed, unexposed],
            col_values=[outcome, nonoutcome],
            where=where,
        )
        exposed_label = self._format_atom(exposed)
        unexposed_label = self._format_atom(unexposed)
        outcome_label = self._format_atom(outcome)
        nonoutcome_label = self._format_atom(nonoutcome)
        a = float(table.counts[exposed_label][outcome_label])
        b = float(table.counts[exposed_label][nonoutcome_label])
        c = float(table.counts[unexposed_label][outcome_label])
        d = float(table.counts[unexposed_label][nonoutcome_label])
        return table, a, b, c, d

    def odds_ratio(
        self,
        exposure_field: str,
        outcome_field: str,
        exposed: Any,
        outcome: Any,
        unexposed: Any | None = None,
        nonoutcome: Any | None = None,
        *,
        where: FilterInput = None,
        correction: float = 0.0,
    ) -> StatResult:
        table, a, b, c, d = self._binary_table(
            exposure_field, outcome_field, exposed, outcome, unexposed, nonoutcome, where
        )
        cells = [a + correction, b + correction, c + correction, d + correction]
        value = self._ratio(cells[0] * cells[3], cells[1] * cells[2])
        warnings = list(table.warnings)
        if correction:
            warnings.append(f"added correction={correction:g} to each 2x2 cell")
        return StatResult(
            statistic=value,
            estimate={"observed": table.counts, "correction": correction},
            n=table.n,
            method="odds_ratio_2x2",
            exact=True,
            queries=table.queries,
            warnings=warnings,
        )

    def relative_risk(
        self,
        exposure_field: str,
        outcome_field: str,
        exposed: Any,
        outcome: Any,
        unexposed: Any | None = None,
        nonoutcome: Any | None = None,
        *,
        where: FilterInput = None,
        correction: float = 0.0,
    ) -> StatResult:
        table, a, b, c, d = self._binary_table(
            exposure_field, outcome_field, exposed, outcome, unexposed, nonoutcome, where
        )
        exposed_events = a + correction
        exposed_total = a + b + (2 * correction)
        unexposed_events = c + correction
        unexposed_total = c + d + (2 * correction)
        value = self._ratio(
            self._ratio(exposed_events, exposed_total),
            self._ratio(unexposed_events, unexposed_total),
        )
        warnings = list(table.warnings)
        if correction:
            warnings.append(f"added correction={correction:g} to each 2x2 cell")
        return StatResult(
            statistic=value,
            estimate={"observed": table.counts, "correction": correction},
            n=table.n,
            method="relative_risk_2x2",
            exact=True,
            queries=table.queries,
            warnings=warnings,
        )

    def ztest_proportion(
        self,
        event_filter: FilterInput,
        denominator_filter: FilterInput = None,
        *,
        p: float = 0.5,
        alternative: str = "two-sided",
    ) -> StatResult:
        self._validate_alternative(alternative)
        if p < 0 or p > 1:
            raise ValueError("p must be between 0 and 1")
        successes, trials, warnings = self._proportion_counts(event_filter, denominator_filter)
        estimate = successes / trials if trials else math.nan
        se = math.sqrt(p * (1.0 - p) / trials) if trials and 0 < p < 1 else 0.0
        statistic = (estimate - p) / se if se else math.nan
        return StatResult(
            statistic=statistic,
            pvalue=self._normal_pvalue(statistic, alternative),
            estimate={"successes": successes, "trials": trials, "proportion": estimate, "null": p},
            n=trials,
            method="one_proportion_ztest",
            exact=True,
            queries=2,
            warnings=warnings,
        )

    def ztest_proportions(
        self,
        event_filter: FilterInput,
        group_field: str,
        group_a: Any,
        group_b: Any,
        *,
        where: FilterInput = None,
        alternative: str = "two-sided",
    ) -> StatResult:
        self._validate_alternative(alternative)
        group_a_filter = [*self._normalize_filters(where), f"{group_field}:{self._format_atom(group_a)}"]
        group_b_filter = [*self._normalize_filters(where), f"{group_field}:{self._format_atom(group_b)}"]
        successes_a, trials_a, warnings_a = self._proportion_counts(event_filter, group_a_filter)
        successes_b, trials_b, warnings_b = self._proportion_counts(event_filter, group_b_filter)
        prop_a = successes_a / trials_a if trials_a else math.nan
        prop_b = successes_b / trials_b if trials_b else math.nan
        pooled_n = trials_a + trials_b
        pooled = (successes_a + successes_b) / pooled_n if pooled_n else math.nan
        se = (
            math.sqrt(pooled * (1.0 - pooled) * ((1.0 / trials_a) + (1.0 / trials_b))) if trials_a and trials_b else 0.0
        )
        statistic = ((prop_a - prop_b) / se) if se else math.nan
        return StatResult(
            statistic=statistic,
            pvalue=self._normal_pvalue(statistic, alternative),
            estimate={
                "group_a": self._format_atom(group_a),
                "group_b": self._format_atom(group_b),
                "successes_a": successes_a,
                "successes_b": successes_b,
                "trials_a": trials_a,
                "trials_b": trials_b,
                "proportion_a": prop_a,
                "proportion_b": prop_b,
                "pooled": pooled,
            },
            n={"group_a": trials_a, "group_b": trials_b},
            method="two_proportion_ztest",
            exact=True,
            queries=4,
            warnings=[*warnings_a, *warnings_b],
        )

    def proportion_ci(
        self,
        event_filter: FilterInput,
        denominator_filter: FilterInput = None,
        *,
        confidence_level: float = 0.95,
        method: str = "wilson",
    ) -> StatResult:
        if confidence_level <= 0 or confidence_level >= 1:
            raise ValueError("confidence_level must be between 0 and 1")
        if method not in {"wilson", "clopper-pearson"}:
            raise ValueError("method must be either 'wilson' or 'clopper-pearson'")
        successes, trials, warnings = self._proportion_counts(event_filter, denominator_filter)
        estimate = successes / trials if trials else math.nan
        alpha = 1.0 - confidence_level
        if not trials:
            interval = (math.nan, math.nan)
        elif method == "wilson":
            z = float(scipy_stats.norm.ppf(1.0 - alpha / 2.0))
            denominator = 1.0 + (z * z / trials)
            center = (estimate + (z * z / (2.0 * trials))) / denominator
            half_width = z * math.sqrt((estimate * (1.0 - estimate) + (z * z / (4.0 * trials))) / trials)
            interval = (max(0.0, center - half_width / denominator), min(1.0, center + half_width / denominator))
        else:
            lower = (
                0.0 if successes == 0 else float(scipy_stats.beta.ppf(alpha / 2.0, successes, trials - successes + 1))
            )
            upper = (
                1.0
                if successes == trials
                else float(scipy_stats.beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
            )
            interval = (lower, upper)
        return StatResult(
            statistic=estimate,
            estimate={"successes": successes, "trials": trials, "confidence_level": confidence_level},
            confidence_interval=interval,
            n=trials,
            method=f"{method}_proportion_interval",
            exact=True,
            queries=2,
            warnings=warnings,
        )

    def _group_filter(self, group_field: str, group_value: Any, where: FilterInput = None) -> list[str]:
        return [*self._normalize_filters(where), f"{group_field}:{self._format_atom(group_value)}"]

    def _numeric_summary(
        self,
        field: str,
        support: Iterable[Any] | None = None,
        where: FilterInput = None,
        ddof: int = 1,
    ) -> dict[str, Any]:
        mean = self.mean(field, where=where)
        variance = self.var(field, support=support, ddof=ddof, where=where)
        n = int(mean.n if mean.n is not None else variance.n or 0)
        return {
            "n": n,
            "mean": float(mean.statistic),
            "variance": float(variance.statistic),
            "queries": mean.queries + variance.queries,
            "exact": mean.exact and variance.exact,
            "warnings": [*mean.warnings, *variance.warnings],
        }

    def groupby_count(
        self,
        group_field: str,
        *,
        values: Iterable[Any] | None = None,
        where: FilterInput = None,
    ) -> GroupedStatsResult:
        groups = self._category_values_for(group_field, values)

        def count_group(value: Any) -> tuple[str, int, list[str]]:
            label = self._format_atom(value)
            result = self.count_eq(group_field, value, where=where)
            return label, int(result.statistic), result.warnings

        triples = self._parallel(groups, count_group)
        statistics = {label: {"count": count} for label, count, _ in triples}
        n = {label: count for label, count, _ in triples}
        warnings = [warning for _, _, group_warnings in triples for warning in group_warnings]
        return GroupedStatsResult(
            field=None,
            group_field=group_field,
            statistics=statistics,
            n=n,
            method="groupby_count",
            exact=True,
            queries=len(groups),
            warnings=warnings,
        )

    def groupby_mean(
        self,
        field: str,
        group_field: str,
        *,
        values: Iterable[Any] | None = None,
        domain: Domain | None = None,
        where: FilterInput = None,
    ) -> GroupedStatsResult:
        groups = self._category_values_for(group_field, values)

        def mean_group(value: Any) -> tuple[str, StatResult]:
            label = self._format_atom(value)
            return label, self.mean(field, domain=domain, where=self._group_filter(group_field, value, where))

        pairs = self._parallel(groups, mean_group)
        statistics = {label: {"mean": result.statistic, "count": int(result.n or 0)} for label, result in pairs}
        n = {label: int(result.n or 0) for label, result in pairs}
        warnings = [warning for _, result in pairs for warning in result.warnings]
        return GroupedStatsResult(
            field=field,
            group_field=group_field,
            statistics=statistics,
            n=n,
            method="groupby_mean",
            exact=all(result.exact for _, result in pairs),
            queries=sum(result.queries for _, result in pairs),
            warnings=warnings,
        )

    def groupby_var(
        self,
        field: str,
        group_field: str,
        *,
        values: Iterable[Any] | None = None,
        support: Iterable[Any] | None = None,
        ddof: int = 1,
        where: FilterInput = None,
    ) -> GroupedStatsResult:
        groups = self._category_values_for(group_field, values)
        support_values = self._support_for(field, support)

        def var_group(value: Any) -> tuple[str, StatResult]:
            label = self._format_atom(value)
            return label, self.var(
                field, support=support_values, ddof=ddof, where=self._group_filter(group_field, value, where)
            )

        pairs = self._parallel(groups, var_group)
        statistics = {
            label: {"variance": result.statistic, "count": int(result.n or 0), "ddof": ddof} for label, result in pairs
        }
        n = {label: int(result.n or 0) for label, result in pairs}
        warnings = [warning for _, result in pairs for warning in result.warnings]
        return GroupedStatsResult(
            field=field,
            group_field=group_field,
            statistics=statistics,
            n=n,
            method=f"groupby_var_ddof_{ddof}",
            exact=all(result.exact for _, result in pairs),
            queries=sum(result.queries for _, result in pairs),
            warnings=warnings,
        )

    def groupby_std(
        self,
        field: str,
        group_field: str,
        *,
        values: Iterable[Any] | None = None,
        support: Iterable[Any] | None = None,
        ddof: int = 1,
        where: FilterInput = None,
    ) -> GroupedStatsResult:
        variances = self.groupby_var(field, group_field, values=values, support=support, ddof=ddof, where=where)
        statistics = {
            label: {
                "std": math.sqrt(values_["variance"]) if not math.isnan(values_["variance"]) else math.nan,
                "variance": values_["variance"],
                "count": values_["count"],
                "ddof": ddof,
            }
            for label, values_ in variances.statistics.items()
        }
        return GroupedStatsResult(
            field=field,
            group_field=group_field,
            statistics=statistics,
            n=variances.n,
            method=f"groupby_std_ddof_{ddof}",
            exact=variances.exact,
            queries=variances.queries,
            warnings=variances.warnings,
        )

    def mean_ci(
        self,
        field: str,
        *,
        support: Iterable[Any] | None = None,
        where: FilterInput = None,
        confidence_level: float = 0.95,
    ) -> StatResult:
        if confidence_level <= 0 or confidence_level >= 1:
            raise ValueError("confidence_level must be between 0 and 1")
        summary = self._numeric_summary(field, support=support, where=where, ddof=1)
        n = summary["n"]
        mean = summary["mean"]
        variance = summary["variance"]
        warnings = list(summary["warnings"])
        if n <= 1 or math.isnan(variance):
            interval = (math.nan, math.nan)
            warnings.append("not enough observations for mean confidence interval")
        else:
            alpha = 1.0 - confidence_level
            se = math.sqrt(variance / n)
            critical = float(scipy_stats.t.ppf(1.0 - alpha / 2.0, n - 1))
            interval = (mean - critical * se, mean + critical * se)
        return StatResult(
            statistic=mean,
            estimate={"mean": mean, "variance": variance, "confidence_level": confidence_level},
            confidence_interval=interval,
            n=n,
            method="t_mean_interval",
            exact=summary["exact"],
            queries=summary["queries"],
            warnings=warnings,
        )

    def ttest_1samp(
        self,
        field: str,
        popmean: float,
        *,
        support: Iterable[Any] | None = None,
        where: FilterInput = None,
        alternative: str = "two-sided",
        confidence_level: float = 0.95,
    ) -> StatResult:
        self._validate_alternative(alternative)
        if confidence_level <= 0 or confidence_level >= 1:
            raise ValueError("confidence_level must be between 0 and 1")
        summary = self._numeric_summary(field, support=support, where=where, ddof=1)
        n = summary["n"]
        mean = summary["mean"]
        variance = summary["variance"]
        warnings = list(summary["warnings"])
        interval: tuple[float, float] | None = None
        if n <= 1 or math.isnan(variance):
            statistic = math.nan
            df = math.nan
            warnings.append("not enough observations for one-sample t-test")
        else:
            df = float(n - 1)
            se = math.sqrt(variance / n)
            statistic = (mean - popmean) / se if se else math.nan
            critical = float(scipy_stats.t.ppf(1.0 - (1.0 - confidence_level) / 2.0, df))
            interval = (mean - critical * se, mean + critical * se)
        return StatResult(
            statistic=statistic,
            pvalue=self._t_pvalue(statistic, df, alternative),
            estimate={"mean": mean, "variance": variance, "popmean": popmean, "df": df, "alternative": alternative},
            confidence_interval=interval,
            n=n,
            method="one_sample_ttest",
            exact=summary["exact"],
            queries=summary["queries"],
            warnings=warnings,
        )

    def ttest_ind(
        self,
        field: str,
        group_field: str,
        group_a: Any,
        group_b: Any,
        *,
        support: Iterable[Any] | None = None,
        equal_var: bool = False,
        where: FilterInput = None,
        alternative: str = "two-sided",
        confidence_level: float = 0.95,
    ) -> StatResult:
        self._validate_alternative(alternative)
        if confidence_level <= 0 or confidence_level >= 1:
            raise ValueError("confidence_level must be between 0 and 1")
        support_values = self._support_for(field, support)
        summary_a = self._numeric_summary(
            field, support=support_values, where=self._group_filter(group_field, group_a, where)
        )
        summary_b = self._numeric_summary(
            field, support=support_values, where=self._group_filter(group_field, group_b, where)
        )
        n_a = summary_a["n"]
        n_b = summary_b["n"]
        mean_a = summary_a["mean"]
        mean_b = summary_b["mean"]
        var_a = summary_a["variance"]
        var_b = summary_b["variance"]
        warnings = [*summary_a["warnings"], *summary_b["warnings"]]
        interval: tuple[float, float] | None = None
        if n_a <= 1 or n_b <= 1 or math.isnan(var_a) or math.isnan(var_b):
            statistic = math.nan
            pvalue = math.nan
            df = math.nan
            warnings.append("not enough observations for two-sample t-test")
        else:
            # SciPy computes the statistic and p-value from the same group moments
            # we already hold; delegating avoids duplicating the Welch/pooled math.
            result = scipy_stats.ttest_ind_from_stats(
                mean_a,
                math.sqrt(var_a),
                n_a,
                mean_b,
                math.sqrt(var_b),
                n_b,
                equal_var=equal_var,
                alternative=alternative,
            )
            statistic = float(result.statistic)
            pvalue = float(result.pvalue)
            if equal_var:
                df = float(n_a + n_b - 2)
                pooled = (((n_a - 1) * var_a) + ((n_b - 1) * var_b)) / df if df > 0 else math.nan
                se = math.sqrt(pooled * ((1.0 / n_a) + (1.0 / n_b))) if not math.isnan(pooled) else math.nan
            else:
                se2_a = var_a / n_a
                se2_b = var_b / n_b
                se = math.sqrt(se2_a + se2_b)
                denominator = ((se2_a * se2_a) / (n_a - 1)) + ((se2_b * se2_b) / (n_b - 1))
                df = ((se2_a + se2_b) ** 2) / denominator if denominator else math.nan
            if se and not math.isnan(df):
                critical = float(scipy_stats.t.ppf(1.0 - (1.0 - confidence_level) / 2.0, df))
                diff = mean_a - mean_b
                interval = (diff - critical * se, diff + critical * se)
        return StatResult(
            statistic=statistic,
            pvalue=pvalue,
            confidence_interval=interval,
            estimate={
                "group_a": self._format_atom(group_a),
                "group_b": self._format_atom(group_b),
                "mean_a": mean_a,
                "mean_b": mean_b,
                "variance_a": var_a,
                "variance_b": var_b,
                "n_a": n_a,
                "n_b": n_b,
                "df": df,
                "equal_var": equal_var,
                "alternative": alternative,
            },
            n={"group_a": n_a, "group_b": n_b},
            method="pooled_two_sample_ttest" if equal_var else "welch_two_sample_ttest",
            exact=summary_a["exact"] and summary_b["exact"],
            queries=summary_a["queries"] + summary_b["queries"],
            warnings=warnings,
        )

    def _group_numeric_summaries(
        self,
        field: str,
        group_field: str,
        groups: Iterable[Any] | None = None,
        support: Iterable[Any] | None = None,
        where: FilterInput = None,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        group_values = self._category_values_for(group_field, groups)
        support_values = self._support_for(field, support)

        def summarize(value: Any) -> tuple[str, dict[str, Any]]:
            return self._format_atom(value), self._numeric_summary(
                field,
                support=support_values,
                where=self._group_filter(group_field, value, where),
            )

        pairs = self._parallel(group_values, summarize)
        return [label for label, _ in pairs], [summary for _, summary in pairs]

    def anova_oneway(
        self,
        field: str,
        group_field: str,
        *,
        groups: Iterable[Any] | None = None,
        support: Iterable[Any] | None = None,
        where: FilterInput = None,
    ) -> StatResult:
        labels, summaries = self._group_numeric_summaries(
            field, group_field, groups=groups, support=support, where=where
        )
        if len(summaries) < 2:
            raise ValueError("anova_oneway requires at least two groups")
        warnings = [warning for summary in summaries for warning in summary["warnings"]]
        ns = [summary["n"] for summary in summaries]
        means = [summary["mean"] for summary in summaries]
        variances = [summary["variance"] for summary in summaries]
        total_n = sum(ns)
        k = len(summaries)
        if total_n <= k or any(n <= 1 for n in ns):
            statistic = math.nan
            pvalue = math.nan
            df_between = k - 1
            df_within = total_n - k
            warnings.append("not enough observations for one-way ANOVA")
        else:
            grand_mean = sum(n * mean for n, mean in zip(ns, means)) / total_n
            ss_between = sum(n * ((mean - grand_mean) ** 2) for n, mean in zip(ns, means))
            ss_within = sum((n - 1) * variance for n, variance in zip(ns, variances))
            df_between = k - 1
            df_within = total_n - k
            ms_between = ss_between / df_between
            ms_within = ss_within / df_within
            statistic = ms_between / ms_within if ms_within else math.nan
            pvalue = (
                float(scipy_stats.f.sf(statistic, df_between, df_within)) if not math.isnan(statistic) else math.nan
            )
        return StatResult(
            statistic=statistic,
            pvalue=pvalue,
            estimate={
                "groups": labels,
                "n": dict(zip(labels, ns)),
                "means": dict(zip(labels, means)),
                "variances": dict(zip(labels, variances)),
                "df_between": df_between,
                "df_within": df_within,
            },
            n=total_n,
            method="one_way_anova",
            exact=all(summary["exact"] for summary in summaries),
            queries=sum(summary["queries"] for summary in summaries),
            warnings=warnings,
        )

    def welch_anova(
        self,
        field: str,
        group_field: str,
        *,
        groups: Iterable[Any] | None = None,
        support: Iterable[Any] | None = None,
        where: FilterInput = None,
    ) -> StatResult:
        labels, summaries = self._group_numeric_summaries(
            field, group_field, groups=groups, support=support, where=where
        )
        if len(summaries) < 2:
            raise ValueError("welch_anova requires at least two groups")
        warnings = [warning for summary in summaries for warning in summary["warnings"]]
        ns = [summary["n"] for summary in summaries]
        means = [summary["mean"] for summary in summaries]
        variances = [summary["variance"] for summary in summaries]
        k = len(summaries)
        total_n = sum(ns)
        if any(n <= 1 for n in ns) or any(variance <= 0 or math.isnan(variance) for variance in variances):
            statistic = math.nan
            pvalue = math.nan
            df1 = float(k - 1)
            df2 = math.nan
            warnings.append("not enough nonzero group variance for Welch ANOVA")
        else:
            weights = [n / variance for n, variance in zip(ns, variances)]
            weight_sum = sum(weights)
            weighted_mean = sum(weight * mean for weight, mean in zip(weights, means)) / weight_sum
            df1 = float(k - 1)
            numerator = sum(weight * ((mean - weighted_mean) ** 2) for weight, mean in zip(weights, means)) / df1
            correction_term = sum(((1.0 - (weight / weight_sum)) ** 2) / (n - 1) for weight, n in zip(weights, ns))
            statistic = numerator / (1.0 + (2.0 * (k - 2) / ((k * k) - 1.0)) * correction_term)
            df2 = ((k * k) - 1.0) / (3.0 * correction_term) if correction_term else math.inf
            pvalue = float(scipy_stats.f.sf(statistic, df1, df2)) if not math.isnan(statistic) else math.nan
        return StatResult(
            statistic=statistic,
            pvalue=pvalue,
            estimate={
                "groups": labels,
                "n": dict(zip(labels, ns)),
                "means": dict(zip(labels, means)),
                "variances": dict(zip(labels, variances)),
                "df1": df1,
                "df2": df2,
            },
            n=total_n,
            method="welch_anova",
            exact=all(summary["exact"] for summary in summaries),
            queries=sum(summary["queries"] for summary in summaries),
            warnings=warnings,
        )

    def _support_samples_for_group(
        self,
        field: str,
        group_field: str,
        group: Any,
        support: Iterable[Any] | None = None,
        where: FilterInput = None,
    ) -> tuple[list[Any], HistogramResult]:
        support_values = self._support_for(field, support)
        hist = self.histogram(field, support=support_values, where=self._group_filter(group_field, group, where))
        samples: list[Any] = []
        for value in support_values:
            samples.extend([value] * hist.counts[self._format_atom(value)])
        return samples, hist

    def ks_2samp(
        self,
        field: str,
        group_field: str,
        group_a: Any,
        group_b: Any,
        *,
        support: Iterable[Any] | None = None,
        where: FilterInput = None,
        alternative: str = "two-sided",
    ) -> StatResult:
        self._validate_alternative(alternative)
        support_values = self._support_for(field, support)
        samples_a, hist_a = self._support_samples_for_group(field, group_field, group_a, support_values, where)
        samples_b, hist_b = self._support_samples_for_group(field, group_field, group_b, support_values, where)
        warnings = [*hist_a.warnings, *hist_b.warnings]
        if not samples_a or not samples_b:
            statistic = math.nan
            pvalue = math.nan
            warnings.append("not enough observations for two-sample KS test")
        else:
            result = scipy_stats.ks_2samp(samples_a, samples_b, alternative=alternative, method="auto")
            statistic = float(result.statistic)
            pvalue = float(result.pvalue)
        return StatResult(
            statistic=statistic,
            pvalue=pvalue,
            estimate={
                "group_a": self._format_atom(group_a),
                "group_b": self._format_atom(group_b),
                "n_a": len(samples_a),
                "n_b": len(samples_b),
                "alternative": alternative,
            },
            n={"group_a": len(samples_a), "group_b": len(samples_b)},
            method="support_counts_ks_2samp",
            exact=True,
            queries=hist_a.queries + hist_b.queries,
            warnings=warnings,
        )

    def mannwhitneyu(
        self,
        field: str,
        group_field: str,
        group_a: Any,
        group_b: Any,
        *,
        support: Iterable[Any] | None = None,
        where: FilterInput = None,
        alternative: str = "two-sided",
    ) -> StatResult:
        self._validate_alternative(alternative)
        support_values = self._support_for(field, support)
        samples_a, hist_a = self._support_samples_for_group(field, group_field, group_a, support_values, where)
        samples_b, hist_b = self._support_samples_for_group(field, group_field, group_b, support_values, where)
        warnings = [*hist_a.warnings, *hist_b.warnings]
        if not samples_a or not samples_b:
            statistic = math.nan
            pvalue = math.nan
            warnings.append("not enough observations for Mann-Whitney U test")
        else:
            result = scipy_stats.mannwhitneyu(samples_a, samples_b, alternative=alternative, method="auto")
            statistic = float(result.statistic)
            pvalue = float(result.pvalue)
        return StatResult(
            statistic=statistic,
            pvalue=pvalue,
            estimate={
                "group_a": self._format_atom(group_a),
                "group_b": self._format_atom(group_b),
                "n_a": len(samples_a),
                "n_b": len(samples_b),
                "alternative": alternative,
            },
            n={"group_a": len(samples_a), "group_b": len(samples_b)},
            method="support_counts_mannwhitneyu",
            exact=True,
            queries=hist_a.queries + hist_b.queries,
            warnings=warnings,
        )

    def _indicator_specs(self, fields: dict[str, Iterable[Any]]) -> list[tuple[str, str, Any]]:
        specs: list[tuple[str, str, Any]] = []
        for field_name, values in fields.items():
            for value in values:
                label = f"{field_name}:{self._format_atom(value)}"
                specs.append((label, field_name, value))
        if not specs:
            raise ValueError("fields must contain at least one indicator value")
        return specs

    def onehot_covariance(
        self,
        fields: dict[str, Iterable[Any]],
        *,
        where: FilterInput = None,
    ) -> MatrixStatsResult:
        specs = self._indicator_specs(fields)
        n = self.count(where=where).statistic
        if n == 0:
            return MatrixStatsResult(
                fields=[label for label, _, _ in specs],
                matrix={label: {other: math.nan for other, _, _ in specs} for label, _, _ in specs},
                n=0,
                method="onehot_covariance",
                exact=True,
                queries=1,
                warnings=["no observations for one-hot covariance"],
            )

        def marginal(spec: tuple[str, str, Any]) -> tuple[str, int]:
            label, field_name, value = spec
            return label, self.count_eq(field_name, value, where=where).statistic

        marginal_counts = dict(self._parallel(specs, marginal))
        pair_specs = [(left, right) for left in specs for right in specs]

        def pair_count(pair: tuple[tuple[str, str, Any], tuple[str, str, Any]]) -> tuple[str, str, int]:
            left, right = pair
            left_label, left_field, left_value = left
            right_label, right_field, right_value = right
            if left_field == right_field and self._format_atom(left_value) == self._format_atom(right_value):
                count = marginal_counts[left_label]
            elif left_field == right_field:
                count = 0
            else:
                filters = [
                    *self._normalize_filters(where),
                    f"{left_field}:{self._format_atom(left_value)}",
                    f"{right_field}:{self._format_atom(right_value)}",
                ]
                count = self._count_only(filters)
            return left_label, right_label, count

        pairs = self._parallel(pair_specs, pair_count)
        labels = [label for label, _, _ in specs]
        matrix = {label: {} for label in labels}
        for left_label, right_label, count in pairs:
            p_left = marginal_counts[left_label] / n
            p_right = marginal_counts[right_label] / n
            matrix[left_label][right_label] = (count / n) - (p_left * p_right)
        return MatrixStatsResult(
            fields=labels,
            matrix=matrix,
            n=n,
            method="onehot_covariance",
            exact=True,
            queries=1 + len(specs) + sum(1 for left, right in pair_specs if left[1] != right[1]),
            metadata={"marginal_counts": marginal_counts},
        )

    def onehot_correlation(
        self,
        fields: dict[str, Iterable[Any]],
        *,
        where: FilterInput = None,
    ) -> MatrixStatsResult:
        covariance = self.onehot_covariance(fields, where=where)
        matrix: dict[str, dict[str, float]] = {label: {} for label in covariance.fields}
        for left in covariance.fields:
            var_left = covariance.matrix[left][left]
            for right in covariance.fields:
                var_right = covariance.matrix[right][right]
                denominator = math.sqrt(var_left * var_right) if var_left >= 0 and var_right >= 0 else math.nan
                matrix[left][right] = covariance.matrix[left][right] / denominator if denominator else math.nan
        return MatrixStatsResult(
            fields=covariance.fields,
            matrix=matrix,
            n=covariance.n,
            method="onehot_correlation",
            exact=covariance.exact,
            queries=covariance.queries,
            warnings=covariance.warnings,
            metadata=covariance.metadata,
        )

    def _binned_pair_table(
        self,
        x: str,
        y: str,
        x_bins: list[Any] | None = None,
        y_bins: list[Any] | None = None,
        where: FilterInput = None,
    ) -> tuple[list[tuple[Any, Any]], list[tuple[Any, Any]], dict[str, dict[str, int]], int, list[str]]:
        x_ranges = self._normalize_bins(x_bins or list(self._domain_for(x)))
        y_ranges = self._normalize_bins(y_bins or list(self._domain_for(y)))
        cells = [(xb, yb) for xb in x_ranges for yb in y_ranges]

        def count_cell(cell: tuple[tuple[Any, Any], tuple[Any, Any]]) -> tuple[str, str, int]:
            x_bin, y_bin = cell
            x_label = f"{self._format_atom(x_bin[0])}~{self._format_atom(x_bin[1])}"
            y_label = f"{self._format_atom(y_bin[0])}~{self._format_atom(y_bin[1])}"
            filters = [
                *self._normalize_filters(where),
                f"{x}:{x_label}",
                f"{y}:{y_label}",
            ]
            return x_label, y_label, self._count_only(filters)

        triples = self._parallel(cells, count_cell)
        table = {f"{self._format_atom(low)}~{self._format_atom(high)}": {} for low, high in x_ranges}
        warnings: list[str] = []
        for x_label, y_label, count in triples:
            table[x_label][y_label] = count
            warnings.extend(self._min_cell_warnings(f"{x}:{x_label},{y}:{y_label}", count))
        return x_ranges, y_ranges, table, len(cells), warnings

    def binned_pearson(
        self,
        x: str,
        y: str,
        *,
        x_bins: list[Any] | None = None,
        y_bins: list[Any] | None = None,
        where: FilterInput = None,
    ) -> StatResult:
        x_ranges, y_ranges, table, queries, warnings = self._binned_pair_table(x, y, x_bins, y_bins, where)
        x_midpoints = {
            f"{self._format_atom(low)}~{self._format_atom(high)}": self._bin_midpoint((low, high))
            for low, high in x_ranges
        }
        y_midpoints = {
            f"{self._format_atom(low)}~{self._format_atom(high)}": self._bin_midpoint((low, high))
            for low, high in y_ranges
        }
        n = sum(sum(row.values()) for row in table.values())
        if n == 0:
            statistic = math.nan
            warnings.append("no observations for binned Pearson correlation")
        else:
            sum_x = sum(x_midpoints[x_label] * sum(cols.values()) for x_label, cols in table.items())
            sum_y = sum(
                y_midpoints[y_label] * sum(table[x_label][y_label] for x_label in table) for y_label in y_midpoints
            )
            mean_x = sum_x / n
            mean_y = sum_y / n
            cov = 0.0
            var_x = 0.0
            var_y = 0.0
            for x_label, cols in table.items():
                dx = x_midpoints[x_label] - mean_x
                for y_label, count in cols.items():
                    dy = y_midpoints[y_label] - mean_y
                    cov += count * dx * dy
                    var_x += count * dx * dx
                    var_y += count * dy * dy
            statistic = cov / math.sqrt(var_x * var_y) if var_x > 0 and var_y > 0 else math.nan
        return StatResult(
            statistic=statistic,
            estimate={"counts": table, "x_midpoints": x_midpoints, "y_midpoints": y_midpoints},
            n=n,
            method="binned_pearson",
            exact=False,
            queries=queries,
            warnings=warnings,
        )

    def binned_spearman(
        self,
        x: str,
        y: str,
        *,
        x_bins: list[Any] | None = None,
        y_bins: list[Any] | None = None,
        where: FilterInput = None,
    ) -> StatResult:
        x_ranges, y_ranges, table, queries, warnings = self._binned_pair_table(x, y, x_bins, y_bins, where)
        x_labels = [f"{self._format_atom(low)}~{self._format_atom(high)}" for low, high in x_ranges]
        y_labels = [f"{self._format_atom(low)}~{self._format_atom(high)}" for low, high in y_ranges]
        n = sum(sum(row.values()) for row in table.values())
        if n == 0:
            statistic = math.nan
            warnings.append("no observations for binned Spearman correlation")
        else:
            x_counts = {label: sum(table[label].values()) for label in x_labels}
            y_counts = {label: sum(table[x_label][label] for x_label in x_labels) for label in y_labels}
            x_ranks: dict[str, float] = {}
            y_ranks: dict[str, float] = {}
            cumulative = 0
            for label in x_labels:
                x_ranks[label] = cumulative + ((x_counts[label] + 1) / 2.0)
                cumulative += x_counts[label]
            cumulative = 0
            for label in y_labels:
                y_ranks[label] = cumulative + ((y_counts[label] + 1) / 2.0)
                cumulative += y_counts[label]
            mean_x = sum(x_ranks[label] * x_counts[label] for label in x_labels) / n
            mean_y = sum(y_ranks[label] * y_counts[label] for label in y_labels) / n
            cov = var_x = var_y = 0.0
            for x_label in x_labels:
                dx = x_ranks[x_label] - mean_x
                for y_label in y_labels:
                    dy = y_ranks[y_label] - mean_y
                    count = table[x_label][y_label]
                    cov += count * dx * dy
                    var_x += count * dx * dx
                    var_y += count * dy * dy
            statistic = cov / math.sqrt(var_x * var_y) if var_x > 0 and var_y > 0 else math.nan
        return StatResult(
            statistic=statistic,
            estimate={"counts": table},
            n=n,
            method="binned_spearman",
            exact=False,
            queries=queries,
            warnings=warnings,
        )

    def point_biserial(
        self,
        field: str,
        group_field: str,
        positive_value: Any,
        *,
        negative_value: Any | None = None,
        support: Iterable[Any] | None = None,
        where: FilterInput = None,
        alternative: str = "two-sided",
    ) -> StatResult:
        self._validate_alternative(alternative)
        negative_value = self._other_category_value(group_field, positive_value, negative_value)
        support_values = self._support_for(field, support)
        pos = self._numeric_summary(
            field, support=support_values, where=self._group_filter(group_field, positive_value, where)
        )
        neg = self._numeric_summary(
            field, support=support_values, where=self._group_filter(group_field, negative_value, where)
        )
        total = self._numeric_summary(field, support=support_values, where=where)
        n_pos = pos["n"]
        n_neg = neg["n"]
        n = n_pos + n_neg
        warnings = [*pos["warnings"], *neg["warnings"], *total["warnings"]]
        if n <= 2 or total["variance"] <= 0 or math.isnan(total["variance"]):
            statistic = pvalue = math.nan
            warnings.append("not enough variance for point-biserial correlation")
        else:
            statistic = ((pos["mean"] - neg["mean"]) / math.sqrt(total["variance"])) * math.sqrt(
                (n_pos * n_neg) / (n * (n - 1))
            )
            t_stat = statistic * math.sqrt((n - 2) / max(1e-300, 1.0 - statistic * statistic))
            pvalue = self._t_pvalue(t_stat, n - 2, alternative)
        return StatResult(
            statistic=statistic,
            pvalue=pvalue,
            estimate={
                "positive_value": self._format_atom(positive_value),
                "negative_value": self._format_atom(negative_value),
                "mean_positive": pos["mean"],
                "mean_negative": neg["mean"],
                "n_positive": n_pos,
                "n_negative": n_neg,
            },
            n=n,
            method="point_biserial",
            exact=pos["exact"] and neg["exact"] and total["exact"],
            queries=pos["queries"] + neg["queries"] + total["queries"],
            warnings=warnings,
        )

    def eta_squared(
        self,
        field: str,
        group_field: str,
        *,
        groups: Iterable[Any] | None = None,
        support: Iterable[Any] | None = None,
        where: FilterInput = None,
    ) -> StatResult:
        labels, summaries = self._group_numeric_summaries(
            field, group_field, groups=groups, support=support, where=where
        )
        total_n = sum(summary["n"] for summary in summaries)
        warnings = [warning for summary in summaries for warning in summary["warnings"]]
        if total_n == 0:
            statistic = math.nan
            warnings.append("no observations for eta squared")
        else:
            grand_mean = sum(summary["n"] * summary["mean"] for summary in summaries) / total_n
            ss_between = sum(summary["n"] * ((summary["mean"] - grand_mean) ** 2) for summary in summaries)
            ss_within = sum((summary["n"] - 1) * summary["variance"] for summary in summaries if summary["n"] > 1)
            statistic = ss_between / (ss_between + ss_within) if (ss_between + ss_within) else math.nan
        return StatResult(
            statistic=statistic,
            estimate={
                "groups": labels,
                "n": dict(zip(labels, [summary["n"] for summary in summaries])),
                "means": dict(zip(labels, [summary["mean"] for summary in summaries])),
            },
            n=total_n,
            method="eta_squared",
            exact=all(summary["exact"] for summary in summaries),
            queries=sum(summary["queries"] for summary in summaries),
            warnings=warnings,
        )

    def missingness_rate(
        self,
        field: str,
        missing_values: Iterable[Any],
        *,
        where: FilterInput = None,
    ) -> StatResult:
        values = list(missing_values)
        denominator = self.count(where=where).statistic

        def count_missing(value: Any) -> tuple[str, int]:
            label = self._format_atom(value)
            return label, self.count_eq(field, value, where=where).statistic

        pairs = self._parallel(values, count_missing)
        counts = dict(pairs)
        missing_count = sum(counts.values())
        rate = missing_count / denominator if denominator else math.nan
        warnings = [] if denominator else ["denominator count is zero"]
        return StatResult(
            statistic=rate,
            estimate={"missing_count": missing_count, "total": denominator, "counts": counts},
            n=denominator,
            method="missingness_rate",
            exact=True,
            queries=1 + len(values),
            warnings=warnings,
        )

    def domain_violation_count(
        self,
        field: str,
        *,
        domain: Domain | None = None,
        values: Iterable[Any] | None = None,
        where: FilterInput = None,
    ) -> StatResult:
        if domain is None and values is None:
            configured = self.field_domains.get(field)
            if isinstance(configured, tuple) and len(configured) == 2:
                domain = configured
            elif configured is not None:
                values = configured
            else:
                raise ValueError("domain_violation_count requires domain=..., values=..., or field_domains[field]")
        total = self.count(where=where).statistic
        if domain is not None:
            valid = self.count_range(field, domain[0], domain[1], where=where).statistic
            method = "domain_range_violation_count"
            queries = 2
        else:
            value_list = list(values or [])

            def count_value(value: Any) -> int:
                return self.count_eq(field, value, where=where).statistic

            valid = sum(self._parallel(value_list, count_value))
            method = "domain_values_violation_count"
            queries = 1 + len(value_list)
        violations = max(0, total - valid)
        return StatResult(
            statistic=violations,
            estimate={"valid_count": valid, "total": total},
            n=total,
            method=method,
            exact=True,
            queries=queries,
        )

    def zscore_outlier_count(
        self,
        field: str,
        *,
        z: float = 3.0,
        support: Iterable[Any] | None = None,
        where: FilterInput = None,
    ) -> StatResult:
        if z <= 0:
            raise ValueError("z must be positive")
        support_values = self._support_for(field, support)
        summary = self._numeric_summary(field, support=support_values, where=where)
        mean = summary["mean"]
        std = math.sqrt(summary["variance"]) if not math.isnan(summary["variance"]) else math.nan
        lower = mean - z * std
        upper = mean + z * std
        hist = self.histogram(field, support=support_values, where=where)
        count = sum(
            bucket_count
            for raw_value, bucket_count in hist.counts.items()
            if bucket_count is not None and (float(raw_value) < lower or float(raw_value) > upper)
        )
        return StatResult(
            statistic=count,
            estimate={"mean": mean, "std": std, "lower": lower, "upper": upper},
            n=hist.n,
            method="zscore_outlier_count",
            exact=summary["exact"] and hist.exact,
            queries=summary["queries"] + hist.queries,
            warnings=[*summary["warnings"], *hist.warnings],
        )

    def iqr_outlier_count(
        self,
        field: str,
        *,
        support: Iterable[Any] | None = None,
        where: FilterInput = None,
    ) -> StatResult:
        support_values = self._support_for(field, support)
        q25 = self.quantile(field, 0.25, support=support_values, where=where)
        q75 = self.quantile(field, 0.75, support=support_values, where=where)
        iqr = q75.statistic - q25.statistic if q25.statistic is not None and q75.statistic is not None else math.nan
        lower = q25.statistic - 1.5 * iqr if not math.isnan(iqr) else math.nan
        upper = q75.statistic + 1.5 * iqr if not math.isnan(iqr) else math.nan
        hist = self.histogram(field, support=support_values, where=where)
        count = sum(
            bucket_count
            for raw_value, bucket_count in hist.counts.items()
            if bucket_count is not None and (float(raw_value) < lower or float(raw_value) > upper)
        )
        return StatResult(
            statistic=count,
            estimate={"q25": q25.statistic, "q75": q75.statistic, "iqr": iqr, "lower": lower, "upper": upper},
            n=hist.n,
            method="iqr_outlier_count",
            exact=q25.exact and q75.exact and hist.exact,
            queries=q25.queries + q75.queries + hist.queries,
            warnings=[*q25.warnings, *q75.warnings, *hist.warnings],
        )

    def distribution(
        self,
        field: str,
        *,
        bins: list[Any] | None = None,
        values: Iterable[Any] | None = None,
        where: FilterInput = None,
        suppression_policy: str = "warn",
    ) -> StatResult:
        if bins is not None:
            hist = self.histogram(field, bins=bins, where=where, suppression_policy=suppression_policy)
            counts = hist.counts
            n = hist.n
            proportions = {
                label: (count / n if count is not None and n else math.nan) for label, count in counts.items()
            }
            return StatResult(
                statistic={"counts": counts, "proportions": proportions},
                estimate={"bins": hist.bins},
                n=n,
                method="histogram_distribution",
                exact=hist.exact,
                queries=hist.queries,
                warnings=hist.warnings,
            )
        if values is None:
            domain = self.field_domains.get(field)
            if isinstance(domain, tuple) and len(domain) == 2 and all(isinstance(v, int | float) for v in domain):
                return self.distribution(
                    field,
                    bins=list(domain),
                    where=where,
                    suppression_policy=suppression_policy,
                )
            values = self._category_values_for(field)
        freq = self.frequency(field, values=values, where=where, suppression_policy=suppression_policy)
        return StatResult(
            statistic={"counts": freq.counts, "proportions": freq.proportions},
            estimate={"values": list(values)},
            n=freq.n,
            method="frequency_distribution",
            exact=freq.exact,
            queries=freq.queries,
            warnings=freq.warnings,
        )

    @staticmethod
    def _aligned_proportions(
        left: StatResult, right: StatResult, epsilon: float
    ) -> tuple[list[str], list[float], list[float]]:
        left_props = left.statistic["proportions"]
        right_props = right.statistic["proportions"]
        labels = list(dict.fromkeys([*left_props.keys(), *right_props.keys()]))
        left_values = [max(float(left_props.get(label, 0.0) or 0.0), epsilon) for label in labels]
        right_values = [max(float(right_props.get(label, 0.0) or 0.0), epsilon) for label in labels]
        left_total = sum(left_values)
        right_total = sum(right_values)
        return labels, [value / left_total for value in left_values], [value / right_total for value in right_values]

    def population_stability_index(
        self,
        field: str,
        baseline_filter: FilterInput,
        current_filter: FilterInput,
        *,
        bins: list[Any] | None = None,
        values: Iterable[Any] | None = None,
        epsilon: float = 1e-12,
    ) -> StatResult:
        baseline = self.distribution(field, bins=bins, values=values, where=baseline_filter)
        current = self.distribution(field, bins=bins, values=values, where=current_filter)
        labels, baseline_props, current_props = self._aligned_proportions(baseline, current, epsilon)
        psi = sum(
            (current_prop - baseline_prop) * self._safe_log_ratio(current_prop, baseline_prop)
            for baseline_prop, current_prop in zip(baseline_props, current_props)
        )
        return StatResult(
            statistic=psi,
            estimate={"labels": labels, "baseline": baseline_props, "current": current_props},
            n={"baseline": baseline.n, "current": current.n},
            method="population_stability_index",
            exact=baseline.exact and current.exact,
            queries=baseline.queries + current.queries,
            warnings=[*baseline.warnings, *current.warnings],
        )

    def distribution_divergence(
        self,
        field: str,
        baseline_filter: FilterInput,
        current_filter: FilterInput,
        *,
        metric: str = "js",
        bins: list[Any] | None = None,
        values: Iterable[Any] | None = None,
        epsilon: float = 1e-12,
    ) -> StatResult:
        metric = metric.lower()
        if metric not in {"kl", "js"}:
            raise ValueError("metric must be either 'kl' or 'js'")
        baseline = self.distribution(field, bins=bins, values=values, where=baseline_filter)
        current = self.distribution(field, bins=bins, values=values, where=current_filter)
        labels, baseline_props, current_props = self._aligned_proportions(baseline, current, epsilon)
        if metric == "kl":
            value = sum(b * self._safe_log_ratio(b, c) for b, c in zip(baseline_props, current_props))
        else:
            midpoint = [(b + c) / 2.0 for b, c in zip(baseline_props, current_props)]
            value = 0.5 * sum(b * self._safe_log_ratio(b, m) for b, m in zip(baseline_props, midpoint))
            value += 0.5 * sum(c * self._safe_log_ratio(c, m) for c, m in zip(current_props, midpoint))
        return StatResult(
            statistic=value,
            estimate={"labels": labels, "baseline": baseline_props, "current": current_props},
            n={"baseline": baseline.n, "current": current.n},
            method=f"{metric}_distribution_divergence",
            exact=baseline.exact and current.exact,
            queries=baseline.queries + current.queries,
            warnings=[*baseline.warnings, *current.warnings],
        )

    def drift_chi2(
        self,
        field: str,
        *,
        cohort_field: str | None = None,
        cohorts: Iterable[Any] | None = None,
        baseline_filter: FilterInput = None,
        current_filter: FilterInput = None,
        bins: list[Any] | None = None,
        values: Iterable[Any] | None = None,
    ) -> StatResult:
        if cohort_field is not None:
            if cohorts is None:
                raise ValueError("cohorts are required when cohort_field is provided")
            cohort_values = list(cohorts)
            bucket_values = list(values) if values is not None else self._category_values_for(field)
            rows: list[list[int]] = []
            counts: dict[str, dict[str, int]] = {}
            queries = 0
            for cohort in cohort_values:
                label = self._format_atom(cohort)
                counts[label] = {}
                row = []
                for bucket in bucket_values:
                    bucket_label = self._format_atom(bucket)
                    count = self._count_only([f"{cohort_field}:{label}", f"{field}:{bucket_label}"])
                    counts[label][bucket_label] = count
                    row.append(count)
                    queries += 1
                rows.append(row)
        else:
            if baseline_filter is None or current_filter is None:
                raise ValueError("baseline_filter and current_filter are required when cohort_field is not provided")
            baseline = self.distribution(field, bins=bins, values=values, where=baseline_filter)
            current = self.distribution(field, bins=bins, values=values, where=current_filter)
            labels = list(baseline.statistic["counts"])
            rows = [
                [int(baseline.statistic["counts"].get(label) or 0) for label in labels],
                [int(current.statistic["counts"].get(label) or 0) for label in labels],
            ]
            counts = {"baseline": dict(zip(labels, rows[0])), "current": dict(zip(labels, rows[1]))}
            queries = baseline.queries + current.queries
        if not rows or any(sum(row) == 0 for row in rows):
            statistic = pvalue = math.nan
            dof = 0
            expected_dict: dict[str, Any] = {}
            warnings = ["not enough observations for drift chi-square test"]
        else:
            statistic, pvalue, dof, expected = scipy_stats.chi2_contingency(rows, correction=False)
            expected_dict = {
                row_label: {
                    col_label: float(expected[row_idx][col_idx])
                    for col_idx, col_label in enumerate(next(iter(counts.values())).keys())
                }
                for row_idx, row_label in enumerate(counts)
            }
            warnings = []
        return StatResult(
            statistic=float(statistic),
            pvalue=float(pvalue),
            estimate={"observed": counts, "expected": expected_dict, "dof": int(dof)},
            n=sum(sum(row) for row in rows),
            method="drift_chi2",
            exact=True,
            queries=queries,
            warnings=warnings,
        )

    def period_counts(
        self,
        period_field: str,
        periods: Iterable[Any],
        *,
        where: FilterInput = None,
    ) -> GroupedStatsResult:
        return self.groupby_count(period_field, values=periods, where=where)

    def period_summary(
        self,
        field: str,
        period_field: str,
        periods: Iterable[Any],
        *,
        agg: str = "mean",
        support: Iterable[Any] | None = None,
        where: FilterInput = None,
    ) -> GroupedStatsResult:
        if agg == "count":
            return self.groupby_count(period_field, values=periods, where=where)
        if agg == "mean":
            return self.groupby_mean(field, period_field, values=periods, where=where)
        if agg == "var":
            return self.groupby_var(field, period_field, values=periods, support=support, where=where)
        if agg == "std":
            return self.groupby_std(field, period_field, values=periods, support=support, where=where)
        period_values = list(periods)

        def summarize(period: Any) -> tuple[str, StatResult]:
            filters = self._group_filter(period_field, period, where)
            if agg == "sum":
                return self._format_atom(period), self.sum_(field, where=filters)
            if agg == "min":
                return self._format_atom(period), self.min_(field, where=filters)
            if agg == "max":
                return self._format_atom(period), self.max_(field, where=filters)
            raise ValueError("agg must be one of: count, mean, sum, min, max, var, std")

        pairs = self._parallel(period_values, summarize)
        statistics = {label: {agg: result.statistic, "count": result.n} for label, result in pairs}
        return GroupedStatsResult(
            field=field,
            group_field=period_field,
            statistics=statistics,
            n={label: int(result.n or 0) for label, result in pairs},
            method=f"period_{agg}",
            exact=all(result.exact for _, result in pairs),
            queries=sum(result.queries for _, result in pairs),
            warnings=[warning for _, result in pairs for warning in result.warnings],
        )

    def feature_screening(
        self,
        *,
        numeric_fields: Iterable[str] | None = None,
        categorical_fields: Iterable[str] | None = None,
        target_field: str | None = None,
        target_positive: Any | None = None,
        supports: dict[str, Iterable[Any]] | None = None,
        values: dict[str, Iterable[Any]] | None = None,
        missing_values: dict[str, Iterable[Any]] | None = None,
        where: FilterInput = None,
    ) -> FeatureScreeningResult:
        supports = supports or {}
        values = values or {}
        missing_values = missing_values or {}
        report: dict[str, Any] = {"numeric": {}, "categorical": {}, "missingness": {}}
        queries = 0
        exact = True
        warnings: list[str] = []
        for field_name in numeric_fields or []:
            desc = self.describe(field_name, support=supports.get(field_name), where=where)
            report["numeric"][field_name] = dict(desc.statistics)
            queries += desc.queries
            exact = exact and all(desc.exact.values())
            warnings.extend(desc.warnings)
            if target_field is not None and target_positive is not None:
                try:
                    pb = self.point_biserial(
                        field_name, target_field, target_positive, support=supports.get(field_name), where=where
                    )
                    report["numeric"][field_name]["point_biserial"] = pb.statistic
                    queries += pb.queries
                    exact = exact and pb.exact
                    warnings.extend(pb.warnings)
                except ValueError as exc:
                    warnings.append(f"{field_name}: {exc}")
        for field_name in categorical_fields or []:
            freq = self.frequency(field_name, values=values.get(field_name), where=where)
            report["categorical"][field_name] = {
                "counts": freq.counts,
                "proportions": freq.proportions,
                "mode": freq.mode,
            }
            queries += freq.queries
            exact = exact and freq.exact
            warnings.extend(freq.warnings)
            if target_field is not None and field_name != target_field:
                try:
                    chi2 = self.chi2_independence(
                        field_name, target_field, row_values=values.get(field_name), where=where
                    )
                    report["categorical"][field_name]["chi2"] = chi2.statistic
                    report["categorical"][field_name]["chi2_pvalue"] = chi2.pvalue
                    queries += chi2.queries
                    exact = exact and chi2.exact
                    warnings.extend(chi2.warnings)
                except ValueError as exc:
                    warnings.append(f"{field_name}: {exc}")
        for field_name, missing in missing_values.items():
            result = self.missingness_rate(field_name, missing, where=where)
            report["missingness"][field_name] = result.estimate
            report["missingness"][field_name]["rate"] = result.statistic
            queries += result.queries
            exact = exact and result.exact
            warnings.extend(result.warnings)
        return FeatureScreeningResult(
            statistics=report,
            method="feature_screening",
            exact=exact,
            queries=queries,
            warnings=warnings,
        )

    def data_quality_report(
        self,
        fields: Iterable[str],
        *,
        missing_values: dict[str, Iterable[Any]] | None = None,
        domains: dict[str, Domain] | None = None,
        values: dict[str, Iterable[Any]] | None = None,
        outlier_fields: Iterable[str] | None = None,
        distribution_bins: dict[str, list[Any]] | None = None,
        where: FilterInput = None,
        suppression_policy: str = "warn",
    ) -> FeatureScreeningResult:
        self._validate_suppression_policy(suppression_policy)
        missing_values = missing_values or {}
        domains = domains or {}
        values = values or {}
        outlier_set = set(outlier_fields or [])
        distribution_bins = distribution_bins or {}
        report: dict[str, Any] = {}
        queries = 0
        exact = True
        warnings: list[str] = []
        for field_name in fields:
            field_report: dict[str, Any] = {}
            if field_name in missing_values:
                result = self.missingness_rate(field_name, missing_values[field_name], where=where)
                field_report["missingness"] = {"rate": result.statistic, **result.estimate}
                queries += result.queries
                exact = exact and result.exact
                warnings.extend(result.warnings)
            if field_name in domains or field_name in values or field_name in self.field_domains:
                try:
                    result = self.domain_violation_count(
                        field_name,
                        domain=domains.get(field_name),
                        values=values.get(field_name),
                        where=where,
                    )
                    field_report["domain_violations"] = result.statistic
                    queries += result.queries
                    exact = exact and result.exact
                    warnings.extend(result.warnings)
                except ValueError as exc:
                    warnings.append(f"{field_name}: {exc}")
            if field_name in outlier_set:
                result = self.iqr_outlier_count(field_name, where=where)
                field_report["iqr_outliers"] = result.statistic
                field_report["iqr_outlier_bounds"] = result.estimate
                queries += result.queries
                exact = exact and result.exact
                warnings.extend(result.warnings)
            if field_name in distribution_bins or field_name in values or field_name in self.field_domains:
                try:
                    result = self.distribution(
                        field_name,
                        bins=distribution_bins.get(field_name),
                        values=values.get(field_name),
                        where=where,
                        suppression_policy=suppression_policy,
                    )
                    field_report["distribution"] = result.statistic
                    queries += result.queries
                    exact = exact and result.exact
                    warnings.extend(result.warnings)
                except ValueError as exc:
                    warnings.append(f"{field_name}: {exc}")
            report[field_name] = field_report
        return FeatureScreeningResult(
            statistics=report,
            method="data_quality_report",
            exact=exact,
            queries=queries,
            warnings=warnings,
            metadata={"suppression_policy": suppression_policy},
        )

    def describe(
        self,
        field: str,
        *,
        support: Iterable[Any] | None = None,
        values: Iterable[Any] | None = None,
        where: FilterInput = None,
    ) -> DescribeResult:
        support_values_input = list(support) if support is not None else None
        stats: dict[str, Any] = {}
        exact: dict[str, bool] = {}
        warnings: list[str] = []
        queries = 0

        numeric_domain: Domain | None = None
        try:
            numeric_domain = (
                (min(support_values_input), max(support_values_input))
                if support_values_input is not None
                else self._domain_for(field)
            )
        except ValueError as exc:
            warnings.append(str(exc))

        if numeric_domain is not None:
            for name, fn in (
                ("count", lambda: self.count_range(field, numeric_domain[0], numeric_domain[1], where=where)),
                ("mean", lambda: self.mean(field, domain=numeric_domain, where=where)),
                ("min", lambda: self.min_(field, domain=numeric_domain, where=where)),
                ("max", lambda: self.max_(field, domain=numeric_domain, where=where)),
                ("range", lambda: self.range_(field, domain=numeric_domain, where=where)),
            ):
                result = fn()
                stats[name] = result.statistic
                exact[name] = result.exact
                queries += result.queries

        if support_values_input is not None or field in self.field_domains:
            try:
                support_values = self._support_for(field, support_values_input)
                for name, fn in (
                    ("variance", lambda: self.var(field, support=support_values, where=where)),
                    ("std", lambda: self.std(field, support=support_values, where=where)),
                    ("q25", lambda: self.quantile(field, 0.25, support=support_values, where=where)),
                    ("median", lambda: self.median(field, support=support_values, where=where)),
                    ("q75", lambda: self.quantile(field, 0.75, support=support_values, where=where)),
                    ("iqr", lambda: self.iqr(field, support=support_values, where=where)),
                ):
                    result = fn()
                    stats[name] = result.statistic
                    exact[name] = result.exact
                    queries += result.queries
                    warnings.extend(result.warnings)
            except ValueError as exc:
                warnings.append(str(exc))

        if values is not None:
            freq = self.frequency(field, values=values, where=where)
            stats["frequency"] = freq.counts
            stats["mode"] = freq.mode
            exact["frequency"] = freq.exact
            exact["mode"] = freq.exact
            queries += freq.queries
            warnings.extend(freq.warnings)

        return DescribeResult(
            field=field,
            statistics=stats,
            exact=exact,
            n=int(stats["count"]) if "count" in stats else None,
            method="bi_describe",
            queries=queries,
            warnings=warnings,
        )


class BILinearRegression:
    """Linear/ridge regression summary from BI aggregate sufficient statistics."""

    def __init__(self) -> None:
        self.result_: RegressionSummaryResult | None = None
        self.coef_: dict[str, float] = {}
        self.intercept_: float | None = None

    @staticmethod
    def _moment_lookup(moment_fields: dict[Any, str], left: str, right: str) -> str | None:
        candidates: list[Any] = [
            (left, right),
            (right, left),
            f"{left}*{right}",
            f"{right}*{left}",
            f"{left}_times_{right}",
            f"{right}_times_{left}",
            f"{left}_x_{right}",
            f"{right}_x_{left}",
        ]
        if left == right:
            candidates.extend([f"{left}_squared", f"{left}_sq"])
        for candidate in candidates:
            if candidate in moment_fields:
                return moment_fields[candidate]
        return None

    @staticmethod
    def _require_moment(moment_fields: dict[Any, str], left: str, right: str) -> str:
        field = BILinearRegression._moment_lookup(moment_fields, left, right)
        if field is None:
            raise ValueError(
                f"missing derived moment field for {left!r} * {right!r}; "
                "pass moment_fields with tuple keys like (left, right)"
            )
        return field

    @staticmethod
    def _field_sum(stats: BIStatsSession, field: str) -> tuple[float, int]:
        result = stats.sum_(field)
        return float(result.statistic), int(result.queries)

    def fit(
        self,
        stats: BIStatsSession,
        *,
        y: str,
        x: Iterable[str],
        moment_fields: dict[Any, str] | None = None,
        fit_intercept: bool = True,
        alpha: float = 0.0,
        confidence_level: float = 0.95,
    ) -> BILinearRegression:
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if confidence_level <= 0 or confidence_level >= 1:
            raise ValueError("confidence_level must be between 0 and 1")
        x_fields = list(x)
        if not x_fields:
            raise ValueError("x must contain at least one predictor")
        moment_fields = moment_fields or {}
        feature_names = ["intercept", *x_fields] if fit_intercept else list(x_fields)
        p = len(feature_names)
        n_result = stats.count()
        n = int(n_result.statistic)
        queries = n_result.queries
        warnings: list[str] = []
        if n <= p:
            warnings.append("observations do not exceed parameter count; inference will be undefined")

        sum_y, q = self._field_sum(stats, y)
        queries += q
        raw_x_sums: dict[str, float] = {}
        for x_field in x_fields:
            value, q = self._field_sum(stats, x_field)
            raw_x_sums[x_field] = value
            queries += q

        x_tx = np.zeros((p, p), dtype=float)
        x_ty = np.zeros(p, dtype=float)
        if fit_intercept:
            x_tx[0, 0] = n
            x_ty[0] = sum_y
            for idx, field in enumerate(x_fields, start=1):
                x_tx[0, idx] = raw_x_sums[field]
                x_tx[idx, 0] = raw_x_sums[field]
        for i, left in enumerate(x_fields):
            row = i + 1 if fit_intercept else i
            product_field = self._require_moment(moment_fields, left, y)
            value, q = self._field_sum(stats, product_field)
            x_ty[row] = value
            queries += q
            for j, right in enumerate(x_fields):
                col = j + 1 if fit_intercept else j
                product_field = self._require_moment(moment_fields, left, right)
                value, q = self._field_sum(stats, product_field)
                x_tx[row, col] = value
                queries += q
        y_squared_field = self._require_moment(moment_fields, y, y)
        y_ty, q = self._field_sum(stats, y_squared_field)
        queries += q

        ridge = np.eye(p, dtype=float) * alpha
        if fit_intercept:
            ridge[0, 0] = 0.0
        system = x_tx + ridge
        try:
            inv_system = np.linalg.inv(system)
        except np.linalg.LinAlgError:
            inv_system = np.linalg.pinv(system)
            warnings.append("used pseudo-inverse because aggregate moment matrix is singular")
        beta = inv_system @ x_ty
        # Residual SSE uses the unregularized sufficient statistics.
        sse = float(y_ty - (2.0 * beta.T @ x_ty) + (beta.T @ x_tx @ beta))
        sse = max(0.0, sse)
        df_resid = n - p
        residual_variance = sse / df_resid if df_resid > 0 else math.nan
        tss = y_ty - ((sum_y * sum_y) / n) if n else math.nan
        r2 = 1.0 - (sse / tss) if tss and not math.isnan(tss) else math.nan
        covariance = (
            inv_system * residual_variance if not math.isnan(residual_variance) else np.full_like(inv_system, math.nan)
        )
        alpha_tail = 1.0 - confidence_level
        critical = float(scipy_stats.t.ppf(1.0 - alpha_tail / 2.0, df_resid)) if df_resid > 0 else math.nan

        coefficients = {name: float(beta[idx]) for idx, name in enumerate(feature_names)}
        standard_errors: dict[str, float] = {}
        tvalues: dict[str, float] = {}
        pvalues: dict[str, float] = {}
        intervals: dict[str, tuple[float, float]] = {}
        for idx, name in enumerate(feature_names):
            se = math.sqrt(covariance[idx, idx]) if covariance[idx, idx] >= 0 else math.nan
            tvalue = coefficients[name] / se if se else math.nan
            standard_errors[name] = se
            tvalues[name] = tvalue
            pvalues[name] = (
                float(2.0 * scipy_stats.t.sf(abs(tvalue), df_resid))
                if df_resid > 0 and not math.isnan(tvalue)
                else math.nan
            )
            intervals[name] = (
                (
                    coefficients[name] - critical * se,
                    coefficients[name] + critical * se,
                )
                if not math.isnan(critical) and not math.isnan(se)
                else (math.nan, math.nan)
            )

        result = RegressionSummaryResult(
            coefficients=coefficients,
            standard_errors=standard_errors,
            tvalues=tvalues,
            pvalues=pvalues,
            confidence_intervals=intervals,
            r2=float(r2),
            residual_variance=float(residual_variance),
            df_resid=int(df_resid),
            n=n,
            method="bi_linear_regression" if alpha == 0 else "bi_ridge_regression",
            exact=True,
            queries=queries,
            warnings=warnings,
            metadata={
                "x": x_fields,
                "y": y,
                "fit_intercept": fit_intercept,
                "alpha": alpha,
                "x_tx": x_tx.tolist(),
                "x_ty": x_ty.tolist(),
            },
        )
        self.result_ = result
        self.coef_ = {name: value for name, value in coefficients.items() if name != "intercept"}
        self.intercept_ = coefficients.get("intercept")
        return self

    def summary(self) -> RegressionSummaryResult:
        if self.result_ is None:
            raise ValueError("model is not fitted")
        return self.result_


__all__ = [
    "BILinearRegression",
    "BIStatsSession",
    "CrosstabResult",
    "DescribeResult",
    "FeatureScreeningResult",
    "FrequencyResult",
    "GroupedStatsResult",
    "HistogramResult",
    "MatrixStatsResult",
    "RegressionSummaryResult",
    "StatResult",
]
