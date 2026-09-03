"""Blind Insight HTTP client — the only module that talks to the Blind Proxy.

`BlindInsightClient` exposes the encrypted query surface that `blind_stats.stats`
builds on: `aggregate` for numeric aggregate functions and `query(count_only=True)`
for counts. Neither path decrypts records.

`ProfilingStats` (the module-level `profiling` instance) records per-query timing
so notebooks can report round-trip cost.
"""

import os
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests


@dataclass
class QueryTiming:
    """Timing data for a single query."""

    query_type: str  # 'query' or 'aggregate'
    filter_str: str  # The filter/agg_filter used
    start_time: float
    end_time: float
    network_ms: float  # Time for HTTP request
    parse_ms: float  # Time to parse JSON response
    total_ms: float  # Total time
    success: bool
    error: str | None = None


class ProfilingStats:
    """Collects and reports profiling statistics."""

    def __init__(self):
        self.enabled = False
        self.timings: list[QueryTiming] = []
        self.session_start: float | None = None

    def enable(self):
        """Enable profiling and reset stats."""
        self.enabled = True
        self.timings = []
        self.session_start = time.time()

    def disable(self):
        """Disable profiling."""
        self.enabled = False

    def record(self, timing: QueryTiming):
        """Record a query timing."""
        if self.enabled:
            self.timings.append(timing)

    def summary(self) -> dict[str, Any]:
        """Generate summary statistics."""
        if not self.timings:
            return {"error": "No timings recorded"}

        query_times = [t for t in self.timings if t.query_type == "query"]
        agg_times = [t for t in self.timings if t.query_type == "aggregate"]

        def stats(timings: list[QueryTiming], field: str) -> dict[str, float]:
            if not timings:
                return {"count": 0, "min": 0, "max": 0, "avg": 0, "total": 0}
            values = [getattr(t, field) for t in timings]
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "total": sum(values),
            }

        return {
            "session_duration_sec": time.time() - self.session_start if self.session_start else 0,
            "total_queries": len(self.timings),
            "successful_queries": sum(1 for t in self.timings if t.success),
            "failed_queries": sum(1 for t in self.timings if not t.success),
            "query_timing_ms": stats(query_times, "total_ms"),
            "aggregate_timing_ms": stats(agg_times, "total_ms"),
            "network_ms": stats(self.timings, "network_ms"),
            "parse_ms": stats(self.timings, "parse_ms"),
            "total_ms": stats(self.timings, "total_ms"),
        }

    def breakdown(self) -> dict[str, float]:
        """Get time breakdown by category."""
        if not self.timings:
            return {}

        total_network = sum(t.network_ms for t in self.timings)
        total_parse = sum(t.parse_ms for t in self.timings)
        total_time = sum(t.total_ms for t in self.timings)

        return {
            "network_ms": total_network,
            "network_pct": (total_network / total_time * 100) if total_time > 0 else 0,
            "parse_ms": total_parse,
            "parse_pct": (total_parse / total_time * 100) if total_time > 0 else 0,
            "overhead_ms": total_time - total_network - total_parse,
            "overhead_pct": ((total_time - total_network - total_parse) / total_time * 100) if total_time > 0 else 0,
            "total_ms": total_time,
        }

    def per_query_breakdown(self) -> list[dict[str, Any]]:
        """Get per-query timing details."""
        return [
            {
                "type": t.query_type,
                "filter": t.filter_str[:50] + "..." if len(t.filter_str) > 50 else t.filter_str,
                "network_ms": round(t.network_ms, 2),
                "parse_ms": round(t.parse_ms, 2),
                "total_ms": round(t.total_ms, 2),
                "success": t.success,
            }
            for t in self.timings
        ]


# Global profiling instance
profiling = ProfilingStats()


class BlindInsightClient:
    """
    Client for querying data from Blind Insight via the Blind Proxy HTTP API.

    Example:
        >>> client = BlindInsightClient(proxy_url="https://local.blindinsight.io")
        >>> data = client.query(
        ...     organization="my-org",
        ...     dataset_slug="iris-dataset",
        ...     schema_slug="iris-schema",
        ...     limit=150
        ... )
        >>> df = client.to_dataframe(data)
        >>> X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
        >>> y = df['species'].values
    """

    def __init__(
        self,
        proxy_url: str = "https://local.blindinsight.io",
        proxy_auth: tuple | None = None,
        verify_ssl: bool = True,
        pool_maxsize: int = 64,
    ):
        """
        Initialize the Blind Insight client (proxy HTTP API).

        Args:
            proxy_url: URL of the Blind Proxy (default: https://local.blindinsight.io)
            proxy_auth: Tuple of (email, password) for proxy auth, or None to use
                BI_EMAIL and BI_PASSWORD env vars. The proxy HTTP API authenticates
                each request separately from ``./blind login``.
            verify_ssl: Whether to verify SSL certificates (default: True).
                Set to False for local dev with self-signed certs.
            pool_maxsize: Max simultaneous HTTP connections per host. Defaults to 64.
                requests.Session's own default is only 10, which silently caps any
                ThreadPoolExecutor concurrency above 10 (the extra workers just wait
                on connections). Set this >= your largest ``max_workers`` so the
                worker count is real. Concurrency benchmarks put the useful range at
                ~24-48 workers before the server plateaus.
        """
        self.proxy_url = proxy_url.rstrip("/")
        self.session = requests.Session()
        self.session.verify = verify_ssl
        # Enlarge the connection pool so concurrent query workers aren't throttled
        # to the 10-connection default (see pool_maxsize docstring above).
        from requests.adapters import HTTPAdapter

        adapter = HTTPAdapter(pool_connections=pool_maxsize, pool_maxsize=pool_maxsize)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.profiling = profiling  # Use global profiling instance

        if proxy_auth:
            self._proxy_auth = proxy_auth
        else:
            email = os.environ.get("BI_EMAIL")
            password = os.environ.get("BI_PASSWORD")
            self._proxy_auth = (email, password) if email and password else None

        self._schema_id_cache: dict[str, str] = {}

    def _get_proxy_auth(self):
        """Return HTTPBasicAuth for proxy requests, or raise with a clear message."""
        if not self._proxy_auth:
            raise ValueError(
                "Proxy auth required. Set BI_EMAIL and BI_PASSWORD in your .env file.\n"
                "The proxy HTTP API authenticates each request separately from ./blind login.\n"
                "See .env.example for the template."
            )
        from requests.auth import HTTPBasicAuth

        return HTTPBasicAuth(*self._proxy_auth)

    def _get_schema_id(self, organization: str, dataset_slug: str, schema_slug: str) -> str:
        """Get schema ID from cache or resolve it via API."""
        cache_key = f"{organization}/{dataset_slug}/{schema_slug}"
        if cache_key in self._schema_id_cache:
            return self._schema_id_cache[cache_key]

        auth = self._get_proxy_auth()

        # Get org ID
        resp = self.session.get(f"{self.proxy_url}/api/organizations/by-slug/{organization}/", auth=auth)
        resp.raise_for_status()
        org_id = resp.json()["id"]

        # Get dataset ID
        resp = self.session.get(f"{self.proxy_url}/api/organizations/{org_id}/dataset/{dataset_slug}/", auth=auth)
        resp.raise_for_status()
        dataset_id = resp.json()["id"]

        # Get schema ID
        resp = self.session.get(f"{self.proxy_url}/api/datasets/{dataset_id}/schema/{schema_slug}/", auth=auth)
        resp.raise_for_status()
        schema_id = resp.json()["id"]

        self._schema_id_cache[cache_key] = schema_id
        return schema_id

    def warm_up(
        self,
        organization: str,
        dataset_slug: str,
        schema_slug: str,
        preflight_aggs: list = None,
        preflight_filters: list = None,
    ) -> None:
        """
        Pre-warm the client by caching schema IDs, establishing connections,
        and running throwaway queries to warm server-side encrypted indexes.

        Args:
            preflight_aggs: Legacy aggregate strings (e.g. "risk_level:count(50~100)")
            preflight_filters: List of filter lists for count_only queries
                               (e.g. [["cancer_5yr:1"], ["cancer_5yr:1", "age_group:50_59"]])
        """
        if preflight_aggs is None and preflight_filters is None:
            preflight_aggs = [
                "risk_level:count(0~100)",
                "risk_level:count(50~100),fraud_type:wire_transfer",
                "risk_level:count(50~100),account_jurisdiction:us",
                "risk_level:count(50~100),is_active:true",
                "risk_level:count(50~100),month:1",
                "risk_level:count(50~100),reporting_bank_id:BANK001",
                "risk_level:count(50~100),year:2024",
            ]

        import time
        from concurrent.futures import ThreadPoolExecutor

        start = time.time()

        # Pre-cache schema ID
        self._get_schema_id(organization, dataset_slug, schema_slug)
        id_elapsed = (time.time() - start) * 1000

        agg_start = time.time()
        n_queries = 0

        if preflight_filters:

            def _run_filter(filt):
                try:
                    self.query(organization, dataset_slug, schema_slug, filters=filt, limit=1, count_only=True)
                except Exception:
                    pass

            n_queries = len(preflight_filters)
            with ThreadPoolExecutor(max_workers=n_queries) as ex:
                list(ex.map(_run_filter, preflight_filters))

        if preflight_aggs:

            def _run_agg(agg):
                try:
                    self.aggregate(organization, dataset_slug, schema_slug, agg)
                except Exception:
                    pass

            n_queries += len(preflight_aggs)
            with ThreadPoolExecutor(max_workers=len(preflight_aggs)) as ex:
                list(ex.map(_run_agg, preflight_aggs))

        agg_elapsed = (time.time() - agg_start) * 1000
        total = (time.time() - start) * 1000
        print(
            f"  Proxy warm-up: schema ({id_elapsed:.0f}ms) "
            f"+ {n_queries} index preflights ({agg_elapsed:.0f}ms) "
            f"= {total:.0f}ms total"
        )

    def _proxy_query(
        self,
        organization: str,
        dataset_slug: str,
        schema_slug: str,
        limit: int,
        offset: int,
        filters: list[str] | None,
        decrypt: bool,
        count_only: bool = False,
    ) -> dict[str, Any]:
        """Query records directly via proxy API (fastest backend)."""
        auth = self._get_proxy_auth()
        start_time = time.time()

        # Get schema ID (cached after first call)
        schema_id = self._get_schema_id(organization, dataset_slug, schema_slug)

        # Build search payload
        payload = {
            "schema": schema_id,
            "limit": limit,
            "offset": offset,
        }

        # Convert filter strings to proxy format
        # Handle comma-separated filters (e.g., "field1:value1,field2:value2")
        if filters:
            proxy_filters = []
            for f in filters:
                # Split comma-separated filter strings first
                individual_filters = f.split(",")
                for individual in individual_filters:
                    if ":" in individual:
                        label, value = individual.split(":", 1)
                        proxy_filters.append({"label": label, "value": value})
            if proxy_filters:
                payload["filters"] = proxy_filters

        # Two-phase count_only: POST to get X-Query header, then GET with count_only
        # The proxy encrypts filters on POST but doesn't forward URL params.
        # The server returns X-Query (encrypted q params) in the response header.
        # A follow-up GET through the proxy passthrough delivers count_only to the server.
        if count_only:
            search_url = f"{self.proxy_url}/api/records/search/"
            phase1_payload = {**payload, "limit": 1}
            resp = self.session.post(search_url, json=phase1_payload, auth=auth)
            resp.raise_for_status()
            phase1_end = time.time()

            x_query = resp.headers.get("X-Query", "")
            if not x_query:
                # Fallback: no X-Query means no filters reached the server;
                # use the records response length as a rough count.
                result = resp.json()
                count_val = len(result) if isinstance(result, list) else 0
            else:
                count_url = f"{self.proxy_url}/api/records/?{x_query}&count_only=true"
                count_resp = self.session.get(count_url, auth=auth)
                count_resp.raise_for_status()
                count_data = count_resp.json()
                count_val = int(count_data.get("count", 0))

            end_time = time.time()
            if self.profiling.enabled:
                filter_str = ",".join(filters) if filters else "count_only"
                timing = QueryTiming(
                    query_type="count_only",
                    filter_str=filter_str,
                    start_time=start_time,
                    end_time=end_time,
                    network_ms=(phase1_end - start_time) * 1000,
                    parse_ms=(end_time - phase1_end) * 1000,
                    total_ms=(end_time - start_time) * 1000,
                    success=True,
                    error=None,
                )
                self.profiling.record(timing)
            return {"success": True, "count": count_val, "records": [], "encrypted": True}

        url = f"{self.proxy_url}/api/records/search/"
        resp = self.session.post(url, json=payload, auth=auth)
        resp.raise_for_status()
        result = resp.json()
        network_end = time.time()

        records = result

        # Decrypt if requested
        if decrypt and records:
            decrypt_resp = self.session.post(f"{self.proxy_url}/api/records/decrypt/", json=records, auth=auth)
            decrypt_resp.raise_for_status()
            records = decrypt_resp.json()

        end_time = time.time()

        # Record timing if profiling
        if self.profiling.enabled:
            filter_str = ",".join(filters) if filters else f"limit={limit}"
            timing = QueryTiming(
                query_type="query",
                filter_str=filter_str,
                start_time=start_time,
                end_time=end_time,
                network_ms=(network_end - start_time) * 1000,
                parse_ms=(end_time - network_end) * 1000,
                total_ms=(end_time - start_time) * 1000,
                success=True,
                error=None,
            )
            self.profiling.record(timing)

        # Normalize record format
        normalized = []
        for r in records:
            if isinstance(r, dict):
                if "data" in r:
                    normalized.append(r)
                else:
                    normalized.append({"data": r, "id": r.get("id", "")})
            else:
                normalized.append({"data": r})

        return {
            "success": True,
            "count": len(normalized),
            "records": normalized,
            "encrypted": not decrypt,
        }

    def query(
        self,
        organization: str,
        dataset_slug: str,
        schema_slug: str,
        limit: int = 1000,
        offset: int = 0,
        filters: list[str] | None = None,
        decrypt: bool = False,
        count_only: bool = False,
    ) -> dict[str, Any]:
        """
        Query records from Blind Insight using encrypted search.

        Blind Insight supports encrypted queries without decryption:
        - Equality: "field:value"
        - Comparisons: "field:>40", "field:<17", "field:>=47", "field:<=17"
        - Ranges: "field:40~45"
        - Aggregations: "field:avg(40~45)", "field:sum(>40)", "field:count(<15)", etc.

        Args:
            organization: Blind Insight organization slug
            dataset_slug: Dataset slug in Blind Insight
            schema_slug: Schema slug in Blind Insight
            limit: Maximum number of records to return (default: 1000)
            offset: Number of records to skip (default: 0)
            filters: List of encrypted filter strings (e.g., ["age:>40", "name:John"])
            decrypt: If True, decrypt data (only needed for ML operations that require plaintext)
            count_only: If True, return only {"count": N} without records (45-200x faster)

        Returns:
            Dictionary containing 'success', 'count', 'records', 'encrypted', etc.

        Raises:
            requests.RequestException: If the API request fails
        """
        return self._proxy_query(
            organization=organization,
            dataset_slug=dataset_slug,
            schema_slug=schema_slug,
            limit=limit,
            offset=offset,
            filters=filters,
            decrypt=decrypt,
            count_only=count_only,
        )

    def aggregate(
        self,
        organization: str,
        dataset_slug: str,
        schema_slug: str,
        agg_filter: str,
        extra_filters: list[str] | None = None,
        decrypt: bool = False,
    ) -> dict[str, Any]:
        """
        Run an aggregation query on encrypted data.

        Args:
            organization: Blind Insight organization slug
            dataset_slug: Dataset slug
            schema_slug: Schema slug
            agg_filter: Aggregation expression, e.g. "sepal-length:avg(0~10)" or "petal-width:count(<1.0)"
            extra_filters: Optional list of additional filters (e.g., ["species:I. setosa"])
            decrypt: Should remain False for encrypted aggregation; set True only if you explicitly need plaintext.

        Returns:
            Dictionary containing aggregation result. The aggregation value is typically in records[0]["data"]["value"].
        """
        filters = extra_filters or []
        filters = filters + [agg_filter]

        result = self.query(
            organization=organization,
            dataset_slug=dataset_slug,
            schema_slug=schema_slug,
            limit=1,
            offset=0,
            filters=filters,
            decrypt=decrypt,
        )

        # Update the last timing to be marked as 'aggregate' for profiling
        if self.profiling.enabled and self.profiling.timings:
            self.profiling.timings[-1].query_type = "aggregate"
            self.profiling.timings[-1].filter_str = agg_filter

        return result

    def to_dataframe(self, query_result: dict[str, Any], records_key: str = "records") -> pd.DataFrame:
        """
        Convert query result to a pandas DataFrame.

        Args:
            query_result: Result dictionary from query() method
            records_key: Key in the result dictionary containing records (default: "records")

        Returns:
            pandas DataFrame with the records
        """
        if not query_result.get("success"):
            raise ValueError(f"Query was not successful: {query_result.get('error', 'Unknown error')}")

        records = query_result.get(records_key, [])

        if not records:
            # Return empty DataFrame with proper structure if no records
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(records)

        return df

    def health_check(self) -> dict[str, Any]:
        """
        Check if the API is available and healthy.

        Returns:
            Dictionary with API status
        """
        url = f"{self.proxy_url}/api/health/"
        response = self.session.get(url, auth=self._get_proxy_auth())
        response.raise_for_status()
        return response.json()
