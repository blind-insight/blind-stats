#!/usr/bin/env python3
"""Repo smoketest — run from repo root: python3 scripts/smoke_test.py

Offline only: the stats check runs BIStatsSession against a fake in-process client
that raises if the stats code ever asks for decrypted records, so this doubles as a
guard on the zero-decrypt invariant. No Blind Proxy required.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

STATS_EXPORTS = [
    "BIStatsSession",
    "BILinearRegression",
    "StatResult",
    "HistogramResult",
    "FrequencyResult",
    "DescribeResult",
    "GroupedStatsResult",
    "CrosstabResult",
    "MatrixStatsResult",
    "RegressionSummaryResult",
    "FeatureScreeningResult",
]


def check(name: str, fn) -> None:
    try:
        fn()
        print(f"  OK  {name}")
    except Exception as e:
        print(f"  FAIL {name}: {e}")
        raise


def _require_symbols(module_name: str, symbols: list[str]) -> None:
    mod = importlib.import_module(module_name)
    missing = [s for s in symbols if not hasattr(mod, s)]
    if missing:
        raise AttributeError(f"{module_name} missing: {', '.join(missing)}")


def main() -> int:
    print("blind-stats smoketest\n")

    check("blind_stats package", lambda: importlib.import_module("blind_stats"))
    check("stats exports", lambda: _require_symbols("blind_stats", STATS_EXPORTS))
    check("stats module", lambda: _require_symbols("blind_stats.stats", STATS_EXPORTS))
    check("client", lambda: _require_symbols("blind_stats.client", ["BlindInsightClient", "profiling"]))
    check(
        "config",
        lambda: _require_symbols(
            "blind_stats.config", ["load_env", "require_env", "get_demo_config", "resolve_target"]
        ),
    )

    def no_decrypt_helpers():
        client_mod = importlib.import_module("blind_stats.client")
        for banned in ("load_data", "load_iris_from_blind", "load_fraud_from_blind", "load_account_risk_from_blind"):
            if hasattr(client_mod, banned) or hasattr(client_mod.BlindInsightClient, banned):
                raise AssertionError(f"blind-stats must not ship the plaintext helper {banned!r}")

    check("no plaintext/decrypt convenience helpers", no_decrypt_helpers)

    def demo_config():
        cfg = importlib.import_module("blind_stats").get_demo_config()
        assert cfg["dataset"] and cfg["schema"]
        assert cfg["field_domains"]["risk_level"] == (0, 102)

    check("demo config", demo_config)

    def client_defaults():
        client_mod = importlib.import_module("blind_stats.client")
        c = client_mod.BlindInsightClient()
        assert "localhost" in c.proxy_url or "blindinsight" in c.proxy_url

    check("BlindInsightClient()", client_defaults)

    def schemas():
        schema = json.loads((REPO_ROOT / "schemas" / "fraud.json").read_text())
        # BI integer schemas need maximum == actual max + 2
        assert schema["properties"]["risk_level"]["maximum"] == 102

    check("JSON schema", schemas)

    def fixture():
        path = REPO_ROOT / "fixtures" / "fraud_train_290.json"
        records = json.loads(path.read_text())
        assert len(records) == 290, f"expected 290 fixture records, got {len(records)}"
        assert "data" in records[0]

    check("fixtures/fraud_train_290.json (290 records)", fixture)

    def stats_smoke():
        from scipy import stats as scipy_stats

        from blind_stats import BILinearRegression, BIStatsSession

        class FakeBIClient:
            def __init__(self):
                base_records = [
                    {"score": 0, "kind": "a", "flag": "no", "period": "jan"},
                    {"score": 1, "kind": "a", "flag": "yes", "period": "jan"},
                    {"score": 2, "kind": "b", "flag": "yes", "period": "feb"},
                    {"score": 3, "kind": "b", "flag": "yes", "period": "feb"},
                    {"score": 4, "kind": "b", "flag": "no", "period": "feb"},
                ]
                self.records = []
                for record in base_records:
                    y = 1 + (2 * record["score"])
                    enriched = dict(record)
                    enriched["y"] = y
                    enriched["score_squared"] = record["score"] * record["score"]
                    enriched["score_y"] = record["score"] * y
                    enriched["y_squared"] = y * y
                    self.records.append(enriched)

            def load_data(self, *args, **kwargs):
                raise AssertionError("stats code must not load plaintext data")

            def query(self, **kwargs):
                if kwargs.get("decrypt"):
                    raise AssertionError("stats code must not request decrypted records")
                if not kwargs.get("count_only"):
                    raise AssertionError("stats code must use count_only for query counts")
                filters = kwargs.get("filters") or []
                return {"success": True, "count": len(self._filtered(filters)), "records": [], "encrypted": True}

            def aggregate(self, **kwargs):
                if kwargs.get("decrypt"):
                    raise AssertionError("stats code must not request decrypted aggregates")
                agg_filter = kwargs["agg_filter"]
                filters = kwargs.get("extra_filters") or []
                field, op, expr = self._parse_agg(agg_filter)
                rows = self._filtered(filters)
                values = [float(row[field]) for row in rows if self._matches_expr(row[field], expr)]
                if op == "count":
                    value = len(values)
                elif op == "sum":
                    value = sum(values)
                elif op == "avg":
                    value = sum(values) / len(values) if values else 0.0
                elif op == "min":
                    value = min(values) if values else 0.0
                elif op == "max":
                    value = max(values) if values else 0.0
                else:
                    raise AssertionError(f"unexpected aggregate op: {op}")
                return {"success": True, "records": [{"data": {"value": value}}], "encrypted": True}

            def _filtered(self, filters):
                rows = self.records
                for raw_filter in filters:
                    for filt in str(raw_filter).split(","):
                        if not filt:
                            continue
                        field, expected = filt.split(":", 1)
                        if "~" in expected:
                            low, high = expected.split("~", 1)
                            rows = [row for row in rows if float(low) <= float(row[field]) <= float(high)]
                        else:
                            rows = [row for row in rows if str(row[field]).lower() == expected.lower()]
                return rows

            @staticmethod
            def _parse_agg(agg_filter):
                field, rest = agg_filter.split(":", 1)
                op, expr = rest.split("(", 1)
                return field, op, expr.rstrip(")")

            @staticmethod
            def _matches_expr(value, expr):
                value = float(value)
                if "~" in expr:
                    low, high = expr.split("~", 1)
                    return float(low) <= value <= float(high)
                return value == float(expr)

        stats = BIStatsSession(
            FakeBIClient(),
            org="org",
            dataset="dataset",
            schema="schema",
            field_domains={"score": (0, 4), "kind": ["a", "b"], "flag": ["yes", "no"]},
            max_workers=2,
        )
        stats.field_domains.update(
            {
                "y": (1, 9),
                "score_squared": (0, 16),
                "score_y": (0, 36),
                "y_squared": (1, 81),
                "period": ["jan", "feb"],
            }
        )

        assert stats.count().statistic == 5
        assert stats.mean("score").statistic == 2.0
        assert stats.min_("score").statistic == 0.0
        assert stats.max_("score").statistic == 4.0
        assert stats.range_("score").statistic == 4.0
        assert math.isclose(stats.var("score").statistic, 2.5)
        assert math.isclose(stats.std("score").statistic, math.sqrt(2.5))
        assert stats.median("score").statistic == 2
        assert stats.iqr("score").statistic == 2

        hist = stats.histogram("score", bins=[(0, 1), (2, 4)])
        assert hist.counts == {"0~1": 2, "2~4": 3}
        freq = stats.frequency("kind")
        assert freq.counts == {"a": 2, "b": 3}
        assert freq.mode == "b"
        assert stats.mode("kind").statistic == "b"
        assert math.isclose(stats.proportion("kind:b").statistic, 0.6)
        assert math.isclose(stats.rate("kind:b").statistic, 0.6)
        assert stats.percentile_rank("score", 2).statistic == 0.6
        assert stats.ecdf("score", [1, 4]).statistic == {"1": 0.4, "4": 1.0}

        desc = stats.describe("score")
        assert desc.n == 5
        assert desc.statistics["mean"] == 2.0
        assert math.isclose(desc.statistics["variance"], 2.5)
        assert desc.statistics["median"] == 2
        assert desc.exact["variance"] is True

        table = stats.crosstab("kind", "flag")
        assert table.counts == {"a": {"yes": 1, "no": 1}, "b": {"yes": 2, "no": 1}}
        assert table.row_totals == {"a": 2, "b": 3}
        assert table.col_totals == {"yes": 3, "no": 2}

        goodness = stats.chisquare("kind")
        expected_goodness = scipy_stats.chisquare([2, 3])
        assert math.isclose(goodness.statistic, expected_goodness.statistic)
        assert math.isclose(goodness.pvalue, expected_goodness.pvalue)

        chi2 = stats.chi2_independence("kind", "flag", correction=False)
        expected_chi2 = scipy_stats.chi2_contingency([[1, 1], [2, 1]], correction=False)
        assert math.isclose(chi2.statistic, expected_chi2.statistic)
        assert math.isclose(chi2.pvalue, expected_chi2.pvalue)
        assert chi2.estimate["dof"] == 1

        fisher = stats.fisher_exact("kind", "flag")
        expected_fisher = scipy_stats.fisher_exact([[1, 1], [2, 1]])
        assert math.isclose(fisher.statistic, expected_fisher.statistic)
        assert math.isclose(fisher.pvalue, expected_fisher.pvalue)

        assert math.isclose(stats.odds_ratio("kind", "flag", "a", "yes").statistic, 0.5)
        assert math.isclose(stats.relative_risk("kind", "flag", "a", "yes").statistic, 0.75)

        z_one = stats.ztest_proportion("flag:yes", p=0.5)
        assert z_one.estimate["successes"] == 3
        assert z_one.estimate["trials"] == 5
        assert math.isfinite(z_one.pvalue)

        z_two = stats.ztest_proportions("flag:yes", "kind", "a", "b")
        assert z_two.estimate["successes_a"] == 1
        assert z_two.estimate["successes_b"] == 2
        assert math.isfinite(z_two.pvalue)

        wilson = stats.proportion_ci("flag:yes", method="wilson")
        clopper = stats.proportion_ci("flag:yes", method="clopper-pearson")
        assert wilson.confidence_interval[0] <= wilson.statistic <= wilson.confidence_interval[1]
        assert clopper.confidence_interval[0] <= clopper.statistic <= clopper.confidence_interval[1]

        grouped = stats.groupby_mean("score", "kind")
        assert grouped.statistics["a"]["count"] == 2
        assert grouped.statistics["b"]["count"] == 3
        assert grouped.statistics["a"]["mean"] == 0.5
        assert grouped.statistics["b"]["mean"] == 3.0

        mean_ci = stats.mean_ci("score")
        assert mean_ci.confidence_interval[0] <= mean_ci.statistic <= mean_ci.confidence_interval[1]

        one_sample = stats.ttest_1samp("score", popmean=2.0)
        assert math.isclose(one_sample.statistic, 0.0)
        assert math.isclose(one_sample.pvalue, 1.0)

        welch_t = stats.ttest_ind("score", "kind", "a", "b")
        expected_welch = scipy_stats.ttest_ind([0, 1], [2, 3, 4], equal_var=False)
        assert math.isclose(welch_t.statistic, expected_welch.statistic)
        assert math.isclose(welch_t.pvalue, expected_welch.pvalue)

        pooled_t = stats.ttest_ind("score", "kind", "a", "b", equal_var=True)
        expected_pooled = scipy_stats.ttest_ind([0, 1], [2, 3, 4], equal_var=True)
        assert math.isclose(pooled_t.statistic, expected_pooled.statistic)
        assert math.isclose(pooled_t.pvalue, expected_pooled.pvalue)

        anova = stats.anova_oneway("score", "kind")
        expected_anova = scipy_stats.f_oneway([0, 1], [2, 3, 4])
        assert math.isclose(anova.statistic, expected_anova.statistic)
        assert math.isclose(anova.pvalue, expected_anova.pvalue)

        ks = stats.ks_2samp("score", "kind", "a", "b")
        expected_ks = scipy_stats.ks_2samp([0, 1], [2, 3, 4], method="auto")
        assert math.isclose(ks.statistic, expected_ks.statistic)
        assert math.isclose(ks.pvalue, expected_ks.pvalue)

        mw = stats.mannwhitneyu("score", "kind", "a", "b")
        expected_mw = scipy_stats.mannwhitneyu([0, 1], [2, 3, 4], method="auto")
        assert math.isclose(mw.statistic, expected_mw.statistic)
        assert math.isclose(mw.pvalue, expected_mw.pvalue)

        matrix = stats.onehot_covariance({"kind": ["a", "b"], "flag": ["yes", "no"]})
        assert matrix.n == 5
        assert set(matrix.fields) == {"kind:a", "kind:b", "flag:yes", "flag:no"}
        assert math.isclose(
            stats.binned_pearson("score", "y", x_bins=[(0, 1), (2, 4)], y_bins=[(1, 3), (5, 9)]).statistic,
            1.0,
        )
        assert math.isfinite(
            stats.population_stability_index("kind", "period:jan", "period:feb", values=["a", "b"]).statistic
        )

        regression = (
            BILinearRegression()
            .fit(
                stats,
                y="y",
                x=["score"],
                moment_fields={
                    ("score", "score"): "score_squared",
                    ("score", "y"): "score_y",
                    ("y", "y"): "y_squared",
                },
            )
            .summary()
        )
        assert math.isclose(regression.coefficients["intercept"], 1.0)
        assert math.isclose(regression.coefficients["score"], 2.0)

    check("BIStatsSession descriptive stats", stats_smoke)

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.exit(1)
