# blind-stats — Statistics on data that is never decrypted

`blind-stats` computes descriptive statistics, hypothesis tests, correlations,
regression summaries, and drift/monitoring reports over data held in
[Blind Insight](https://blindinsight.com) searchable encryption — **without ever
decrypting a record**.

Every number comes out of encrypted `aggregate` and `count` query responses. Raw
rows never leave the vault, and no code path in this package asks for plaintext.

```python
from blind_stats import BIStatsSession, BlindInsightClient, resolve_target

target = resolve_target()
client = BlindInsightClient(proxy_url=target["proxy_url"], verify_ssl=target["verify_ssl"])

stats = BIStatsSession(
    client,
    org=target["org"],
    dataset=target["dataset"],
    schema=target["schema"],
    field_domains={"risk_level": (0, 102)},
)

stats.mean("risk_level").statistic  # 48.31…
stats.median("risk_level").statistic  # binary search over count queries
stats.chi2_independence("fraud_type", "is_active")
stats.describe("risk_level")
```

## The invariant

Statistics code must work from aggregate/count responses only. It must never
request decrypted records or depend on plaintext row access. If a feature would
need record-level data, it does not belong in `blind_stats/stats.py`.

The bundled plaintext fixture (`fixtures/fraud_train_290.json`) exists **only** as
a source-of-truth benchmark, so the notebook can prove the encrypted results match
exactly. It is never a data source for `BIStatsSession`.

`scripts/smoke_test.py` enforces this offline: it drives `BIStatsSession` against a
fake client that raises if the stats code ever requests decryption or skips
`count_only`.

## Architecture

Four modules, ~3.7k lines, no ML dependencies:

| Module | Role |
| --- | --- |
| `blind_stats/client.py` | `BlindInsightClient` — the only thing that talks to Blind Insight, over the Blind Proxy HTTP API. `query`, `aggregate`, `warm_up`, `to_dataframe`. All queries route through `_proxy_query`; `ProfilingStats` (global `profiling`) records per-query timing. Schema IDs are resolved and cached from org/dataset/schema slugs. |
| `blind_stats/stats.py` | `BIStatsSession` — the statistics core. Every public method decomposes into `_aggregate_value` / `_count_only` primitives with retry and result caching. Also `BILinearRegression` and the `*Result` dataclasses. |
| `blind_stats/config.py` | `load_env`, `require_env`, `get_demo_config`, `resolve_target`. Where to point a session — no domain knowledge. |
| `blind_stats/__init__.py` | Flat re-export surface. |

`BIStatsSession` takes any object with `.aggregate(...)` and
`.query(..., count_only=True)`, so it is testable against a fake client and not
coupled to the HTTP layer.

## What it computes

**Descriptive** — `count`, `sum_`, `mean`, `min_`, `max_`, `range_`, `var`, `std`,
`median`, `quantile`, `iqr`, `percentile_rank`, `ecdf`, `mode`, `histogram`,
`frequency`, `proportion`, `rate`, `count_range`, `count_eq`, `describe`

**Categorical tables** — `crosstab`, `chisquare`, `chi2_independence`,
`fisher_exact`, `odds_ratio`, `relative_risk`, `ztest_proportion`,
`ztest_proportions`, `proportion_ci`

**Grouped + mean inference** — `groupby_count`, `groupby_mean`, `groupby_var`,
`groupby_std`, `mean_ci`, `ttest_1samp`, `ttest_ind`, `anova_oneway`, `welch_anova`

**Nonparametric** — `ks_2samp`, `mannwhitneyu`

**Correlation + effect size** — `onehot_covariance`, `onehot_correlation`,
`binned_pearson`, `binned_spearman`, `point_biserial`, `eta_squared`

**Regression** — `BILinearRegression.fit(...).summary()` (OLS/ridge from
aggregate sufficient statistics), `feature_screening`

**Monitoring, drift, data quality** — `missingness_rate`,
`domain_violation_count`, `zscore_outlier_count`, `iqr_outlier_count`,
`distribution`, `population_stability_index`, `distribution_divergence`,
`drift_chi2`, `period_counts`, `period_summary`, `data_quality_report`

## How it works

Blind Insight answers **aggregate** and **count** queries over an encrypted index.
That is enough for a surprising amount of statistics:

- **Sums and means** come straight from numeric aggregate functions.
- **Variance** comes from `E[X²] - E[X]²` — two aggregate calls.
- **Quantiles and the median** are a bounded binary search over `count(lo~hi)`
  range queries. This is why `field_domains` matters: it gives the search its
  bracket. Without a domain, the session cannot bound the search.
- **Contingency tables** are a grid of filtered counts; every χ², Fisher, odds
  ratio, and proportion z-test is built from that grid.
- **Correlation** needs cross-moments. For categoricals, `onehot_covariance`
  gets them from co-occurrence counts. For numeric pairs, either bin both fields
  (`binned_pearson`) or precompute a derived product column at upload time and
  point `moment_fields` at it.
- **Regression** assembles X'X and X'y from those same sufficient statistics,
  then solves locally.

`min_cell_size` suppresses any result derived from a cell below the threshold, so
small-cell disclosure is a configuration setting rather than a review step.

### Query syntax gotchas

- Ranges use a tilde, not a comma: `field:count(50~100)`.
- An integer schema's `maximum` must be the actual max **+ 2** (see
  `schemas/fraud.json`: `risk_level` tops out at 100 in the data, 102 in the schema).
- `count()` targets on string fields fail silently — count string fields by
  filtering, not by aggregating.
- A zero count and an unresolvable query look identical in the response. Verify
  the record count first; the notebook stops early on a mismatch for this reason.

## Quick start

### 1. Install

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev,notebooks]"
```

### 2. Two independent auth mechanisms — both required

```bash
./blind login          # proxy CLI + keyring: the encryption keys
cp .env.example .env   # then fill in BI_EMAIL / BI_PASSWORD / BI_ORG
```

The Python client authenticates each HTTP request separately with basic auth from
`.env`. Blind Insight uses two keys per field — a **query key** for aggregates and
a **field key** for decrypt. `blind-stats` only ever needs the query key.

### 3. Start the Blind Proxy

It listens on `https://local.blindinsight.io` with a self-signed cert, which is
why `BI_VERIFY_SSL` defaults to `false`.

### 4. Create the dataset + schema and upload the fixture

Create a dataset and schema in Blind Insight using `schemas/fraud.json`, then
upload `fixtures/fraud_train_290.json` to it. Set `BI_STATS_DATASET` and
`BI_STATS_SCHEMA` in `.env` to match.

### 5. Run it

```bash
python3 scripts/smoke_test.py     # offline: imports, config, full stats battery
jupyter notebook statistics.ipynb # live: 54 cells against the real proxy
```

## The notebook

`statistics.ipynb` walks every family of statistics against a real proxy. Each
section ends with a parity check: the encrypted result next to the same statistic
computed on the plaintext fixture, confirming they match **exactly** while the
encrypted path decrypted **0 rows**.

It refuses to draw conclusions against the wrong data — if the schema's record
count does not match the fixture, it says so up front rather than comparing
mismatched populations.

## Pointing it at your own data

`get_demo_config()` in `blind_stats/config.py` returns the fixture's slugs and
field domains. For your own schema, set `BI_STATS_DATASET` / `BI_STATS_SCHEMA` and
pass your own `field_domains` — an `(lo, hi)` tuple for each numeric field you
want quantiles on, an explicit value list for each categorical.

## Development

```bash
ruff check --fix .
ruff format .
nbqa ruff --fix --extend-ignore=E402,F401,E702,E401,I001,F811,F541 statistics.ipynb
python3 scripts/smoke_test.py
```

Clear notebook outputs before every PR — CI does not strip them:

```bash
jupyter nbconvert --clear-output --inplace statistics.ipynb
```

## License

MIT — see [LICENSE](LICENSE).
