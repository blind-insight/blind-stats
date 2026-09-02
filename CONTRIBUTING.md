# Contributing to blind-stats

Thanks for your interest. Contributions of all sizes are welcome — bug fixes, new
statistical methods, performance work, documentation improvements.

## Ground rules

- **Fork → branch → PR.** External contributors should fork the repo, push to a branch on their fork, and open a PR against `main`.
- **One change per PR.** Smaller PRs ship faster.
- **Open an issue first for anything substantial.** Getting alignment on the approach before you invest a weekend saves everyone time.
- **All contributions are MIT-licensed.** By submitting a PR, you agree your contribution is licensed under the same MIT license as the rest of the repo.

## The one rule that is not negotiable

Everything in `blind_stats/stats.py` must be computable from Blind Insight
`aggregate` and `count` responses. A method may not request decrypted records or
depend on plaintext row access. If your statistic needs record-level data, it
does not belong here — no matter how useful it is.

`scripts/smoke_test.py` enforces this: it drives `BIStatsSession` against a fake
client that raises if the stats code ever asks for decryption or omits
`count_only` on a count query. If your change makes that test fail, the change is
wrong, not the test.

The plaintext fixture in `fixtures/` is a benchmark for parity checks only. Never
read it from library code.

## Setup

```bash
git clone https://github.com/<your-fork>/blind-stats.git
cd blind-stats
pip install -e ".[dev,notebooks]"   # editable install + dev + notebook deps
cp .env.example .env                 # fill in your Blind Insight credentials
```

You'll need a Blind Insight account to run the notebook end-to-end. Sign up at
[blindinsight.com](https://blindinsight.com); the proxy and CLI are downloadable
from [docs.blindinsight.io](https://docs.blindinsight.io).

Create a dataset and schema from `schemas/fraud.json`, upload
`fixtures/fraud_train_290.json` to it, and point `BI_STATS_DATASET` /
`BI_STATS_SCHEMA` at it in `.env`.

## Adding a statistical method

Public methods on `BIStatsSession` decompose into the `_aggregate_value` and
`_count_only` primitives, which handle retry and result caching. Use them rather
than calling the client directly, and return one of the existing `*Result`
dataclasses so callers get a consistent `.statistic` / `.pvalue` / `.queries`
surface.

Two things worth knowing before you start:

- **Query count is the cost model.** Each primitive is a network round-trip.
  Reusing an already-computed marginal beats issuing a fresh query; the `queries`
  field on every result exists so callers can see what a method cost them.
- **Add a case to `stats_smoke()`** in `scripts/smoke_test.py` asserting your
  method against a hand-computable expected value on the 5-record fake dataset.
  That is the only automated coverage in this repo.

## Linting and formatting

Run before opening a PR — CI will reject otherwise.

```bash
ruff check --fix .
ruff format .
nbqa ruff --fix --extend-ignore=E402,F401,E702,E401,I001,F811,F541 statistics.ipynb
python3 scripts/smoke_test.py
```

Clear notebook outputs before pushing — CI does not strip them:

```bash
jupyter nbconvert --clear-output --inplace statistics.ipynb
```

## Submitting a PR

1. Push your branch to your fork
2. Open a PR against `blind-insight/blind-stats:main`
3. Fill out the PR template (what changed, how you tested, linked issue)
4. CI runs automatically. A Blind Insight maintainer will review and merge.

External contributors cannot self-merge — every PR is reviewed and merged by a maintainer.

## What we're especially looking for

- **More inferential methods** — anything expressible in counts and moments
- **Better quantile search** — the binary search over range counts is the hot path
- **Privacy guardrails** — differential-privacy noise, smarter cell suppression
- **Performance work** — query batching, smarter caching, parallel execution patterns
- **Docs** — clearer explanations, diagrams, runnable tutorials

## Questions?

Open a discussion or ping the maintainers in your PR. We're happy to help shape contributions early.
