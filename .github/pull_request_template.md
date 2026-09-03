## Summary

<!-- One or two sentences on what this PR changes and why. -->

## Changes

<!-- Bullet list of the key changes. -->

-
-

## How I tested

<!-- Did you run the smoke test? The notebook end-to-end against a live proxy? Anything reviewers should reproduce? -->

- [ ] Ran `ruff check .` and `ruff format --check .` — no errors
- [ ] Ran `python3 scripts/smoke_test.py` — all checks pass
- [ ] Ran `statistics.ipynb` end-to-end against a live proxy (if the change affects query behavior)
- [ ] Cleared notebook outputs before push
- [ ] No new code path requests decrypted records or plaintext rows
- [ ] Updated `README.md` and/or `CLAUDE.md` if behavior or paths changed
- [ ] No secrets, credentials, or internal infra references in the diff

## Linked issue

<!-- Closes #123 -->

## Notes for reviewers

<!-- Anything specific you'd like a maintainer to look at? Trade-offs you considered? -->
