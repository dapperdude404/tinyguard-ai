# Contributing to SmartPlant AI

Thanks for helping improve this project. Contributions should be useful, truthful and reproducible; GitHub stars are appreciated but never required for support or participation.

## Before contributing

1. Read [README.md](README.md) and [DISCOVERY_PACK.md](DISCOVERY_PACK.md) for features, known limitations and current scope.
2. Search [existing issues](https://github.com/dapperdude404/tinyguard-ai/issues) to avoid duplicates. For a new report, use the issue templates.
3. Keep changes focused and work in your own branch. Do not change production deployment settings, credentials, releases, database data or model checkpoints without explicit review.
4. Add a test or a reproducible manual check when changing behavior.
5. Open a pull request with the problem, implementation, actual test output, screenshots where relevant, and unresolved risks.

## Running relevant checks

Install dependencies from README, then run `python -m pytest tests -q`; model/data-dependent tests may require additional resources.

If you cannot run a dependency-dependent check, write **not run** and why. Do not claim tests passed without logs or actual execution.

## Project-specific review

Research prototype, not certified agricultural diagnosis. Class probabilities are not real-world accuracy.

Image preprocessing, model checkpoints, dataset provenance and reproducible evaluation.

## Security and privacy

- Never commit passwords, API keys, OAuth credentials, dataset secrets or private user documents.
- Avoid uploading identifiable photos, confidential research data, personal paths, production logs or device identifiers in public issues.
- Share only assets you have rights to use and preserve credits and licenses.
- Use safe, consent-based examples. Disclose synthetic data, unverified results, model limits and unimplemented integrations.
- For security-sensitive vulnerabilities, do not post exploitation details or secrets publicly. Contact the maintainer privately using a trusted GitHub channel.

## Review and release

Maintainers may ask for smaller PRs or more test evidence. A passing build is not necessarily a validated deployment. **No PR should be merged or deployed solely on the basis of this documentation change without explicit maintainer review.**

## Community behavior

Be respectful, specific and constructive. Do not spam requests for GitHub stars, stage fake endorsements or manufacture contribution activity.
