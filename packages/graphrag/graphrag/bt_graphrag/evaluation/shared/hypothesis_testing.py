"""Component-agnostic statistical hypothesis tests.

Works with any pair of scorers that produce a ``pair_details`` list where
each item has an ``"outcome"`` key in {"TP", "FP", "TN", "FN"} and a
``"score"`` key with the raw similarity score.

Implemented tests
-----------------
McNemar          – paired binary disagreement test (chi-squared, 1 df)
Bootstrap F1     – bootstrapped confidence interval for F1 difference
Permutation      – exact permutation test on accuracy difference
Friedman         – multi-scorer rank test (chi-squared, k-1 df)
Wilcoxon         – signed-rank test on per-pair score differences

All p-values are two-sided unless otherwise noted.  Significance level α=0.05.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Shared result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    test_name: str
    component: str          # "CGER", "CGRR", …
    scorer_a: str
    scorer_b: str | None    # None for omnibus tests (Friedman)
    statistic: float
    p_value: float
    significant: bool       # p < 0.05
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "component": self.component,
            "scorer_a": self.scorer_a,
            "scorer_b": self.scorer_b,
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 6),
            "significant": self.significant,
            "details": self.details,
        }


@dataclass
class HypothesisReport:
    component: str
    tests: list[TestResult]
    summary: dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "tests": [t.to_dict() for t in self.tests],
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------

def mcnemar_test(
    outcomes_a: list[bool],
    outcomes_b: list[bool],
    scorer_a: str,
    scorer_b: str,
    component: str = "",
) -> TestResult:
    """Paired binary test — H0: P(A correct & B wrong) = P(A wrong & B correct)."""
    b = sum(1 for a, bv in zip(outcomes_a, outcomes_b) if a and not bv)
    c = sum(1 for a, bv in zip(outcomes_a, outcomes_b) if not a and bv)

    if b + c == 0:
        return TestResult(
            test_name="McNemar", component=component,
            scorer_a=scorer_a, scorer_b=scorer_b,
            statistic=0.0, p_value=1.0, significant=False,
            details={"b": b, "c": c, "note": "No disagreements between scorers"},
        )

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = _chi2_sf(chi2, df=1)
    return TestResult(
        test_name="McNemar", component=component,
        scorer_a=scorer_a, scorer_b=scorer_b,
        statistic=chi2, p_value=p_value, significant=p_value < 0.05,
        details={"b_a_right_b_wrong": b, "c_a_wrong_b_right": c},
    )


# ---------------------------------------------------------------------------
# Bootstrapped paired F1
# ---------------------------------------------------------------------------

def bootstrap_paired_f1(
    pairs_a: list[dict],
    pairs_b: list[dict],
    scorer_a: str,
    scorer_b: str,
    component: str = "",
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> TestResult:
    """Non-parametric bootstrap CI for F1(A) − F1(B)."""
    rng = random.Random(seed)
    n = min(len(pairs_a), len(pairs_b))

    def _f1(pairs: list[dict]) -> float:
        tp = sum(1 for p in pairs if p["outcome"] == "TP")
        fp = sum(1 for p in pairs if p["outcome"] == "FP")
        fn = sum(1 for p in pairs if p["outcome"] == "FN")
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0

    f1_a = _f1(pairs_a[:n])
    f1_b = _f1(pairs_b[:n])
    diff_obs = f1_a - f1_b

    diffs: list[float] = []
    for _ in range(n_bootstrap):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        diffs.append(_f1([pairs_a[i] for i in idx]) - _f1([pairs_b[i] for i in idx]))
    diffs.sort()

    lo = diffs[int(n_bootstrap * 0.025)]
    hi = diffs[int(n_bootstrap * 0.975)]
    p_val = sum(1 for d in diffs if d * diff_obs <= 0) / n_bootstrap
    significant = not (lo <= 0 <= hi)

    return TestResult(
        test_name="Bootstrap_F1", component=component,
        scorer_a=scorer_a, scorer_b=scorer_b,
        statistic=diff_obs, p_value=p_val, significant=significant,
        details={
            "f1_a": round(f1_a, 4), "f1_b": round(f1_b, 4),
            "diff_obs": round(diff_obs, 4),
            "ci_95": [round(lo, 4), round(hi, 4)],
            "n_bootstrap": n_bootstrap,
        },
    )


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def permutation_test(
    outcomes_a: list[bool],
    outcomes_b: list[bool],
    scorer_a: str,
    scorer_b: str,
    component: str = "",
    n_permutations: int = 10_000,
    seed: int = 42,
) -> TestResult:
    """Exact permutation test on paired accuracy difference."""
    rng = random.Random(seed)
    n = min(len(outcomes_a), len(outcomes_b))
    acc_a = sum(outcomes_a[:n]) / n
    acc_b = sum(outcomes_b[:n]) / n
    obs = acc_a - acc_b

    extreme = sum(
        1 for _ in range(n_permutations)
        if abs(sum(
            (outcomes_a[i] - outcomes_b[i]) * (1 if rng.random() < 0.5 else -1)
            for i in range(n)
        ) / n) >= abs(obs)
    )
    p_val = extreme / n_permutations

    return TestResult(
        test_name="Permutation", component=component,
        scorer_a=scorer_a, scorer_b=scorer_b,
        statistic=obs, p_value=p_val, significant=p_val < 0.05,
        details={
            "accuracy_a": round(acc_a, 4), "accuracy_b": round(acc_b, 4),
            "n_permutations": n_permutations,
        },
    )


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test on per-pair scores
# ---------------------------------------------------------------------------

def wilcoxon_test(
    scores_a: list[float],
    scores_b: list[float],
    scorer_a: str,
    scorer_b: str,
    component: str = "",
) -> TestResult:
    """Wilcoxon signed-rank test on raw score differences."""
    n = min(len(scores_a), len(scores_b))
    diffs = [scores_a[i] - scores_b[i] for i in range(n)]
    nonzero = [(d, abs(d)) for d in diffs if d != 0]
    if not nonzero:
        return TestResult(
            test_name="Wilcoxon", component=component,
            scorer_a=scorer_a, scorer_b=scorer_b,
            statistic=0.0, p_value=1.0, significant=False,
            details={"note": "All differences are zero"},
        )

    ranked = sorted(enumerate(nonzero), key=lambda x: x[1][1])
    w_plus = w_minus = 0.0
    for rank, (orig_idx, (d, _)) in enumerate(ranked, 1):
        if d > 0:
            w_plus += rank
        else:
            w_minus += rank

    w = min(w_plus, w_minus)
    m = len(nonzero)
    # Normal approximation (valid for m >= 10)
    mu = m * (m + 1) / 4
    sigma = math.sqrt(m * (m + 1) * (2 * m + 1) / 24)
    z = (w - mu) / sigma if sigma > 0 else 0.0
    p_val = 2 * _normal_sf(abs(z))

    return TestResult(
        test_name="Wilcoxon", component=component,
        scorer_a=scorer_a, scorer_b=scorer_b,
        statistic=w, p_value=p_val, significant=p_val < 0.05,
        details={
            "W_plus": round(w_plus, 2), "W_minus": round(w_minus, 2),
            "z_approx": round(z, 4), "n_nonzero": m,
        },
    )


# ---------------------------------------------------------------------------
# Friedman test (omnibus, k >= 3 scorers)
# ---------------------------------------------------------------------------

def friedman_test(
    scorer_pair_details: dict[str, list[dict]],
    component: str = "",
) -> TestResult:
    """Friedman rank test — H0: all scorers have equal mean rank."""
    names = sorted(scorer_pair_details.keys())
    k, n = len(names), len(scorer_pair_details[names[0]])

    if k < 3:
        return TestResult(
            test_name="Friedman", component=component,
            scorer_a="all", scorer_b=None,
            statistic=0.0, p_value=1.0, significant=False,
            details={"note": "Need ≥3 scorers"},
        )

    rank_sums: dict[str, float] = {n: 0.0 for n in names}
    for i in range(n):
        row = sorted(
            names,
            key=lambda nm: (
                1 if scorer_pair_details[nm][i]["outcome"] in ("TP", "TN") else 0,
                scorer_pair_details[nm][i].get("score", 0.0),
            ),
            reverse=True,
        )
        for rank, nm in enumerate(row, 1):
            rank_sums[nm] += rank

    mean_ranks = {nm: rank_sums[nm] / n for nm in names}
    chi2 = (12 * n / (k * (k + 1))) * sum(
        (mean_ranks[nm] - (k + 1) / 2) ** 2 for nm in names
    )
    p_val = _chi2_sf(chi2, df=k - 1)

    return TestResult(
        test_name="Friedman", component=component,
        scorer_a="all_scorers", scorer_b=None,
        statistic=chi2, p_value=p_val, significant=p_val < 0.05,
        details={
            "mean_ranks": {nm: round(r, 3) for nm, r in mean_ranks.items()},
            "k": k, "n": n,
        },
    )


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_hypothesis_tests(
    scorer_results: dict[str, Any],
    component: str = "",
) -> HypothesisReport:
    """Run all hypothesis tests for a set of scorer evaluation results.

    Parameters
    ----------
    scorer_results:
        ``{scorer_name: ScorerResult}`` — each ScorerResult must have a
        ``.pair_details`` attribute (or ``["pair_details"]`` key) containing
        dicts with ``"outcome"`` and ``"score"`` fields.
    component:
        Label for the pipeline component being tested, e.g. "CGER" or "CGRR".
    """
    all_tests: list[TestResult] = []
    names = sorted(scorer_results.keys())

    def _details(sr: Any) -> list[dict]:
        return sr.pair_details if hasattr(sr, "pair_details") else sr.get("pair_details", [])

    def _outcomes(details: list[dict]) -> list[bool]:
        return [d["outcome"] in ("TP", "TN") for d in details]

    def _scores(details: list[dict]) -> list[float]:
        return [d.get("score", 0.0) for d in details]

    pair_details = {nm: _details(scorer_results[nm]) for nm in names}
    outcomes     = {nm: _outcomes(pair_details[nm])   for nm in names}
    raw_scores   = {nm: _scores(pair_details[nm])     for nm in names}

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sa, sb = names[i], names[j]
            n = min(len(pair_details[sa]), len(pair_details[sb]))
            all_tests.append(mcnemar_test(outcomes[sa][:n],    outcomes[sb][:n],    sa, sb, component))
            all_tests.append(bootstrap_paired_f1(pair_details[sa][:n], pair_details[sb][:n], sa, sb, component))
            all_tests.append(permutation_test(outcomes[sa][:n], outcomes[sb][:n],   sa, sb, component))
            all_tests.append(wilcoxon_test(raw_scores[sa][:n],  raw_scores[sb][:n], sa, sb, component))

    if len(names) >= 3:
        all_tests.append(friedman_test(pair_details, component))

    # Summary
    def _f1(sr: Any) -> float:
        return sr.f1 if hasattr(sr, "f1") else sr.get("f1", 0.0)

    def _acc(sr: Any) -> float:
        return sr.accuracy if hasattr(sr, "accuracy") else sr.get("accuracy", 0.0)

    summary = {
        "component": component,
        "total_tests": len(all_tests),
        "significant": sum(1 for t in all_tests if t.significant),
        "best_by_f1":       max(names, key=lambda nm: _f1(scorer_results[nm])),
        "best_by_accuracy": max(names, key=lambda nm: _acc(scorer_results[nm])),
        "metrics": {
            nm: {
                "f1":       round(_f1(scorer_results[nm]),  4),
                "accuracy": round(_acc(scorer_results[nm]), 4),
            }
            for nm in names
        },
    }

    return HypothesisReport(component=component, tests=all_tests, summary=summary)


# ---------------------------------------------------------------------------
# Maths helpers
# ---------------------------------------------------------------------------

def _chi2_sf(x: float, df: int) -> float:
    if x <= 0:
        return 1.0
    if df >= 30:
        z = ((x / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
        return _normal_sf(z)
    return max(0.0, math.gamma(df / 2) - _lower_incomplete_gamma(df / 2, x / 2)) / math.gamma(df / 2)


def _lower_incomplete_gamma(s: float, x: float) -> float:
    term = x ** s * math.exp(-x) / s
    total = term
    for k in range(1, 300):
        term *= x / (s + k)
        total += term
        if abs(term) < 1e-12:
            break
    return total


def _normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))
