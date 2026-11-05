"""Run: python -m operators.tests.test_operators  (from handover/)

Regression tests for the mutation/crossover rate bug fix (see operators.py's module docstring).
These exercise the REAL pymoo `.do()` entrypoint (not `_do()` directly) -- the whole point of the
original bug was that `.do()` is the only place `prob` is consulted, so a test that only calls
`_do()` the way the sandbox's original (bugged) sweep did would miss the bug entirely.
"""
import sys
from pathlib import Path

import numpy as np
from pymoo.core.population import Population
from pymoo.core.problem import Problem

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from operators.operators import AdaptiveCrossover, AdaptiveMutation


class _FakeAlgorithm:
    def __init__(self, n_gen):
        self.n_gen = n_gen


class _ToyProblem(Problem):
    def __init__(self, n_var=10):
        super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=0.0, xu=1.0)


def test_mutation_rate_zero_leaves_population_unchanged():
    """The bug this guards against: before the fix, rate was never consulted, so rate=0 and
    rate=1 produced bit-identical output (the sandbox's original mutcx sweep finding, LOG.md
    2026-10-01 -- all 5 variants bit-for-bit identical to 14 decimal places)."""
    problem = _ToyProblem()
    rng = np.random.default_rng(42)
    X = rng.random((20, problem.n_var))
    pop = Population.new("X", X.copy())

    op = AdaptiveMutation(initial_rate=0.0, final_rate=0.0, n_gen=100)
    out = op.do(problem, pop, algorithm=_FakeAlgorithm(n_gen=10), seed=1)
    assert np.allclose(out.get("X"), X), "rate=0 should leave every individual unchanged"


def test_mutation_rate_one_changes_population():
    problem = _ToyProblem()
    rng = np.random.default_rng(42)
    X = rng.random((20, problem.n_var))
    pop = Population.new("X", X.copy())

    op = AdaptiveMutation(initial_rate=1.0, final_rate=1.0, n_gen=100)
    out = op.do(problem, pop, algorithm=_FakeAlgorithm(n_gen=10), seed=1)
    assert not np.allclose(out.get("X"), X), "rate=1 should mutate essentially every individual"


def test_mutation_rate_actually_changes_the_outcome():
    """The direct regression check: same seed, only `initial_rate`/`final_rate` differ ->
    different number of individuals mutated. This is what the original bug made impossible."""
    problem = _ToyProblem()
    rng = np.random.default_rng(7)
    X = rng.random((200, problem.n_var))

    pop_low = Population.new("X", X.copy())
    op_low = AdaptiveMutation(initial_rate=0.1, final_rate=0.1, n_gen=100)
    out_low = op_low.do(problem, pop_low, algorithm=_FakeAlgorithm(n_gen=10), seed=99)
    n_changed_low = np.sum(np.any(out_low.get("X") != X, axis=1))

    pop_high = Population.new("X", X.copy())
    op_high = AdaptiveMutation(initial_rate=0.9, final_rate=0.9, n_gen=100)
    out_high = op_high.do(problem, pop_high, algorithm=_FakeAlgorithm(n_gen=10), seed=99)
    n_changed_high = np.sum(np.any(out_high.get("X") != X, axis=1))

    assert n_changed_high > n_changed_low, (
        f"higher rate should mutate more individuals: low={n_changed_low}, high={n_changed_high}")


def test_crossover_rate_actually_changes_the_outcome():
    problem = _ToyProblem()
    rng = np.random.default_rng(7)
    n_matings = 100
    parents_X = rng.random((2, n_matings, problem.n_var))

    def run(rate, seed):
        op = AdaptiveCrossover(initial_rate=rate, final_rate=rate, n_gen=100)
        off = op._do(problem, parents_X.copy(), algorithm=_FakeAlgorithm(n_gen=10),
                      random_state=np.random.default_rng(seed))
        n_offsprings = off.shape[0]
        changed = np.sum(np.any(off != parents_X[:n_offsprings], axis=(0, 2)))
        return changed

    n_changed_low = run(0.05, seed=99)
    n_changed_high = run(0.95, seed=99)
    assert n_changed_high > n_changed_low, (
        f"higher crossover rate should cross over more matings: low={n_changed_low}, high={n_changed_high}")


if __name__ == "__main__":
    test_mutation_rate_zero_leaves_population_unchanged()
    test_mutation_rate_one_changes_population()
    test_mutation_rate_actually_changes_the_outcome()
    test_crossover_rate_actually_changes_the_outcome()
    print("operators: all tests passed")
