"""Adaptive mutation/crossover rate schedules -- anneal linearly from a high initial rate
(exploration) to a low final rate (exploitation) over the run.

**Bug fix vs. ../MUTE / production** (`backend/app/optimizer/route_pymoo_config.py` and
`speed_pymoo_config.py` have the identical pattern, unfixed there per the scientific-core
sign-off rule -- flagged in sandbox/optimizer/LOG.md, 2026-10-01): the original builds a fresh
`PolynomialMutation(prob=rate, ...)` / `SBX(prob=rate, ...)` each generation and calls its
`._do()` directly. Pymoo's `Mutation.do()`/`Crossover.do()` — NOT `_do()` — is the only place
`prob` is ever consulted (it draws the per-individual/per-mating Bernoulli mask deciding whether
the operator's output is actually used, vs. passing the original/parent value through
unchanged). Calling `._do()` directly means the computed `rate` is constructed into the operator
but never read, and the operator's output is used unconditionally for every individual, every
generation. `eta` (held fixed) was the only thing that could ever differ between configurations
with this bug.

Verified (inspecting pymoo's installed `core/mutation.py` / `core/crossover.py` directly, this
session): confirms the diagnosis exactly. Fix here: replicate the same Bernoulli-masking logic
`do()` performs, inline in our own `_do()`, instead of delegating to the inner operator's `_do()`
unconditionally. See tests/test_operators.py for the regression test this fix makes pass (and
that would have caught the original bug by construction).
"""
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation


def _anneal_rate(initial_rate, final_rate, current_gen, n_gen):
    frac = min(1.0, current_gen / n_gen) if n_gen > 0 else 1.0
    return initial_rate + (final_rate - initial_rate) * frac


class AdaptiveMutation(Mutation):
    def __init__(self, initial_rate, final_rate, n_gen):
        super().__init__()
        self.initial_rate = initial_rate
        self.final_rate = final_rate
        self.n_gen = n_gen

    def _do(self, problem, X, *args, random_state=None, **kwargs):
        current_gen = kwargs["algorithm"].n_gen
        rate = _anneal_rate(self.initial_rate, self.final_rate, current_gen, self.n_gen)
        mutation = PolynomialMutation(eta=10)   # prob applied below, not via this instance's .do()
        Xp = mutation._do(problem, X, *args, random_state=random_state, **kwargs)

        n = X.shape[0]
        mask = random_state.random(n) <= rate
        out = X.copy()
        out[mask] = Xp[mask]
        return out


class AdaptiveCrossover(Crossover):
    def __init__(self, initial_rate, final_rate, n_gen):
        super().__init__(2, 2)
        self.initial_rate = initial_rate
        self.final_rate = final_rate
        self.n_gen = n_gen

    def _do(self, problem, X, *args, random_state=None, **kwargs):
        current_gen = kwargs["algorithm"].n_gen
        rate = _anneal_rate(self.initial_rate, self.final_rate, current_gen, self.n_gen)
        crossover = SBX(eta=15)   # prob applied below, not via this instance's .do()
        Q = crossover._do(problem, X, *args, random_state=random_state, **kwargs)

        n_offsprings, n_matings, n_var = Q.shape
        mask = random_state.random(n_matings) <= rate
        out = Q.copy()
        # matings that DON'T cross over: pass the first n_offsprings parents straight through,
        # same fallback pymoo's own Crossover.do() uses
        for k in range(n_matings):
            if not mask[k]:
                out[:, k, :] = X[:n_offsprings, k, :]
        return out
