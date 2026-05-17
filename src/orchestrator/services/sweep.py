"""Sweep-combo derivation.

Two strategies are supported:

* ``grid`` (default, original) — derives the next combo from the
  cross-product of ``params[i].values`` at ``iter_index``. Pure function;
  deterministic; matches the behaviour of ``research-tick.sh``.

* ``tpe`` (added 2026-05-05) — Tree-structured Parzen Estimator via
  Optuna. Asks the sampler for the next point given the history of past
  ``(params, pf)`` trials on the same queue's axis-set. Sample-efficient
  alternative to grid for high-dimensional or fine-resolution searches.

Order is ``itertools.product`` left-to-right (first parameter cycles
slowest) for grid. The bash script and the agent's queue-strategy helper
both assume this order; do not change it.
"""

from __future__ import annotations

from itertools import product
from typing import Any


def derive_combo(sweep_config: dict[str, Any], iter_index: int) -> dict[str, Any] | None:
    """Return the combo for ``iter_index`` (0-based), or None if exhausted.

    Only valid for ``strategy=='grid'`` configs. TPE sweeps go through
    ``next_combo_tpe`` instead.
    """
    params = sweep_config.get("params", [])
    if not params:
        return {} if iter_index == 0 else None
    keys = [p["name"] for p in params]
    values = [p["values"] for p in params]
    combos = list(product(*values))
    if iter_index < 0 or iter_index >= len(combos):
        return None
    return dict(zip(keys, combos[iter_index], strict=True))


def total_combos(sweep_config: dict[str, Any]) -> int:
    """Grid only — total cells in the cross-product."""
    params = sweep_config.get("params", [])
    if not params:
        return 1
    n = 1
    for p in params:
        n *= len(p["values"])
    return n


def is_tpe(sweep_config: dict[str, Any]) -> bool:
    return sweep_config.get("strategy") == "tpe"


def tpe_trial_budget(sweep_config: dict[str, Any]) -> int:
    """How many trials this TPE sweep is allowed before exhausting.

    ``n_trials`` is the budget; default 20 if absent.
    """
    return int(sweep_config.get("n_trials", 20))


def _build_distributions(params: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert param specs to Optuna distributions. Imported lazily so
    Optuna is only required when TPE is actually used (keeps test imports
    cheap on grid-only paths)."""
    import optuna.distributions as od

    out: dict[str, Any] = {}
    for p in params:
        name = p["name"]
        ptype = p.get("type", "float")
        if ptype == "choice":
            vals = p.get("values") or []
            if not vals:
                raise ValueError(f"tpe choice param {name!r} has no values")
            out[name] = od.CategoricalDistribution(vals)
        elif ptype == "int":
            lo = int(float(p["low"]))
            hi = int(float(p["high"]))
            if hi < lo:
                raise ValueError(f"tpe int param {name!r}: low > high")
            out[name] = od.IntDistribution(lo, hi)
        else:  # float
            lo_f = float(p["low"])
            hi_f = float(p["high"])
            if hi_f < lo_f:
                raise ValueError(f"tpe float param {name!r}: low > high")
            out[name] = od.FloatDistribution(lo_f, hi_f)
    return out


def _coerce_param_value(name: str, raw: Any, distribution: Any) -> Any:
    """Coerce a JSONB-loaded param value to the distribution's expected
    Python type. Past trials were written by the JVM as strings; Optuna
    wants float/int/exact-categorical."""
    import optuna.distributions as od

    if isinstance(distribution, od.FloatDistribution):
        return float(raw)
    if isinstance(distribution, od.IntDistribution):
        return int(float(raw))
    if isinstance(distribution, od.CategoricalDistribution):
        # Try exact match first; fall back to string comparison.
        if raw in distribution.choices:
            return raw
        for choice in distribution.choices:
            if str(choice) == str(raw):
                return choice
        raise ValueError(
            f"tpe categorical param {name!r} value {raw!r} not in choices "
            f"{list(distribution.choices)}"
        )
    raise TypeError(f"unknown distribution for {name!r}: {type(distribution)!r}")


def next_combo_tpe(
    sweep_config: dict[str, Any],
    past_trials: list[tuple[dict[str, Any], float | None]],
    iter_index: int,
    seed: int | None = None,
) -> dict[str, str] | None:
    """Ask Optuna's TPE sampler for the next combo.

    Args:
        sweep_config: must have ``strategy == "tpe"`` and ``params`` with
            per-param ``type``/``low``/``high``/``values``.
        past_trials: history of ``(params, pf_or_None)`` for prior
            iterations on the same axis set. ``None`` value marks a
            failed trial — added as PRUNED so the sampler learns to
            avoid that region rather than re-suggesting it. Empty list
            → random first draw.
        iter_index: 0-based iteration counter on this queue. Returns
            None when ``iter_index >= n_trials`` (budget exhausted).
        seed: deterministic seed; falls back to ``sweep_config["seed"]``
            then 42.

    Returns the next combo with values stringified (BigDecimal-as-string
    matching the JVM convention). Categorical values are stringified too.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_trials = tpe_trial_budget(sweep_config)
    if iter_index < 0 or iter_index >= n_trials:
        return None

    params = sweep_config.get("params", [])
    if not params:
        return {} if iter_index == 0 else None

    distributions = _build_distributions(params)
    effective_seed = seed if seed is not None else int(sweep_config.get("seed", 42))
    sampler = optuna.samplers.TPESampler(seed=effective_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Replay past trials. Skip rows whose params are missing keys we now
    # care about (axis-set drift) — better to lose history than crash.
    # Failed trials (value=None) are added as PRUNED so TPE learns to
    # avoid the region; completed trials carry their objective value.
    for trial_params, value in past_trials:
        try:
            coerced = {
                name: _coerce_param_value(name, trial_params[name], distributions[name])
                for name in distributions
            }
        except (KeyError, ValueError):
            continue
        if value is None:
            trial = optuna.trial.create_trial(
                params=coerced,
                distributions=distributions,
                state=optuna.trial.TrialState.PRUNED,
            )
        else:
            trial = optuna.trial.create_trial(
                params=coerced,
                distributions=distributions,
                value=float(value),
            )
        study.add_trial(trial)

    trial = study.ask(distributions)
    return {k: str(v) for k, v in trial.params.items()}
