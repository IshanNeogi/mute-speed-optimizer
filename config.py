"""Single config object every module reads from -- loaded from config/default_config.yaml (or
any YAML file with the same shape). Nothing in this package hardcodes a ship/weight/path value
outside of here; change the YAML, not the code, to retune a run.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from tl_model.tl_model import TLModelConfig


@dataclass
class ShipConfig:
    ship_class: str = "Other"
    ship_length_ft: float = 697.18
    min_speed_kn: float = 8.0
    max_speed_kn: float = 14.5
    acceleration_limit_kn_per_hr: float = 9.0


@dataclass
class OptimizerConfig:
    n_waypoints: int = 30
    w_time: float = 20.0
    sel_max_weight: float = 0.5
    pop_size: int = 32
    n_gen: int = 200
    n_seeds: int = 5
    use_hard_mammal_constraint: bool = True   # the validated contribution; False = old unconstrained formulation
    constraint_voyage_time_hr: Optional[float] = None   # the "mid_speed" heuristic/constraint
                                                   # floor, expressed as a voyage TIME (hours) for
                                                   # a constant-speed transit of the route, not a
                                                   # speed directly -- the constant speed implied
                                                   # by this time is computed in the background
                                                   # (route_distance_nm / this value). None
                                                   # (default) = the echo_ais recorded voyage's own
                                                   # time, i.e. "don't let the optimizer make any
                                                   # mammal worse off than the ship taking as long
                                                   # as the real recorded voyage, at constant
                                                   # speed." The arithmetic-midpoint-speed default
                                                   # used before this field existed is NOT what
                                                   # None means now -- set an explicit value if you
                                                   # need that old behavior (~11.25kn's implied
                                                   # time on the shipped baseline_route.csv).
    mutation_initial_rate: float = 0.3
    mutation_final_rate: float = 0.01
    crossover_initial_rate: float = 0.95
    crossover_final_rate: float = 0.7
    seed_population_shape: str = "min_speed"  # "min_speed" (validated standard) or "baseline"


@dataclass
class DataPaths:
    baseline_route_csv: str = "data/baseline_route.csv"
    mammal_paths_csv: str = "data/mammal_paths.csv"


@dataclass
class RunConfig:
    ship: ShipConfig = field(default_factory=ShipConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    tl_model: TLModelConfig = field(default_factory=TLModelConfig)
    data: DataPaths = field(default_factory=DataPaths)
    mammal_ids: list = field(default_factory=lambda: [3, 12, 8])
    output_dir: str = "results"


def config_to_dict(cfg: RunConfig) -> dict:
    """Full, flat, JSON-serializable dump -- everything needed to deterministically reproduce a
    run's forward evaluation (route/mammal data are referenced by path, not embedded; they're
    already fixed CSVs under data/, see data/README.md for provenance)."""
    return asdict(cfg)


def load_config(path: str | Path) -> RunConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    ship = ShipConfig(**raw.get("ship", {}))
    optimizer = OptimizerConfig(**raw.get("optimizer", {}))
    tl_raw = raw.get("tl_model", {})
    # YAML null (or the key missing) falls back to these env vars -- lets a fresh download run
    # every example config unedited, by setting the path once, instead of hand-editing every YAML
    # file that references a TL model. YAML value always wins if explicitly set.
    light_dir = tl_raw.get("light_model_dir") or os.environ.get("MUTE_LIGHT_MODEL_DIR")
    full_dir = tl_raw.get("full_model_dir") or os.environ.get("MUTE_FULL_MODEL_DIR")
    tl_model = TLModelConfig(
        kind=tl_raw.get("kind", "light"),
        light_model_dir=light_dir,
        full_model_dir=full_dir,
        full_n_components=tl_raw.get("full_n_components", 10),
    )
    data = DataPaths(**raw.get("data", {}))

    return RunConfig(
        ship=ship, optimizer=optimizer, tl_model=tl_model, data=data,
        mammal_ids=raw.get("mammal_ids", [3, 12, 8]),
        output_dir=raw.get("output_dir", "results"),
    )
