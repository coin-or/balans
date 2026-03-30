import json
import os
import time
from typing import NamedTuple, Optional

import mabwiser.utils
from alns.accept import (LateAcceptanceHillClimbing, NonLinearGreatDeluge, AlwaysAccept,
                          MovingAverageThreshold, GreatDeluge, HillClimbing,
                          RecordToRecordTravel, SimulatedAnnealing, RandomAccept)
from alns.select import AlphaUCB, MABSelector, RandomSelect, RouletteWheel, SegmentedRouletteWheel
from alns.stop import MaxIterations, MaxRuntime, NoImprovement
from mabwiser.mab import LearningPolicy


class Constants(NamedTuple):
    """
    Constant values used by the modules.
    """

    # Default seed
    default_seed = 1283

    # Default number of parallel MIP jobs
    default_n_mip_jobs = 1

    # Default MIP Solver
    scip_solver = "scip"
    gurobi_solver = "gurobi"
    highs_solver = "highs"
    default_solver = scip_solver

    # Optimization sense
    minimize = "minimize"
    maximize = "maximize"

    # Scip variable types
    binary = "BINARY"
    integer = "INTEGER"
    continuous = "CONTINUOUS"

    # Column names for features df
    var_type = "var_type"
    var_lb = "var_lb"
    var_ub = "var_ub"

    # Time limit for the initial solution to get feasible solution as a starting point for ALNS
    timelimit_first_solution = 20

    # Time limit for finding a random feasible solution in crossover operator
    timelimit_crossover_random_feasible = 20

    # time limit for one iteration is ALNS, local branching has longer time because hard problem created
    timelimit_alns_iteration = 60

    # time limit for one local branching iteration.
    # paper says Each LNS iteration is limited to 1 minute, except for Local Branching with 2.5 minutes.
    timelimit_local_branching_iteration = 150

    # for Big-M constraint, currently used in Proximity
    M = 1000

    # Default output directory for ParBalans results
    default_parbalans_output_dir = "parbalans_results/"

    # Folder names
    _TEST_DIR_NAME = "tests"
    _DATA_DIR_NAME = "data"
    _TEST_DATA_DIR_NAME = _TEST_DIR_NAME + os.sep + _DATA_DIR_NAME
    _CONFIGS_DIR_NAME = "configs"
    _TOP_CONFIGS_DIR_NAME = "top_configs"

    # Folder paths
    _FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_TEST = _FILE_DIR + os.sep + ".." + os.sep + _TEST_DATA_DIR_NAME
    CONFIGS = _FILE_DIR + os.sep + _CONFIGS_DIR_NAME
    TOP_CONFIGS = CONFIGS + os.sep + _TOP_CONFIGS_DIR_NAME


# Global start time for elapsed time tracking, set when Balans.solve() is called
_solve_start_time: float = 0.0

# Global wall-clock deadline (absolute time). None means no deadline.
_solve_deadline: Optional[float] = None


def set_solve_start_time():
    """Set the global solve start time to now."""
    global _solve_start_time
    _solve_start_time = time.time()


def set_solve_deadline(max_runtime: float = None):
    """Set a global wall-clock deadline for the entire Balans.solve() call.

    Parameters
    ----------
    max_runtime : float or None
        If given, the deadline is set to now + max_runtime seconds.
        If None, no deadline is enforced.
    """
    global _solve_deadline
    if max_runtime is not None:
        _solve_deadline = time.time() + max_runtime
    else:
        _solve_deadline = None


def remaining_time() -> float:
    """Return seconds remaining until the global deadline.

    Returns float('inf') if no deadline has been set.
    """
    if _solve_deadline is None:
        return float('inf')
    return max(0.0, _solve_deadline - time.time())


def cap_timelimit(requested: float) -> float:
    """Return min(requested, remaining_time()), ensuring the MIP solver
    never exceeds the global wall-clock budget."""
    return min(requested, remaining_time())


def timestamp() -> str:
    """Return a formatted string showing elapsed time since solve started."""
    elapsed = time.time() - _solve_start_time
    return f"[{elapsed:8.2f}s]"



def create_rng(seed):
    return mabwiser.utils.create_rng(seed)


def check_false(expression: bool, exception: Exception):
    return mabwiser.utils.check_false(expression, exception)


def check_true(expression: bool, exception: Exception):
    return mabwiser.utils.check_true(expression, exception)


# ---------------------------------------------------------------------------
# ConfigFactory: builds Balans configuration objects from JSON config files
# ---------------------------------------------------------------------------
class ConfigFactory:
    """Factory for building Balans configuration from JSON config files.

    Handles parsing JSON configs,
    building ALNS selector / acceptance / stop objects, and resolving operator names.
    Operator name-to-function resolution requires an operator map that lives in solver.py (to avoid circular imports).
    """

    # Default config file shipped with the package (lives inside balans/configs/)
    DEFAULT_CONFIG_PATH = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'configs', 'default.json'))

    # ----- Learning Policy -----
    @staticmethod
    def build_learning_policy(lp_config: dict):
        """Build a MABWiser LearningPolicy from a config dict."""
        lp_type = lp_config["type"]
        if lp_type == "ThompsonSampling":
            return LearningPolicy.ThompsonSampling()
        elif lp_type == "EpsilonGreedy":
            return LearningPolicy.EpsilonGreedy(epsilon=lp_config.get("epsilon"))
        elif lp_type == "Softmax":
            return LearningPolicy.Softmax(tau=lp_config.get("tau"))
        elif lp_type == "UCB1":
            return LearningPolicy.UCB1(alpha=lp_config.get("alpha"))
        else:
            raise ValueError(f"Unknown learning policy type in config.json: {lp_type}")

    # ----- Selector -----
    @staticmethod
    def build_selector(sel_config: dict, num_destroy: int, num_repair: int):
        """Build an ALNS selector from a config dict."""
        sel_type = sel_config["type"]
        if sel_type == "MABSelector":
            lp = ConfigFactory.build_learning_policy(sel_config["learning_policy"])
            return MABSelector(scores=sel_config["scores"],
                               num_destroy=num_destroy,
                               num_repair=num_repair,
                               learning_policy=lp)
        elif sel_type == "RouletteWheel":
            return RouletteWheel(scores=sel_config["scores"],
                                 num_destroy=num_destroy,
                                 num_repair=num_repair,
                                 decay=sel_config.get("decay"))
        elif sel_type == "RandomSelect":
            return RandomSelect(num_destroy=num_destroy, num_repair=num_repair)
        elif sel_type == "AlphaUCB":
            return AlphaUCB(scores=sel_config["scores"],
                            num_destroy=num_destroy,
                            num_repair=num_repair,
                            alpha=sel_config.get("alpha"))
        else:
            raise ValueError(f"Unknown selector type config.json: {sel_type}")

    # ----- Acceptance -----
    @staticmethod
    def build_acceptance(acc_config: dict):
        """Build an ALNS acceptance criterion from a config dict."""
        acc_type = acc_config["type"]
        if acc_type == "SimulatedAnnealing":
            kwargs = {k: acc_config[k] for k in
                      ("start_temperature", "end_temperature", "step", "method")
                      if k in acc_config}
            return SimulatedAnnealing(**kwargs)
        elif acc_type == "HillClimbing":
            return HillClimbing()
        elif acc_type == "RecordToRecordTravel":
            kwargs = {k: acc_config[k] for k in
                      ("start_threshold", "end_threshold", "step", "method")
                      if k in acc_config}
            return RecordToRecordTravel(**kwargs)
        elif acc_type == "GreatDeluge":
            kwargs = {k: acc_config[k] for k in
                      ("alpha", "beta")
                      if k in acc_config}
            return GreatDeluge(**kwargs)
        elif acc_type == "RandomAccept":
            kwargs = {k: acc_config[k] for k in
                      ("start_prob", "end_prob", "step")
                      if k in acc_config}
            return RandomAccept(**kwargs)
        elif acc_type == "AlwaysAccept":
            return AlwaysAccept()
        else:
            raise ValueError(f"Unknown acceptance type config.json: {acc_type}")

    # ----- Stop -----
    @staticmethod
    def build_stop(stop_config: dict):
        """Build an ALNS stopping criterion from a config dict."""
        stop_type = stop_config["type"]
        if stop_type == "MaxIterations":
            return MaxIterations(stop_config.get("max_iterations"))
        elif stop_type == "MaxRuntime":
            return MaxRuntime(stop_config.get("max_runtime"))
        elif stop_type == "NoImprovement":
            return NoImprovement(stop_config.get("max_iterations"))
        else:
            raise ValueError(f"Unknown stop type config.json: {stop_type}")

    # ----- Operator resolution -----
    @staticmethod
    def resolve_operators(operator_names: list, operator_map: dict, kind: str = "operator") -> list:
        """Resolve a list of operator name strings to function references using the given map."""
        resolved = []
        for name in operator_names:
            if name not in operator_map:
                raise ValueError(f"Unknown {kind}: '{name}'. "
                                 f"Available: {sorted(operator_map.keys())}")
            resolved.append(operator_map[name])
        return resolved

    # ----- Load -----

    # All top-level keys that Balans.__init__ can consume from a config file.
    # Mirrors the constructor parameters; M maps to big_m.
    KNOWN_CONFIG_KEYS = frozenset({
        'seed',
        'n_mip_jobs',
        'mip_solver',
        'destroy_ops',
        'repair_ops',
        'selector',
        'accept',
        'stop',
        'timelimit_first_solution',
        'timelimit_alns_iteration',
        'timelimit_local_branching_iteration',
        'timelimit_crossover_random_feasible',
        'big_m',
    })

    @staticmethod
    def load(config_path: str) -> dict:
        """Load a Balans JSON config file and return its contents as-is.

        Returns the raw JSON dict. Keys present in the file are returned with
        their values; keys absent from the file are simply not in the dict.
        No defaults are applied here — all fallback logic lives in Balans.__init__.

        Raises
        ------
        ValueError
            If the config file contains any unrecognised top-level key.
        """
        with open(config_path, 'r') as f:
            cfg = json.load(f)

        unknown = set(cfg.keys()) - ConfigFactory.KNOWN_CONFIG_KEYS
        if unknown:
            raise ValueError(
                f"Unknown config key(s): {sorted(unknown)}. "
                f"Known keys: {sorted(ConfigFactory.KNOWN_CONFIG_KEYS)}")

        return cfg

