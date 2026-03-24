import json
import os
import time
from typing import NamedTuple

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

    # Time limit for finding a random feasible solution
    timelimit_random_feasible = 20

    # time limit for one iteration is ALNS, local branching has longer time because hard problem created
    timelimit_alns_iteration = 60

    # time limit for one local branching iteration.
    # paper says Each LNS iteration is limited to 1 minute, except for Local Branching with 2.5 minutes.
    timelimit_local_branching_iteration = 150

    # for Big-M constraint, currently used in Proximity
    M = 1000

    # Data folder constants
    _TEST_DIR_NAME = "tests"
    _DATA_DIR_NAME = "data"
    _TEST_DATA_DIR_NAME = _TEST_DIR_NAME + os.sep + _DATA_DIR_NAME

    # Data paths
    _FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_TEST = _FILE_DIR + os.sep + ".." + os.sep + _TEST_DATA_DIR_NAME



# Global start time for elapsed time tracking, set when Balans.solve() is called
_solve_start_time: float = 0.0


def set_solve_start_time():
    """Set the global solve start time to now."""
    global _solve_start_time
    _solve_start_time = time.time()


def timestamp() -> str:
    """Return a formatted string showing elapsed time since solve started."""
    elapsed = time.time() - _solve_start_time
    return f"[{elapsed:8.2f}s]"


def update_constants(**kwargs):
    """Override Constants class attributes with values from a config file."""
    for key, value in kwargs.items():
        if hasattr(Constants, key):
            setattr(Constants, key, value)
        else:
            raise ValueError(f"Unknown constant: {key}")


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

    Handles parsing JSON configs, building ALNS selector / acceptance / stop
    objects, and resolving operator names. Operator name-to-function resolution
    requires an operator map that lives in solver.py (to avoid circular imports).
    """

    # Default config file shipped with the package
    DEFAULT_CONFIG_PATH = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'configs', 'default.json'))

    # ----- Learning Policy -----
    @staticmethod
    def build_learning_policy(lp_config: dict):
        """Build a MABWiser LearningPolicy from a config dict."""
        lp_type = lp_config["type"]
        if lp_type == "ThompsonSampling":
            return LearningPolicy.ThompsonSampling()
        elif lp_type == "EpsilonGreedy":
            return LearningPolicy.EpsilonGreedy(epsilon=lp_config.get("epsilon", 0.1))
        elif lp_type == "Softmax":
            return LearningPolicy.Softmax(tau=lp_config.get("tau", 1.0))
        elif lp_type == "UCB1":
            return LearningPolicy.UCB1(alpha=lp_config.get("alpha", 1.0))
        else:
            raise ValueError(f"Unknown learning policy type: {lp_type}")

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
                                 decay=sel_config.get("decay", 0.5))
        elif sel_type == "RandomSelect":
            return RandomSelect(num_destroy=num_destroy, num_repair=num_repair)
        elif sel_type == "AlphaUCB":
            return AlphaUCB(scores=sel_config["scores"],
                            num_destroy=num_destroy,
                            num_repair=num_repair,
                            alpha=sel_config.get("alpha", 1.0))
        else:
            raise ValueError(f"Unknown selector type: {sel_type}")

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
            raise ValueError(f"Unknown acceptance type: {acc_type}")

    # ----- Stop -----
    @staticmethod
    def build_stop(stop_config: dict):
        """Build an ALNS stopping criterion from a config dict."""
        stop_type = stop_config["type"]
        if stop_type == "MaxIterations":
            return MaxIterations(stop_config.get("max_iterations", 10))
        elif stop_type == "MaxRuntime":
            return MaxRuntime(stop_config.get("max_runtime", 100))
        elif stop_type == "NoImprovement":
            return NoImprovement(stop_config.get("max_iterations", 100))
        else:
            raise ValueError(f"Unknown stop type: {stop_type}")

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
    @staticmethod
    def load(config_path: str) -> dict:
        """Load a Balans JSON config file and return a configuration dict.

        The returned dict has keys:
            destroy_operator_names, repair_operator_names  (lists of strings)
            selector_config   (raw dict, not yet built — depends on num_destroy/num_repair)
            accept            (built AcceptType object)
            stop              (built StopType object)
            seed, n_mip_jobs, mip_solver   (scalars)
            constants         (dict or None)

        Operator names are left as strings. Use resolve_operators() with the
        appropriate operator map (defined in solver.py) to convert them to
        function references.
        """
        with open(config_path, 'r') as f:
            cfg = json.load(f)

        result = {}

        # Simple scalar params
        result['seed'] = cfg.get('seed', Constants.default_seed)
        result['n_mip_jobs'] = cfg.get('n_mip_jobs', 1)
        result['mip_solver'] = cfg.get('mip_solver', Constants.default_solver)

        # Operators: keep as string names (resolved later with operator maps)
        result['destroy_operator_names'] = cfg.get('destroy_operators', [])
        result['repair_operator_names'] = cfg.get('repair_operators', ['Repair'])

        # Selector: keep raw config dict (built later with correct num_destroy/num_repair)
        result['selector_config'] = cfg.get('selector', None)

        # Acceptance: build object
        acc_cfg = cfg.get('acceptance', {"type": "SimulatedAnnealing",
                                         "start_temperature": 20,
                                         "end_temperature": 1,
                                         "step": 0.1})
        result['accept'] = ConfigFactory.build_acceptance(acc_cfg)

        # Stop: build object
        stop_cfg = cfg.get('stop', {"type": "MaxIterations", "max_iterations": 10})
        result['stop'] = ConfigFactory.build_stop(stop_cfg)

        # Constants overrides
        result['constants'] = cfg.get('constants', None)

        return result

