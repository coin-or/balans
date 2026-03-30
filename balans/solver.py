import glob
import os
import random
from typing import List, Optional, Dict, Tuple
from typing import NamedTuple
import pickle
from multiprocessing import Pool

import numpy as np
from alns.ALNS import ALNS
from alns.Result import Result
from alns.accept import LateAcceptanceHillClimbing, NonLinearGreatDeluge, AlwaysAccept
from alns.accept import MovingAverageThreshold, GreatDeluge, HillClimbing
from alns.accept import RecordToRecordTravel, SimulatedAnnealing, RandomAccept
from alns.select import AlphaUCB, MABSelector, RandomSelect, RouletteWheel, SegmentedRouletteWheel
from alns.stop import MaxIterations, MaxRuntime, NoImprovement, StoppingCriterion
from mabwiser.mab import LearningPolicy

from balans.base_instance import _Instance
from balans.base_mip import create_mip_solver
from balans.base_state import _State
from balans.destroy.crossover import crossover
from balans.destroy.dins import dins
from balans.destroy.local_branching import local_branching_05, local_branching_10, local_branching_15, \
    local_branching_20, local_branching_25, local_branching_30, local_branching_35, local_branching_40, \
    local_branching_45, local_branching_50, local_branching_55, local_branching_60, local_branching_65, \
    local_branching_70, local_branching_75, local_branching_80, local_branching_85, local_branching_90, \
    local_branching_95
from balans.destroy.mutation import mutation_05, mutation_10, mutation_15, mutation_20, mutation_25, mutation_30, \
    mutation_35, mutation_40, mutation_45, mutation_50, mutation_55, mutation_60, mutation_65, mutation_70, mutation_75, \
    mutation_80, mutation_85, mutation_90, mutation_95
from balans.destroy.proximity import proximity_005, proximity_010, proximity_015, proximity_020, proximity_025, \
    proximity_030, proximity_035, proximity_040, proximity_045, proximity_05, proximity_055, proximity_060, \
    proximity_065, \
    proximity_070, proximity_075, proximity_080, proximity_085, proximity_090, proximity_095, proximity_10
from balans.destroy.random_objective import random_objective
from balans.destroy.rens import rens_05, rens_10, rens_15, rens_20, rens_25, rens_30, rens_35, rens_40, rens_45, \
    rens_50, rens_55, rens_60, rens_65, rens_70, rens_75, rens_80, rens_85, rens_90, rens_95
from balans.destroy.rins import rins_05, rins_10, rins_15, rins_20, rins_25, rins_30, rins_35, rins_40, rins_45, \
    rins_50, rins_55, rins_60, rins_65, rins_70, rins_75, rins_80, rins_85, rins_90, rins_95
from balans.repair.repair import repair
from balans.utils import Constants, ConfigFactory, check_false, check_true, create_rng, set_solve_start_time, \
    set_solve_deadline, timestamp


class DestroyOperators(NamedTuple):
    Crossover = crossover

    Dins = dins

    Local_Branching_05 = local_branching_05
    Local_Branching_10 = local_branching_10
    Local_Branching_15 = local_branching_15
    Local_Branching_20 = local_branching_20
    Local_Branching_25 = local_branching_25
    Local_Branching_30 = local_branching_30
    Local_Branching_35 = local_branching_35
    Local_Branching_40 = local_branching_40
    Local_Branching_45 = local_branching_45
    Local_Branching_50 = local_branching_50
    Local_Branching_55 = local_branching_55
    Local_Branching_60 = local_branching_60
    Local_Branching_65 = local_branching_65
    Local_Branching_70 = local_branching_70
    Local_Branching_75 = local_branching_75
    Local_Branching_80 = local_branching_80
    Local_Branching_85 = local_branching_85
    Local_Branching_90 = local_branching_90
    Local_Branching_95 = local_branching_95

    Mutation_05 = mutation_05
    Mutation_10 = mutation_10
    Mutation_15 = mutation_15
    Mutation_20 = mutation_20
    Mutation_25 = mutation_25
    Mutation_30 = mutation_30
    Mutation_35 = mutation_35
    Mutation_40 = mutation_40
    Mutation_45 = mutation_45
    Mutation_50 = mutation_50
    Mutation_55 = mutation_55
    Mutation_60 = mutation_60
    Mutation_65 = mutation_65
    Mutation_70 = mutation_70
    Mutation_75 = mutation_75
    Mutation_80 = mutation_80
    Mutation_85 = mutation_85
    Mutation_90 = mutation_90
    Mutation_95 = mutation_95

    Proximity_005 = proximity_005
    Proximity_010 = proximity_010
    Proximity_015 = proximity_015
    Proximity_020 = proximity_020
    Proximity_025 = proximity_025
    Proximity_030 = proximity_030
    Proximity_035 = proximity_035
    Proximity_040 = proximity_040
    Proximity_045 = proximity_045
    Proximity_05 = proximity_05
    Proximity_055 = proximity_055
    Proximity_060 = proximity_060
    Proximity_065 = proximity_065
    Proximity_070 = proximity_070
    Proximity_075 = proximity_075
    Proximity_080 = proximity_080
    Proximity_085 = proximity_085
    Proximity_090 = proximity_090
    Proximity_095 = proximity_095
    Proximity_10 = proximity_10

    Rens_05 = rens_05
    Rens_10 = rens_10
    Rens_15 = rens_15
    Rens_20 = rens_20
    Rens_25 = rens_25
    Rens_30 = rens_30
    Rens_35 = rens_35
    Rens_40 = rens_40
    Rens_45 = rens_45
    Rens_50 = rens_50
    Rens_55 = rens_55
    Rens_60 = rens_60
    Rens_65 = rens_65
    Rens_70 = rens_70
    Rens_75 = rens_75
    Rens_80 = rens_80
    Rens_85 = rens_85
    Rens_90 = rens_90
    Rens_95 = rens_95

    Rins_05 = rins_05
    Rins_10 = rins_10
    Rins_15 = rins_15
    Rins_20 = rins_20
    Rins_25 = rins_25
    Rins_30 = rins_30
    Rins_35 = rins_35
    Rins_40 = rins_40
    Rins_45 = rins_45
    Rins_50 = rins_50
    Rins_55 = rins_55
    Rins_60 = rins_60
    Rins_65 = rins_65
    Rins_70 = rins_70
    Rins_75 = rins_75
    Rins_80 = rins_80
    Rins_85 = rins_85
    Rins_90 = rins_90
    Rins_95 = rins_95

    Random_Objective = random_objective


class RepairOperators(NamedTuple):
    Repair = repair


# Type Declarations
DestroyType = (type(DestroyOperators.Crossover),
               type(DestroyOperators.Dins),
               type(DestroyOperators.Local_Branching_05),
               type(DestroyOperators.Local_Branching_10),
               type(DestroyOperators.Local_Branching_15),
               type(DestroyOperators.Local_Branching_20),
               type(DestroyOperators.Local_Branching_25),
               type(DestroyOperators.Local_Branching_30),
               type(DestroyOperators.Local_Branching_35),
               type(DestroyOperators.Local_Branching_40),
               type(DestroyOperators.Local_Branching_45),
               type(DestroyOperators.Local_Branching_50),
               type(DestroyOperators.Local_Branching_55),
               type(DestroyOperators.Local_Branching_60),
               type(DestroyOperators.Local_Branching_65),
               type(DestroyOperators.Local_Branching_70),
               type(DestroyOperators.Local_Branching_75),
               type(DestroyOperators.Local_Branching_80),
               type(DestroyOperators.Local_Branching_85),
               type(DestroyOperators.Local_Branching_90),
               type(DestroyOperators.Local_Branching_95),
               type(DestroyOperators.Mutation_05),
               type(DestroyOperators.Mutation_10),
               type(DestroyOperators.Mutation_15),
               type(DestroyOperators.Mutation_20),
               type(DestroyOperators.Mutation_25),
               type(DestroyOperators.Mutation_30),
               type(DestroyOperators.Mutation_35),
               type(DestroyOperators.Mutation_40),
               type(DestroyOperators.Mutation_45),
               type(DestroyOperators.Mutation_50),
               type(DestroyOperators.Mutation_55),
               type(DestroyOperators.Mutation_60),
               type(DestroyOperators.Mutation_65),
               type(DestroyOperators.Mutation_70),
               type(DestroyOperators.Mutation_75),
               type(DestroyOperators.Mutation_80),
               type(DestroyOperators.Mutation_85),
               type(DestroyOperators.Mutation_90),
               type(DestroyOperators.Mutation_95),
               type(DestroyOperators.Proximity_005),
               type(DestroyOperators.Proximity_010),
               type(DestroyOperators.Proximity_015),
               type(DestroyOperators.Proximity_020),
               type(DestroyOperators.Proximity_025),
               type(DestroyOperators.Proximity_030),
               type(DestroyOperators.Proximity_035),
               type(DestroyOperators.Proximity_040),
               type(DestroyOperators.Proximity_045),
               type(DestroyOperators.Proximity_05),
               type(DestroyOperators.Proximity_055),
               type(DestroyOperators.Proximity_060),
               type(DestroyOperators.Proximity_065),
               type(DestroyOperators.Proximity_070),
               type(DestroyOperators.Proximity_075),
               type(DestroyOperators.Proximity_080),
               type(DestroyOperators.Proximity_085),
               type(DestroyOperators.Proximity_090),
               type(DestroyOperators.Proximity_095),
               type(DestroyOperators.Proximity_10),
               type(DestroyOperators.Rens_05),
               type(DestroyOperators.Rens_10),
               type(DestroyOperators.Rens_15),
               type(DestroyOperators.Rens_20),
               type(DestroyOperators.Rens_25),
               type(DestroyOperators.Rens_30),
               type(DestroyOperators.Rens_35),
               type(DestroyOperators.Rens_40),
               type(DestroyOperators.Rens_45),
               type(DestroyOperators.Rens_50),
               type(DestroyOperators.Rens_55),
               type(DestroyOperators.Rens_60),
               type(DestroyOperators.Rens_65),
               type(DestroyOperators.Rens_70),
               type(DestroyOperators.Rens_75),
               type(DestroyOperators.Rens_80),
               type(DestroyOperators.Rens_85),
               type(DestroyOperators.Rens_90),
               type(DestroyOperators.Rens_95),
               type(DestroyOperators.Rins_05),
               type(DestroyOperators.Rins_10),
               type(DestroyOperators.Rins_15),
               type(DestroyOperators.Rins_20),
               type(DestroyOperators.Rins_25),
               type(DestroyOperators.Rins_30),
               type(DestroyOperators.Rins_35),
               type(DestroyOperators.Rins_40),
               type(DestroyOperators.Rins_45),
               type(DestroyOperators.Rins_50),
               type(DestroyOperators.Rins_55),
               type(DestroyOperators.Rins_60),
               type(DestroyOperators.Rins_65),
               type(DestroyOperators.Rins_70),
               type(DestroyOperators.Rins_75),
               type(DestroyOperators.Rins_80),
               type(DestroyOperators.Rins_85),
               type(DestroyOperators.Rins_90),
               type(DestroyOperators.Rins_95),
               type(DestroyOperators.Random_Objective))

RepairType = (type(RepairOperators.Repair))

AcceptType = (MovingAverageThreshold,
              GreatDeluge,
              HillClimbing,
              LateAcceptanceHillClimbing,
              NonLinearGreatDeluge,
              AlwaysAccept,
              RecordToRecordTravel,
              SimulatedAnnealing,
              RandomAccept)

SelectorType = (AlphaUCB,
                MABSelector,
                RandomSelect,
                RouletteWheel,
                SegmentedRouletteWheel)

StopType = (MaxIterations,
            MaxRuntime,
            NoImprovement,
            StoppingCriterion)

# ---------------------------------------------------------------------------
# Operator lookup maps: JSON config name -> function reference
# ---------------------------------------------------------------------------
_DESTROY_OP_MAP = {name: getattr(DestroyOperators, name)
                   for name in dir(DestroyOperators)
                   if not name.startswith('_') and name[0].isupper()
                   and callable(getattr(DestroyOperators, name))}

_REPAIR_OP_MAP = {name: getattr(RepairOperators, name)
                  for name in dir(RepairOperators)
                  if not name.startswith('_') and name[0].isupper()
                  and callable(getattr(RepairOperators, name))}


class Balans:
    """
    High-Level Architecture:

    From the input MIP file, an Instance() is created.
    The Instance() provides:
        - Seed and timelimit parameters
        - MIP model
        - LP solution and objective
        - Indices for binary, discrete variables
        - solve(operator_settings)
        - undo_solve()

    From an Instance(), a State() is created.
    The State() provides:
        - Instance
        - Solution
        - Previous solution
        - Operator settings
        - solve_and_update()
            - calls Instance.solve(operator_settings)
    From initial state, ALNS() is created.
    ALNS iterates by calling a pair of destroy_repair on State()

    Operators takes a State()
        - Destroy operators updates States.operator_settings
        - Repair operator calls State.solve_and_update()
    """

    def __init__(self,
                 destroy_ops=None,
                 repair_ops=None,
                 selector=None,
                 accept=None,
                 stop=None,
                 seed: int = None,
                 n_mip_jobs: int = None,
                 mip_solver: str = None,
                 timelimit_first_solution: float = None,
                 timelimit_alns_iteration: float = None,
                 timelimit_local_branching_iteration: float = None,
                 timelimit_crossover_random_feasible: float = None,
                 big_m: float = None,
                 *,
                 config: str = None,
                 ):

        # ------------------------------------------------------------------
        # Resolve configuration: constructor kwarg > config file > hardcoded default
        # ------------------------------------------------------------------
        config_path = config if config is not None else ConfigFactory.DEFAULT_CONFIG_PATH
        cfg = ConfigFactory.load(config_path)  # raw JSON dict, no defaults applied

        # --- Scalars: kwarg > cfg value > Constants default ---
        if seed is None:
            seed = cfg.get('seed', Constants.default_seed)
        if n_mip_jobs is None:
            n_mip_jobs = cfg.get('n_mip_jobs', Constants.default_n_mip_jobs)
        if mip_solver is None:
            mip_solver = cfg.get('mip_solver', Constants.default_solver)
        if timelimit_first_solution is None:
            timelimit_first_solution = cfg.get('timelimit_first_solution', Constants.timelimit_first_solution)
        if timelimit_alns_iteration is None:
            timelimit_alns_iteration = cfg.get('timelimit_alns_iteration', Constants.timelimit_alns_iteration)
        if timelimit_local_branching_iteration is None:
            timelimit_local_branching_iteration = cfg.get('timelimit_local_branching_iteration',
                                                          Constants.timelimit_local_branching_iteration)
        if timelimit_crossover_random_feasible is None:
            timelimit_crossover_random_feasible = cfg.get('timelimit_crossover_random_feasible',
                                                          Constants.timelimit_crossover_random_feasible)
        if big_m is None:
            big_m = cfg.get('big_m', Constants.M)

        # --- Complex types: kwarg > build from cfg > hardcoded default (matching default.json) ---
        # Destroy operators
        if destroy_ops is None:
            if 'destroy_ops' in cfg:
                destroy_ops = ConfigFactory.resolve_operators(cfg['destroy_ops'],
                                                              _DESTROY_OP_MAP, kind="destroy operator")
            else:
                destroy_ops = [crossover,
                               mutation_50,
                               rins_40,
                               rens_40, rens_50,
                               proximity_005,
                               local_branching_10, local_branching_20,
                               local_branching_30, local_branching_50]

        # Repair operators
        if repair_ops is None:
            if 'repair_ops' in cfg:
                repair_ops = ConfigFactory.resolve_operators(cfg['repair_ops'],
                                                             _REPAIR_OP_MAP, kind="repair operator")
            else:
                repair_ops = [repair]

        # Acceptance criterion
        if accept is None:
            if 'accept' in cfg:
                accept = ConfigFactory.build_acceptance(cfg['accept'])
            else:
                accept = HillClimbing()

        # Stopping criterion
        if stop is None:
            if 'stop' in cfg:
                stop = ConfigFactory.build_stop(cfg['stop'])
            else:
                stop = MaxRuntime(300)

        # Selector
        if selector is None:
            if 'selector' in cfg:
                selector = ConfigFactory.build_selector(cfg['selector'], len(destroy_ops), len(repair_ops))
            else:
                selector = MABSelector(scores=[8, 4, 2, 1],
                                       num_destroy=len(destroy_ops),
                                       num_repair=len(repair_ops),
                                       learning_policy=LearningPolicy.Softmax(tau=1.359686))

        self._validate_balans_args(destroy_ops, repair_ops, selector, accept, stop,
                                   seed, n_mip_jobs, mip_solver,
                                   timelimit_first_solution,
                                   timelimit_alns_iteration,
                                   timelimit_crossover_random_feasible)

        # Parameters
        self.config = config
        self.destroy_ops = destroy_ops
        self.repair_ops = repair_ops
        self.selector = selector
        self.accept = accept
        self.stop = stop
        self.seed = seed
        self.n_mip_jobs = n_mip_jobs
        self.mip_solver_str = mip_solver
        self.timelimit_first_solution = timelimit_first_solution
        self.timelimit_alns_iteration = timelimit_alns_iteration
        self.timelimit_local_branching_iteration = timelimit_local_branching_iteration
        self.timelimit_crossover_random_feasible = timelimit_crossover_random_feasible
        self.big_m = big_m

        # RNG
        self._rng = create_rng(self.seed)
        self.alns_seed = self._rng.randint(0, self.seed)

        # ALNS
        self.alns = None

        # Instance and the first solution
        self._instance: Optional[_Instance] = None
        self._initial_index_to_val: Optional[Dict[int, float]] = None
        self._initial_obj_val: Optional[float] = None

    @property
    def instance(self) -> _Instance:
        return self._instance

    @property
    def initial_index_to_val(self) -> Dict[int, float]:
        return self._initial_index_to_val

    @property
    def initial_obj_val(self) -> float:
        return self._initial_obj_val

    def solve(self, instance_path, index_to_val=None) -> Result:
        """
        instance_path: the path to the MIP instance file
        index_to_val: initial (partial) solution to warm start the variables

        Returns
        -------
        Result
            ALNS result object, containing the best solution and some additional statistics.
                result.best_state.solution()
                result.best_state.objective()
        """
        self._validate_solve_args(instance_path)

        # Start timing
        set_solve_start_time()

        # Set global wall-clock deadline so every MIP call respects the budget
        if isinstance(self.stop, MaxRuntime):
            set_solve_deadline(self.stop.max_runtime)
        else:
            set_solve_deadline(None)

        # Print configuration
        print(self)
        print(f"{timestamp()} Solving instance: {instance_path}")

        # MIP is an instance of _BaseMIP created from given mip instance.
        # big_m is passed directly so Constants.M is never globally mutated.
        mip = create_mip_solver(instance_path, self.seed, self.n_mip_jobs, self.mip_solver_str,
                                big_m=self.big_m)

        sense = "maximize (converted to minimize)" if mip.is_obj_sense_changed else "minimize"
        print(f"{timestamp()} Objective sense: {sense}")

        # Create instance with timelimits set from this Balans configuration.
        self._instance = _Instance(mip, self.seed,
                                   timelimit_first_solution=self.timelimit_first_solution,
                                   timelimit_alns_iteration=self.timelimit_alns_iteration,
                                   timelimit_local_branching_iteration=self.timelimit_local_branching_iteration,
                                   timelimit_crossover_random_feasible=self.timelimit_crossover_random_feasible)

        print(f"{timestamp()} Finding initial solution...")
        self._initial_index_to_val, self._initial_obj_val = self._instance.initial_solve(index_to_val=index_to_val)

        # Display the initial objective in original space for the user
        display_obj = self._initial_obj_val
        if self._instance.mip.is_obj_sense_changed:
            display_obj = -self._initial_obj_val
        print(f"{timestamp()} >>> START objective: {display_obj}")

        # Initial state and solution
        initial_state = _State(self.instance, self.initial_index_to_val, self.initial_obj_val,
                               previous_index_to_val=self.initial_index_to_val)

        # Create ALNS
        self.alns = ALNS(np.random.default_rng(self.alns_seed))

        # Set ALNS operators according to MIP type, and if successful, start iterating ALNS
        if self._set_alns_operators():

            # Iterate ALNS
            result = self.alns.iterate(initial_state, self.selector, self.accept, self.stop)

            # During ALNS, all objectives are in minimized space (negated for max problems).
            # Convert everything back to original space for user-facing output.
            if self.instance.mip.is_obj_sense_changed:
                result.statistics._objectives = [-obj for obj in result.statistics.objectives]
                result.best_state.obj_val = -result.best_state.obj_val
                self._initial_obj_val = -self._initial_obj_val

            print(f"{timestamp()} >>> FINISH objective: {result.best_state.objective()} "
                  f"(initial objective: {self._initial_obj_val})")
        else:
            result = None

        return result

    @staticmethod
    def _is_local_branching(op):
        return (op == DestroyOperators.Local_Branching_05 or
                op == DestroyOperators.Local_Branching_10 or
                op == DestroyOperators.Local_Branching_15 or
                op == DestroyOperators.Local_Branching_20 or
                op == DestroyOperators.Local_Branching_25 or
                op == DestroyOperators.Local_Branching_30 or
                op == DestroyOperators.Local_Branching_35 or
                op == DestroyOperators.Local_Branching_40 or
                op == DestroyOperators.Local_Branching_45 or
                op == DestroyOperators.Local_Branching_50 or
                op == DestroyOperators.Local_Branching_55 or
                op == DestroyOperators.Local_Branching_60 or
                op == DestroyOperators.Local_Branching_65 or
                op == DestroyOperators.Local_Branching_70 or
                op == DestroyOperators.Local_Branching_75 or
                op == DestroyOperators.Local_Branching_80 or
                op == DestroyOperators.Local_Branching_85 or
                op == DestroyOperators.Local_Branching_90 or
                op == DestroyOperators.Local_Branching_95)

    @staticmethod
    def _is_proximity(op):
        return (op == DestroyOperators.Proximity_005 or
                op == DestroyOperators.Proximity_010 or
                op == DestroyOperators.Proximity_015 or
                op == DestroyOperators.Proximity_020 or
                op == DestroyOperators.Proximity_025 or
                op == DestroyOperators.Proximity_030 or
                op == DestroyOperators.Proximity_035 or
                op == DestroyOperators.Proximity_040 or
                op == DestroyOperators.Proximity_045 or
                op == DestroyOperators.Proximity_05 or
                op == DestroyOperators.Proximity_055 or
                op == DestroyOperators.Proximity_060 or
                op == DestroyOperators.Proximity_065 or
                op == DestroyOperators.Proximity_070 or
                op == DestroyOperators.Proximity_075 or
                op == DestroyOperators.Proximity_080 or
                op == DestroyOperators.Proximity_085 or
                op == DestroyOperators.Proximity_090 or
                op == DestroyOperators.Proximity_095 or
                op == DestroyOperators.Proximity_10)

    def _set_alns_operators(self):

        num_destroy_removed = 0
        # If the problem has no binary, remove Local Branching and Proximity
        if len(self.instance.binary_indexes) == 0:
            for op in self.destroy_ops:
                if self._is_local_branching(op) or self._is_proximity(op):
                    num_destroy_removed += 1
                    continue
                self.alns.add_destroy_operator(op)
        # If the problem has no integer, remove Dins
        elif len(self.instance.integer_indexes) == 0:
            for op in self.destroy_ops:
                if op == DestroyOperators.Dins:
                    num_destroy_removed += 1
                    continue
                self.alns.add_destroy_operator(op)
        else:
            for op in self.destroy_ops:
                self.alns.add_destroy_operator(op)

        for op in self.repair_ops:
            self.alns.add_repair_operator(op)

        num_remaining_destroy = self.selector.num_destroy - num_destroy_removed

        # No more operators left, return failure
        if num_remaining_destroy == 0:
            return False

        # If ops are removed, re-create bandit selector with adjusted arm counter
        if num_destroy_removed > 0:
            if isinstance(self.selector, MABSelector):
                self.selector = MABSelector(scores=self.selector.scores,
                                            num_destroy=num_remaining_destroy,
                                            num_repair=self.selector.num_repair,
                                            learning_policy=self.selector.mab.learning_policy)

        # Ops added successfully
        return True

    @staticmethod
    def _validate_balans_args(destroy_ops, repair_ops, selector, accept, stop,
                              seed, n_mip_jobs, mip_solver,
                              timelimit_first_solution,
                              timelimit_alns_iteration,
                              timelimit_crossover_random_feasible):

        # Destroy Type
        for op in destroy_ops:
            check_true(isinstance(op, DestroyType), TypeError("Destroy Type mismatch." + str(op)))

        # Repair Type
        for op in repair_ops:
            check_true(isinstance(op, RepairType), TypeError("Repair Type mismatch." + str(op)))

        # Selector Type
        check_true(isinstance(selector, SelectorType), TypeError("Selector Type mismatch." + str(selector)))

        # Selector Type
        check_true(isinstance(accept, AcceptType), TypeError("Selector Type mismatch." + str(accept)))

        # Stop Type
        check_true(isinstance(stop, StopType), TypeError("Stop Type mismatch." + str(stop)))

        # Seed
        check_true(isinstance(seed, int), TypeError("The seed must be an integer." + str(seed)))

        # Parallel MIP jobs
        check_true(isinstance(n_mip_jobs, int),
                   TypeError("Number of parallel jobs must be an integer." + str(n_mip_jobs)))
        check_true(n_mip_jobs != 0, ValueError("Number of parallel jobs cannot be zero." + str(n_mip_jobs)))

        # MIP solver
        check_true(isinstance(mip_solver, str), TypeError("MIP solver backend must be a string." + str(mip_solver)))
        check_true(mip_solver in ["scip", "gurobi"],
                   ValueError("MIP solver backend must be a scip or gurobi." + str(mip_solver)))

        # Timelimit consistency
        if isinstance(stop, MaxRuntime):
            if stop.max_runtime < timelimit_first_solution:
                raise ValueError(
                    f"MaxRuntime ({stop.max_runtime}s) must be >= "
                    f"timelimit_first_solution ({timelimit_first_solution}s). "
                    f"The initial solution search alone needs {timelimit_first_solution}s.")

        if timelimit_alns_iteration < timelimit_crossover_random_feasible:
            raise ValueError(
                f"timelimit_alns_iteration ({timelimit_alns_iteration}s) must be >= "
                f"timelimit_crossover_random_feasible ({timelimit_crossover_random_feasible}s). "
                f"The crossover destroy runs inside an ALNS iteration.")

    @staticmethod
    def _validate_solve_args(instance_path):

        check_true(isinstance(instance_path, str), TypeError("Instance path must be a string: " + str(instance_path)))
        check_false(instance_path == "", ValueError("Instance cannot be empty: " + str(instance_path)))
        check_false(instance_path is None, ValueError("Instance cannot be None: " + str(instance_path)))
        check_true(os.path.isfile(instance_path), ValueError("Instance must exist: " + str(instance_path)))

    def __str__(self) -> str:
        separator = "=" * 60
        lines = [
            separator,
            "BALANS Configuration",
            separator,
        ]
        if self.config is not None:
            lines.append(f"  Config File         : {self.config}")
        lines += [
            f"  MIP Solver          : {self.mip_solver_str}",
            f"  Seed                : {self.seed}",
            f"  ALNS Seed           : {self.alns_seed}",
            f"  MIP Jobs            : {self.n_mip_jobs}",
            "",
            f"  {len(self.destroy_ops)} Destroy Operators  :",
        ]
        for op in self.destroy_ops:
            lines.append(f"    - {op.__name__}")
        lines.append("")
        lines.append(f"  {len(self.repair_ops)} Repair Operators   :")
        for op in self.repair_ops:
            lines.append(f"    - {op.__name__}")
        lines.append("")

        # Selector details
        lines.append(f"  Selector            : {type(self.selector).__name__}")
        if isinstance(self.selector, MABSelector):
            lines.append(f"    Scores            : {self.selector.scores}")
            lines.append(f"    Learning Policy   : {self.selector.mab.learning_policy}")
        lines.append("")

        # Acceptance details
        lines.append(f"  Acceptance          : {type(self.accept).__name__}")
        if isinstance(self.accept, SimulatedAnnealing):
            lines.append(f"    Start Temperature : {self.accept.start_temperature}")
            lines.append(f"    End Temperature   : {self.accept.end_temperature}")
            lines.append(f"    Step              : {self.accept.step}")
            lines.append(f"    Method            : {self.accept.method}")
        elif isinstance(self.accept, RecordToRecordTravel):
            lines.append(f"    Start Threshold   : {self.accept.start_threshold}")
            lines.append(f"    End Threshold     : {self.accept.end_threshold}")
            lines.append(f"    Step              : {self.accept.step}")
            lines.append(f"    Method            : {self.accept.method}")
        lines.append("")

        # Stopping criterion details
        lines.append(f"  Stopping Criterion  : {type(self.stop).__name__}")
        if isinstance(self.stop, MaxIterations):
            lines.append(f"    Max Iterations    : {self.stop.max_iterations}")
        elif isinstance(self.stop, MaxRuntime):
            lines.append(f"    Max Runtime       : {self.stop.max_runtime}s")
        elif isinstance(self.stop, NoImprovement):
            lines.append(f"    Max No Improvement: {self.stop._max_iterations}")
        lines.append("")

        lines.append("  ALNS Time Limits:")
        lines.append(f"    Initial Solution            : {self.timelimit_first_solution}s")
        lines.append(f"    ALNS Iteration              : {self.timelimit_alns_iteration}s")
        lines.append(f"    Local Branching Iteration   : {self.timelimit_local_branching_iteration}s")
        lines.append(f"    Crossover Random Feasible   : {self.timelimit_crossover_random_feasible}s")
        lines.append("")
        lines.append(f"  Big-M               : {self.big_m}")
        lines.append(separator)
        return "\n".join(lines)


class ParBalans:
    """
    ParBalans: Run several Balans configurations in parallel.

    .. note::

       On Windows (and any platform that uses the *spawn* multiprocessing
       start method), the calling script **must** guard its top-level code
       with ``if __name__ == '__main__':`` to avoid an infinite re-import
       loop.  Example::

           if __name__ == '__main__':
               parbalans = ParBalans(n_jobs=4, mip_solver="scip")
               best_sol, best_obj = parbalans.run("instance.mps")
    """

    # Valid string aliases for the built-in generators
    RANDOM_CONFIGS = "random_configs"
    TOP_CONFIGS    = "top_configs"

    def __init__(self,
                 n_jobs: int = 1,
                 n_mip_jobs: int = Constants.default_n_mip_jobs,
                 mip_solver: str = Constants.default_solver,
                 output_dir: str = Constants.default_parbalans_output_dir,
                 balans_generator=None):
        """
        ParBalans runs several Balans configurations in parallel.
        Configurations can be random (ParBalans.RANDOM_CONFIGS) or top performance (ParBalans.TOP_CONFIGS)
        Alternatively, a custom function can be given that returns a list of Balans objects of size n_jobs.

        Parameters
        ----------
        n_jobs: Parallel Balans runs
        n_mip_jobs: The number of threads for the underlying mip solver, only supported by Gurobi
        mip_solver: "scip" or "gurobi"
        output_dir: Saves one file per parallel Balans run as a pickle object
                    The object is a tuple with three elements: obj_of_iteration, time_of_iteration, and arm_to_reward_counts
                    There are N+1 iterations, including the initial solution
                    The time is cumulative runtime when the iteration happens
                    Reward counts is the overall statistics
        balans_generator: Controls which Balans configurations are run in parallel. Three options:
                          - "random_configs" : each parallel slot gets a randomly generated Balans config
                            (calls ParBalans._generate_random_balans).
                          - "top_configs"    : parallel slots are filled with the curated top configs
                            from balans/configs/top_configs/, in priority order
                            (calls ParBalans._generate_top_configs).
                          - callable         : a user-supplied function that takes no arguments and
                            returns a Balans instance (or a list of Balans instances for n_jobs slots).
                          - None             : defaults to "top_configs".
        """

        # Set params
        self.n_jobs = n_jobs
        self.n_mip_jobs = n_mip_jobs
        self.mip_solver = mip_solver
        self.output_dir = output_dir

        # Resolve balans_generator: string alias, callable, or default
        if balans_generator is None or balans_generator == ParBalans.TOP_CONFIGS:
            self.balans_generator = ParBalans._generate_top_configs
        elif balans_generator == ParBalans.RANDOM_CONFIGS:
            self.balans_generator = ParBalans._generate_random_balans
        elif callable(balans_generator):
            self.balans_generator = balans_generator
        else:
            raise ValueError(f"Error: ParBalans balans_generator must be '{ParBalans.RANDOM_CONFIGS}', "
                             f""f"'{ParBalans.TOP_CONFIGS}', a callable, or None. "
                             f"Got: {balans_generator!r}")

        # Create the results directory
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, instance_path, index_to_val=None) -> Tuple[Dict[int, float], float]:
        """
        instance_path: the path to the MIP instance file
        index_to_val: initial (partial) solution to warm start the variables

        Returns
        -------
        best_index_to_val, best_obj a tuple that contains the best solution (index_to_val) dictionary
                                    and the best objective found
        """

        # Create a dummy solver to understand objective sense
        mip = create_mip_solver(instance_path, Constants.default_seed, self.n_mip_jobs, self.mip_solver)

        # Generate one Balans instance per parallel job
        balans_list = self.balans_generator(self.n_jobs, self.mip_solver, self.n_mip_jobs)

        with Pool(processes=self.n_jobs) as pool:
            best_sol_and_obj_of_job = pool.starmap(self._solve_instance_with_balans,
                                                   [(idx, instance_path, index_to_val, balans_list[idx])
                                                    for idx in range(self.n_jobs)])

        # Balans.solve() converts objectives to original space before returning.
        # Compare based on the original objective sense.
        if mip.is_obj_sense_changed:
            # Maximization: pick the largest original-space objective
            best_index_to_val, best_obj = max(best_sol_and_obj_of_job, key=lambda t: t[1])
        else:
            # Minimization: pick the smallest objective
            best_index_to_val, best_obj = min(best_sol_and_obj_of_job, key=lambda t: t[1])

        return best_index_to_val, best_obj

    def _solve_instance_with_balans(self, idx, instance_path, index_to_val, balans):

        if index_to_val:
            result = balans.solve(instance_path, index_to_val)
        else:
            result = balans.solve(instance_path)

        if result:
            # There are N+1 iterations, including the initial solution
            # The time is cumulative runtime when the iteration happens
            # Reward counts is the overall statistics
            obj_of_iteration = result.statistics.objectives
            time_of_iteration = np.cumsum(result.statistics.runtimes)
            arm_to_reward_counts = dict(result.statistics.destroy_operator_counts)

            r = [obj_of_iteration, time_of_iteration, arm_to_reward_counts]
            result_path = os.path.join(self.output_dir, f"result_{idx}.pkl")
            with open(result_path, "wb") as fp:
                pickle.dump(r, fp)

        return result.best_state.solution(), result.best_state.objective()

    @staticmethod
    def _generate_random_balans(n_jobs: int,
                                mip_solver: str = Constants.default_solver,
                                n_mip_jobs: int = Constants.default_n_mip_jobs) -> List[Balans]:
        """Return a list of *n_jobs* independently randomised Balans instances.

        Parameters
        ----------
        n_jobs : int
            Number of Balans instances to create.
        mip_solver : str
            MIP solver backend passed to every Balans instance.
        n_mip_jobs : int
            Number of parallel MIP threads passed to every Balans instance.

        Returns
        -------
        list[Balans]
            Exactly *n_jobs* Balans instances, each with a different random config.
        """

        # Pool of options
        DESTROY_CATEGORIES = {"crossover": [DestroyOperators.Crossover],
                              "mutation": [DestroyOperators.Mutation_10, DestroyOperators.Mutation_20,
                                           DestroyOperators.Mutation_30,
                                           DestroyOperators.Mutation_40, DestroyOperators.Mutation_50],
                              "local_branching": [DestroyOperators.Local_Branching_10,
                                                  DestroyOperators.Local_Branching_20,
                                                  DestroyOperators.Local_Branching_30,
                                                  DestroyOperators.Local_Branching_40,
                                                  DestroyOperators.Local_Branching_50],
                              "proximity": [DestroyOperators.Proximity_020, DestroyOperators.Proximity_040,
                                            DestroyOperators.Proximity_060,
                                            DestroyOperators.Proximity_080, DestroyOperators.Proximity_10],
                              "rens": [DestroyOperators.Rens_10, DestroyOperators.Rens_20, DestroyOperators.Rens_30,
                                       DestroyOperators.Rens_40,
                                       DestroyOperators.Rens_50],
                              "rins": [DestroyOperators.Rins_10, DestroyOperators.Rins_20, DestroyOperators.Rins_30,
                                       DestroyOperators.Rins_40,
                                       DestroyOperators.Rins_50]}
        ACCEPT_TYPE = ["HillClimbing", "SimulatedAnnealing"]
        LEARNING_POLICY = ["EpsilonGreedy", "Softmax", "ThompsonSampling"]
        REPAIR_OPERATORS = [RepairOperators.Repair]
        LP_to_REWARDS = {"binary": [[1, 1, 0, 0], [1, 1, 1, 0]],
                         "numeric": [[3, 2, 1, 0], [5, 2, 1, 0], [5, 4, 2, 0], [8, 3, 1, 0],
                                     [8, 4, 2, 1], [16, 4, 2, 1]]}

        balans_list: List[Balans] = []
        for _ in range(n_jobs):

            # Destroy
            num_destroy = random.randint(len(DESTROY_CATEGORIES) - 2, len(DESTROY_CATEGORIES) * 3)
            chosen_destroy_ops = []
            if num_destroy > len(DESTROY_CATEGORIES) - 1:
                # Choose at least one member from each category
                for category in DESTROY_CATEGORIES:
                    element = random.choice(DESTROY_CATEGORIES[category])
                    chosen_destroy_ops.append(element)

                # Remove the already chosen elements from the pool
                all_elements = [item for sublist in DESTROY_CATEGORIES.values() for item in sublist]
                remaining_pool = list(set(all_elements) - set(chosen_destroy_ops))

                # Choose the remaining elements randomly from the remaining pool
                remaining_elements = num_destroy - len(DESTROY_CATEGORIES)
                chosen_destroy_ops.extend(random.sample(remaining_pool, remaining_elements))
            else:
                # Choose the categories to be included
                chosen_categories = random.sample(list(DESTROY_CATEGORIES.keys()), num_destroy)

                # Choose one element from each chosen category
                for category in chosen_categories:
                    element = random.choice(DESTROY_CATEGORIES[category])
                    chosen_destroy_ops.append(element)

            # Accept
            chosen_accept_type = []
            for op in ACCEPT_TYPE:
                if "HillClimbing" in op:
                    chosen_accept_type.append(HillClimbing())
                if "SimulatedAnnealing" in op:
                    chosen_accept_type.append(SimulatedAnnealing(start_temperature=10,
                                                                 end_temperature=1,
                                                                 step=random.uniform(0.01, 1),
                                                                 method="linear"))
            acceptance_obj = random.choice(chosen_accept_type)

            # Learning Policy
            chosen_learning_policy = []
            for op in LEARNING_POLICY:
                if "EpsilonGreedy" in op:
                    chosen_learning_policy.append(LearningPolicy.EpsilonGreedy(epsilon=random.uniform(0.01, 0.5)))
                if "Softmax" in op:
                    chosen_learning_policy.append(LearningPolicy.Softmax(tau=random.uniform(1, 3)))
                if "ThompsonSampling" in op:
                    chosen_learning_policy.append(LearningPolicy.ThompsonSampling())
            chosen_lp = random.choice(chosen_learning_policy)

            # Rewards
            chosen_scores = random.choice(LP_to_REWARDS["numeric"])
            if isinstance(chosen_lp, LearningPolicy.ThompsonSampling):
                chosen_scores = random.choice(LP_to_REWARDS["binary"])

            # Seed
            chosen_seed = random.randint(1, 100000)

            # Stop
            stop = MaxIterations(10)  # MaxRuntime(100)

            # Balans
            balans = Balans(destroy_ops=chosen_destroy_ops,
                            repair_ops=REPAIR_OPERATORS,
                            selector=MABSelector(scores=chosen_scores,
                                                 num_destroy=len(chosen_destroy_ops),
                                                 num_repair=len(REPAIR_OPERATORS),
                                                 learning_policy=chosen_lp,
                                                 seed=chosen_seed),
                            accept=acceptance_obj,
                            stop=stop,
                            n_mip_jobs=n_mip_jobs,
                            mip_solver=mip_solver)

            balans_list.append(balans)

        return balans_list

    # ----- Priority order for top_configs folders -----
    _TOP_CONFIG_PRIORITY = ['miplibhard', 'distmiplib', 'ijcai25']

    @staticmethod
    def _generate_top_configs(n_jobs: int,
                              mip_solver: str = Constants.default_solver,
                              n_mip_jobs: int = Constants.default_n_mip_jobs) -> List[Balans]:
        """Return a list of *n_jobs* Balans instances built from top config files.

        Selection order
        ---------------
        1. Discover all sub-folders under ``top_configs/``.
        2. Priority order among folders: miplibhard → distmiplib → ijcai25 →
           remaining folders in alphabetical order.
        3. **Round 1 (top 3):** From each folder (in priority order), take the
           first 3 config files sorted alphabetically by filename.
        4. **Round 2 (the rest):** From each folder (same order), take the
           remaining files sorted alphabetically.
        5. If *n_jobs* exceeds the total number of config files, cycle through
           the ordered list again with a unique random seed for each extra job.

        Parameters
        ----------
        n_jobs : int
            Number of Balans instances to create.
        mip_solver : str
            MIP solver backend passed to every Balans instance.
        n_mip_jobs : int
            Number of parallel MIP threads passed to every Balans instance.

        Returns
        -------
        list[Balans]
            Exactly *n_jobs* Balans instances.
        """
        # --- Discover folders and sort by priority ---
        all_folders = sorted(
            [d for d in os.listdir(Constants.TOP_CONFIGS)
             if os.path.isdir(os.path.join(Constants.TOP_CONFIGS, d))])

        priority = ParBalans._TOP_CONFIG_PRIORITY
        prioritised = [f for f in priority if f in all_folders]
        remaining = [f for f in all_folders if f not in priority]
        ordered_folders = prioritised + remaining

        # --- Collect config paths per folder (sorted by filename) ---
        folder_to_configs: Dict[str, List[str]] = {}
        for folder in ordered_folders:
            folder_path = os.path.join(Constants.TOP_CONFIGS, folder)
            configs = sorted(glob.glob(os.path.join(folder_path, '*.json')))
            if configs:
                folder_to_configs[folder] = configs

        # --- Build the priority-ordered flat list ---
        # Round 1: top 3 from each folder
        ordered_paths: List[str] = []
        for folder in ordered_folders:
            configs = folder_to_configs.get(folder, [])
            ordered_paths.extend(configs[:3])

        # Round 2: remaining files from each folder
        for folder in ordered_folders:
            configs = folder_to_configs.get(folder, [])
            ordered_paths.extend(configs[3:])

        # --- Create Balans instances ---
        balans_list: List[Balans] = []
        for i in range(n_jobs):
            config_path = ordered_paths[i % len(ordered_paths)]
            if i < len(ordered_paths):
                # Unique config — use the seed from the config file
                balans_list.append(Balans(config=config_path,
                                          mip_solver=mip_solver,
                                          n_mip_jobs=n_mip_jobs))
            else:
                # Wrapped around — reuse config with a unique random seed
                balans_list.append(Balans(config=config_path,
                                          seed=random.randint(1, 100000),
                                          mip_solver=mip_solver,
                                          n_mip_jobs=n_mip_jobs))

        return balans_list

