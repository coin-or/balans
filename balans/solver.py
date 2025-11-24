import os
import random
from typing import List, Optional, Dict, Tuple
from typing import NamedTuple
import pickle
from multiprocessing import Pool
from functools import partial

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
from balans.utils import Constants, check_false, check_true, create_rng


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


class Balans:
    """
    High-Level Architecture:

    From the input MIP file, an Instance() is created.
    The Instance() provides:
        - Seed
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
                 destroy_ops = None, # List
                 repair_ops = None, # List
                 selector = None, # SelectorType
                 accept = None, # AcceptType
                 stop = None, # StopType
                 seed: int = Constants.default_seed,  # The random seed
                 n_mip_jobs: int = 1,  # Number of threads for the solver
                 mip_solver: str = Constants.default_solver  # MIP solver scip/gurobi
                 ):

        # Validate arguments
        if destroy_ops is None:
            destroy_ops = [DestroyOperators.Crossover,
                           DestroyOperators.Mutation_25,
                           DestroyOperators.Mutation_50,
                           DestroyOperators.Mutation_75,
                           DestroyOperators.Local_Branching_10,
                           DestroyOperators.Local_Branching_25,
                           DestroyOperators.Local_Branching_50,
                           DestroyOperators.Proximity_005,
                           DestroyOperators.Proximity_015,
                           DestroyOperators.Proximity_030,
                           DestroyOperators.Rens_25,
                           DestroyOperators.Rens_50,
                           DestroyOperators.Rens_75,
                           DestroyOperators.Rins_25,
                           DestroyOperators.Rins_50,
                           DestroyOperators.Rins_75]
        if repair_ops is None:
            repair_ops = [RepairOperators.Repair]
        if selector is None:
            best, better, accept, reject = 1, 1, 0, 0
            selector = MABSelector(scores=[best, better, accept, reject],
                                   num_destroy=len(destroy_ops),
                                   num_repair=len(repair_ops),
                                   learning_policy=LearningPolicy.ThompsonSampling())
        if accept is None:
            accept = SimulatedAnnealing(start_temperature=20, end_temperature=1, step=0.1)
        if stop is None:
            stop = MaxIterations(10)

        self._validate_balans_args(destroy_ops, repair_ops, selector, accept, stop, seed, n_mip_jobs, mip_solver)

        # Parameters
        self.destroy_ops = destroy_ops
        self.repair_ops = repair_ops
        self.selector = selector
        self.accept = accept
        self.stop = stop
        self.seed = seed
        self.n_mip_jobs = n_mip_jobs
        self.mip_solver_str = mip_solver

        # RNG
        self._rng = create_rng(self.seed)
        self.alns_seed = self._rng.randint(0, self.seed)

        # ALNS
        self.destroy_ops = destroy_ops
        self.repair_ops = repair_ops
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

        # MIP is an instance of _BaseMIP created from given mip instance
        mip = create_mip_solver(instance_path, self.seed, self.n_mip_jobs, self.mip_solver_str)

        # Create instance with mip model created from mip instance
        self._instance = _Instance(mip, self.seed)

        self._initial_index_to_val, self._initial_obj_val = self._instance.initial_solve(index_to_val=index_to_val)

        print(">>> START objective:", self._initial_obj_val)
        # print(">>> START values:", self._initial_index_to_val)

        # Initial state and solution
        initial_state = _State(self.instance, self.initial_index_to_val, self.initial_obj_val,
                               previous_index_to_val=self.initial_index_to_val)

        # Create ALNS
        self.alns = ALNS(np.random.default_rng(self.alns_seed))

        # Set ALNS operators according to MIP type, and if successful, start iterating ALNS
        if self._set_alns_operators():

            # Iterate ALNS
            result = self.alns.iterate(initial_state, self.selector, self.accept, self.stop)

            # Restore the original objective results, if max problem is reversed
            if self.instance.mip.is_obj_sense_changed:
                obj_list = []
                for obj in result.statistics.objectives:
                    obj_list.append(-obj)
                result.statistics._objectives = obj_list

            print(">>> FINISH objective:", result.best_state.objective())
        else:
            result = None

        # Result run
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
    def _validate_balans_args(destroy_ops, repair_ops, selector, accept, stop, seed, n_mip_jobs, mip_solver):

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

    @staticmethod
    def _validate_solve_args(instance_path):

        check_true(isinstance(instance_path, str), TypeError("Instance path must be a string: " + str(instance_path)))
        check_false(instance_path == "", ValueError("Instance cannot be empty: " + str(instance_path)))
        check_false(instance_path is None, ValueError("Instance cannot be None: " + str(instance_path)))
        check_true(os.path.isfile(instance_path), ValueError("Instance must exist: " + str(instance_path)))


class ParBalans:
    """
    ParBalans: Run several Balans configurations in parallel.
    """

    def __init__(self,
                 n_jobs: int = 1,
                 n_mip_jobs: int = 1,
                 mip_solver: str = Constants.default_solver,
                 output_dir: str = "results/",
                 balans_generator=None):
        """
        ParBalans runs several Balans configurations in parallel.
        See class members for the possible pool of configurations used to generate random Balans configs.

        Parameters
        ----------
        n_jobs: Parallel Balans runs
        n_mip_jobs: The number of threads for the underlying mip solver, only supported by Gurobi
        mip_solver: "scip" or "gurobi"
        output_dir: Saves one file per parallel Balans run as a pickle object
                    The object is tuple with three elements: obj_of_iteration, time_of_iteration, and arm_to_reward_counts
                    There N+1 iterations, including the initial solution
                    The time is cumulative runtime when the iteration happens
                    Reward counts is the overall statistics
        balans_generator: A function that generates a Balans instance for each parallel run.
                          If none given, a default random Balans generator is used.
        """

        # Set params
        self.n_jobs = n_jobs
        self.n_mip_jobs = n_mip_jobs
        self.mip_solver = mip_solver
        self.output_dir = output_dir
        self.balans_generator = balans_generator

        # If no balans generator is given, use random balans generator by default
        if self.balans_generator is None:
            self.balans_generator = ParBalans._generate_random_balans

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

        # Can create other config generator function, random_config() just an example
        with Pool(processes=self.n_jobs) as pool:
            best_sol_and_obj_of_job = pool.map(partial(self._solve_instance_with_balans,
                                                       instance_path=instance_path,
                                                       index_to_val=index_to_val,
                                                       balans=self.balans_generator()),
                                               range(self.n_jobs))

        # Get the best objective value and its index base on the objective sense
        if mip.org_objective_sense == -1 or Constants.maximize:
            best_index_to_val, best_obj = max(best_sol_and_obj_of_job, key=lambda t: t[1])
        else:
            best_index_to_val, best_obj = min(best_sol_and_obj_of_job, key=lambda t: t[1])
        return best_index_to_val, best_obj

    def _solve_instance_with_balans(self, idx, instance_path, index_to_val, balans):

        if index_to_val:
            result = balans.solve(instance_path, index_to_val)
        else:
            result = balans.solve(instance_path)

        if result:
            # There N+1 iterations, including the initial solution
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
    def _generate_random_balans(n_mip_jobs: int = 1, mip_solver: str = Constants.default_solver) -> Balans:

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
        stop = MaxIterations(10) # MaxRuntime(100)

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

        return balans
