import os

import numpy as np
from alns.ALNS import ALNS
from alns.accept import *
from alns.select import *
from alns.stop import *
from mabwiser.mab import LearningPolicy

from balans.base_instance import _Instance
from balans.base_mip import create_mip_solver
from balans.base_state import _State
from balans.solver import DestroyOperators, RepairOperators
from balans.utils import Constants
from tests.test_base import BaseTest


class LocalBranchingTest(BaseTest):

    BaseTest.mip_solver = Constants.gurobi_solver

    def test_local_branching_t1(self):
        # Input
        instance = "model4.lp"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        # Parameters
        seed = 123456

        mip = create_mip_solver(instance_path, seed, mip_solver=BaseTest.mip_solver)
        instance = _Instance(mip)

        index_to_val = {0: 0.0, 1: 0.0, 2: 0.0, 3: 10.0, 4: 10.0, 5: 20.0, 6: 20.0}
        print("initial index to val:", index_to_val)

        initial2 = _State(instance, index_to_val, -40)

        # Initial solution
        initial_index_to_val, initial_obj_val = instance.initial_solve()

        # Create ALNS and add one or more destroy and repair operators
        alns = ALNS(np.random.default_rng(seed))
        alns.add_destroy_operator(DestroyOperators.Local_Branching_10)
        alns.add_repair_operator(RepairOperators.Repair)

        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=1, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
        accept = HillClimbing()
        stop = MaxIterations(1)

        # Run the ALNS algorithm
        result = alns.iterate(initial2, selector, accept, stop)
        # Retrieve the final solution
        best = result.best_state
        print(f"Best heuristic solution objective is {best.objective()}.")
        self.assertEqual(result.best_state.objective(), -40.0)

    def test_local_branching_classic_t2(self):
        # Input
        instance = "model4.lp"
        instance_path = os.path.join(Constants.DATA_TEST, instance)
        print(instance_path)

        # Parameters
        seed = 123456

        mip = create_mip_solver(instance_path, seed, mip_solver=BaseTest.mip_solver)
        instance = _Instance(mip)

        index_to_val = {0: 1.0, 1: 1.0, 2: 0.0, 3: 10.0, 4: 10.0, 5: 20.0, 6: 20.0}
        print("initial index to val:", index_to_val)

        initial2 = _State(instance, index_to_val, -42)

        # Initial solution
        initial_index_to_val, initial_obj_val = instance.initial_solve()

        # Create ALNS and add one or more destroy and repair operators
        alns = ALNS(np.random.default_rng(seed))
        alns.add_destroy_operator(DestroyOperators.Local_Branching_10)
        alns.add_repair_operator(RepairOperators.Repair)

        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=1, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
        accept = HillClimbing()
        stop = MaxIterations(1)

        # Run the ALNS algorithm
        result = alns.iterate(initial2, selector, accept, stop)
        # Retrieve the final solution
        best = result.best_state
        print(f"Best heuristic solution objective is {best.objective()}.")
        self.assertEqual(result.best_state.objective(), -42.0)

    def test_local_branching_v2_t3(self):
        # Input
        instance = "model4.lp"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        # Parameters
        seed = 123456

        mip = create_mip_solver(instance_path, seed, mip_solver=BaseTest.mip_solver)
        instance = _Instance(mip)

        index_to_val = {0: 1.0, 1: 0.0, 2: 0.0, 3: 10.0, 4: 10.0, 5: 20.0, 6: 20.0}
        print("initial index to val:", index_to_val)

        initial2 = _State(instance, index_to_val, -41)

        # Initial solution
        initial_index_to_val, initial_obj_val = instance.initial_solve()

        # Create ALNS and add one or more destroy and repair operators
        alns = ALNS(np.random.default_rng(seed))
        alns.add_destroy_operator(DestroyOperators.Local_Branching_10)
        alns.add_repair_operator(RepairOperators.Repair)

        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=1, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
        accept = HillClimbing()
        stop = MaxIterations(1)

        # Run the ALNS algorithm
        result = alns.iterate(initial2, selector, accept, stop)
        # Retrieve the final solution
        best = result.best_state
        print(f"Best heuristic solution objective is {best.objective()}.")
        self.assertEqual(result.best_state.objective(), -41.0)
