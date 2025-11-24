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
from balans.solver import Balans, DestroyOperators, RepairOperators
from balans.utils import Constants
from tests.test_base import BaseTest


class ProximityTest(BaseTest):

    BaseTest.mip_solver = Constants.scip_solver

    def test_proximity_t1(self):
        # Input
        instance = "model3.lp"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        # Parameters
        seed = Constants.default_seed
        destroy_ops = [DestroyOperators.Proximity_05]
        repair_ops = [RepairOperators.Repair]

        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=1, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
        accept = HillClimbing()
        stop = MaxIterations(1)

        # Solver
        balans = Balans(destroy_ops, repair_ops, selector, accept, stop, seed, mip_solver=BaseTest.mip_solver)

        # Run
        result = balans.solve(instance_path)
        print("Best solution:", result.best_state.objective())

        # Assert
        self.assertEqual(result.best_state.objective(), -60.0)

    def test_proximity_t2(self):
        # Input
        instance = "model3.lp"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        # Parameters
        seed = 123456

        mip = create_mip_solver(instance_path, seed, mip_solver=BaseTest.mip_solver)
        instance = _Instance(mip)

        initial_index_to_val, initial_obj_val = instance.initial_solve()
        print("initial index to val:", initial_index_to_val)
        print("initial initial_obj_val:", initial_obj_val)

        # Here is a different solution than the initial
        index_to_val = {0: 1.0, 1: 1.0, 2: 0.0, 3: 10.0, 4: 10.0, 5: 20.0, 6: 20.0}
        print("index to val:", index_to_val)
        print("obj: ", mip.get_obj_value(index_to_val))

        initial2 = _State(instance, index_to_val, -40)

        # Create ALNS and add one or more destroy and repair operators
        alns = ALNS(np.random.default_rng(seed))
        alns.add_destroy_operator(DestroyOperators.Proximity_05)
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

        # Assert
        self.is_not_worse(-40, result.best_state.objective(), "minimize")
