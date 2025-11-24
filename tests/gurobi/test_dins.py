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


class DinsTest(BaseTest):

    BaseTest.mip_solver = Constants.gurobi_solver
    
    def test_dins_t1(self):
        # Input
        instance = "model.lp"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        # Parameters
        seed = 123456
        destroy_ops = [DestroyOperators.Dins]
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

        self.assertEqual(result.best_state.objective(), 4)

    def test_dins_t2(self):
        # Input
        instance = "model2.lp"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        # Parameters
        seed = Constants.default_seed
        destroy_ops = [DestroyOperators.Dins]
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
        self.assertEqual(result.best_state.objective(), -60)

    def test_dins_t3(self):
        # Input
        instance = "model2.lp"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        # Parameters
        seed = 123456

        # MIP is an instance of _BaseMIP created from given mip instance
        mip = create_mip_solver(instance_path, seed, mip_solver = Constants.gurobi_solver)
        instance = _Instance(mip)

        index_to_val = {0: -0.0, 1: 10.0, 2: 10.0, 3: 20.0, 4: 20.0}
        print("initial index to val:", index_to_val)

        initial2 = _State(instance, index_to_val,  -30)

        # Initial solution
        initial_index_to_val, initial_obj_val = instance.initial_solve()

        # Create ALNS and add one or more destroy and repair operators
        alns = ALNS(np.random.default_rng(seed))
        alns.add_destroy_operator(DestroyOperators.Dins)
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

        self.assertIsBetter(-30, result.best_state.objective(), "minimize")

    def test_dins_t4(self):
        # Input
        instance = "model2.lp"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        # Parameters
        seed = 123456

        destroy_ops = [DestroyOperators.Dins]
        repair_ops = [RepairOperators.Repair]

        # MIP is an instance of _BaseMIP created from given mip instance
        mip = create_mip_solver(instance_path, seed, mip_solver = Constants.gurobi_solver)
        instance = _Instance(mip)

        # Initial solution
        initial_index_to_val, initial_obj_val = instance.initial_solve()

        initial_index_to_val = {0: -0.0, 1: 10.0, 2: 10.0, 3: 20.0, 4: 20.0}
        print("initial index to val:", initial_index_to_val)

        initial2 = _State(instance, initial_index_to_val, -30)

        # Create ALNS and add one or more destroy and repair operators
        alns = ALNS(np.random.default_rng(seed))
        alns.add_destroy_operator(DestroyOperators.Dins)
        alns.add_repair_operator(RepairOperators.Repair)

        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=1, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
        accept = HillClimbing()
        stop = MaxIterations(1)

        # Run the ALNS algorithm
        result = alns.iterate(initial2, selector, accept, stop)

        # Retrieve the final solution
        best_state = result.best_state
        best_objective = best_state.objective()

        # best_index_to_val = None # best.index_to_val
        #
        # # First variable must remain fixed
        # self.assertEqual(initial_index_to_val[0], best_index_to_val[0])

        print(f"Best heuristic solution objective is {best_objective}.")
        self.assertEqual(best_objective, -40.0)

    def test_dins_t5(self):
        # Input
        instance = "model3.lp"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        # Parameters
        seed = 123456

        # MIP is an instance of _BaseMIP created from given mip instance
        mip = create_mip_solver(instance_path, seed, mip_solver = Constants.gurobi_solver)
        instance = _Instance(mip)

        index_to_val = {0: 1.0, 1: 0.0, 2: 0.0, 3: 10.0, 4: 10.0, 5: 20.0, 6: 20.0}
        print("initial index to val:", index_to_val)

        initial2 = _State(instance, index_to_val, -40)

        # Initial solution
        initial_index_to_val, initial_obj_val = instance.initial_solve()

        # Create ALNS and add one or more destroy and repair operators
        alns = ALNS(np.random.default_rng(seed))
        alns.add_destroy_operator(DestroyOperators.Dins)
        alns.add_repair_operator(RepairOperators.Repair)

        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=1, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
        accept = HillClimbing()
        stop = MaxIterations(5)

        # Run the ALNS algorithm
        result = alns.iterate(initial2, selector, accept, stop)
        # Retrieve the final solution
        best = result.best_state
        print(f"Best heuristic solution objective is {best.objective()}.")
        self.assertEqual(result.best_state.objective(), -40.0)
