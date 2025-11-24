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


class MutationTest(BaseTest):

    BaseTest.mip_solver = Constants.gurobi_solver

    def test_mutation(self):
        # Input
        instance = "noswot.mps"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        # Parameters
        seed = Constants.default_seed
        destroy_ops = [DestroyOperators.Mutation_25]
        repair_ops = [RepairOperators.Repair]

        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=1, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
        accept = HillClimbing()
        stop = MaxIterations(45)

        # Solver
        balans = Balans(destroy_ops, repair_ops, selector, accept, stop, seed, mip_solver=BaseTest.mip_solver)

        # Run
        result = balans.solve(instance_path)
        print("Best solution:", result.best_state.objective())

        # Assert
        self.is_not_worse(balans.initial_obj_val, result.best_state.objective(),
                          balans.instance.mip.org_objective_sense)

    def test_mutation_t1(self):
        # Input
        instance = "model.lp"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        # Parameters
        seed = Constants.default_seed
        destroy_ops = [DestroyOperators.Mutation_25]
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

    def test_mutation_t2(self):
        # Input
        instance = "model2.lp"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        # Parameters
        seed = Constants.default_seed
        destroy_ops = [DestroyOperators.Mutation_50]
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

    def test_mutation_t3(self):
        # Input
        instance = "model2.lp"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        # Parameters
        seed = 123456

        destroy_ops = [DestroyOperators.Mutation_50]
        repair_ops = [RepairOperators.Repair]

        mip = create_mip_solver(instance_path, seed, mip_solver=BaseTest.mip_solver)
        instance = _Instance(mip)

        # Initial solution
        initial_index_to_val, initial_obj_val = instance.initial_solve()

        # Indexes 0, 1, 2 are discrete so only these indexes can be destroyed
        # With this seed, in the firs iteration index=1 is destroy
        # Hence var0 and var2 must remain fixed and only the other variables can change
        # Objective in the next iteration is 50 (minus since sense is minimization)
        initial_index_to_val = {0: -0.0, 1: 10.0, 2: 10.0, 3: 20.0, 4: 20.0}
        print("initial index to val:", initial_index_to_val)

        initial2 = _State(instance, initial_index_to_val, -30)

        # Create ALNS and add one or more destroy and repair operators
        alns = ALNS(np.random.default_rng(seed))
        alns.add_destroy_operator(DestroyOperators.Mutation_50)
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
        # self.assertEqual(result.best_state.objective(), -60)
        self.is_not_worse(-30, result.best_state.objective(), "minimize")

    def test_mutation_t4(self):
        # Input
        instance = "model2.lp"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        # Parameters
        seed = 123456

        destroy_ops = [DestroyOperators.Mutation_50]
        repair_ops = [RepairOperators.Repair]

        mip = create_mip_solver(instance_path, seed, mip_solver=BaseTest.mip_solver)
        instance = _Instance(mip)

        # Initial solution
        initial_index_to_val, initial_obj_val = instance.initial_solve()

        # Indexes 0, 1, 2 are discrete so only these indexes can be destroyed
        # With this seed, in the firs iteration index=1 is destroy
        # Hence var0 and var2 must remain fixed and only the other variables can change
        # Objective in the next iteration is 50 (minus since sense is minimization)
        initial_index_to_val = {0: -0.0, 1: 10.0, 2: 10.0, 3: 20.0, 4: 20.0}
        print("initial index to val:", initial_index_to_val)

        initial2 = _State(instance, initial_index_to_val, -30)

        # Create ALNS and add one or more destroy and repair operators
        alns = ALNS(np.random.default_rng(seed))
        alns.add_destroy_operator(DestroyOperators.Mutation_50)
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

        print(f"Best heuristic solution objective is {best_objective}.")
        self.assertEqual(best_objective, -30.0)

    def test_mutation_t4_with_warm_start(self):
        # Input
        instance = "model2.lp"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        # Parameters
        seed = 123456
        destroy_ops = [DestroyOperators.Mutation_50]
        repair_ops = [RepairOperators.Repair]
        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=1, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
        accept = HillClimbing()
        stop = MaxIterations(5)

        # Solver
        balans = Balans(destroy_ops, repair_ops, selector, accept, stop, seed, mip_solver=BaseTest.mip_solver)

        # Indexes 0, 1, 2 are discrete so only these indexes can be destroyed
        # With this seed, in the firs iteration index=1 is destroy
        # Hence var0 and var2 must remain fixed and only the other variables can change
        # Objective in the next iteration is -30 (minus since sense is minimization)
        initial_index_to_val = {0: -0.0, 1: 10.0, 2: 10.0, 3: 20.0, 4: 20.0}
        print("initial index to val:", initial_index_to_val)

        # Run
        result = balans.solve(instance_path, initial_index_to_val)

        # Retrieve the final solution
        best_state = result.best_state
        best_objective = best_state.objective()
        best_solution = best_state.solution()

        print("Best solution:", result.best_state.solution())
        print("Best solution value:", result.best_state.objective())

        # Assert solution
        best_solution_expected = {0: 0.0, 1: 10.0, 2: 10.0, 3: 20.0, 4: 20.0}
        self.assertDictEqual(best_solution_expected, best_solution)

        # Assert objective
        self.is_not_worse(-20, result.best_state.objective(), "minimize")
