from alns.accept import *
from alns.select import *
from alns.stop import *
from mabwiser.mab import LearningPolicy

from balans.solver import Balans, DestroyOperators, RepairOperators
from balans.utils import Constants
from tests.test_base import BaseTest


class InvalidTest(BaseTest):

    BaseTest.mip_solver = Constants.scip_solver

    def test_invalid_destroy_op(self):
        with self.assertRaises(TypeError):
            # Parameters
            seed = Constants.default_seed
            destroy_ops = [DestroyOperators.Mutation_25, "NOT VALID"]
            repair_ops = [RepairOperators.Repair]
            selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=5, num_repair=1,
                                   learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
            accept = HillClimbing()
            stop = MaxIterations(5)

            # Solver
            balans = Balans(destroy_ops, repair_ops, selector, accept, stop, seed, mip_solver=BaseTest.mip_solver)

    def test_invalid_destroy_op_none(self):
        with self.assertRaises(TypeError):
            # Parameters
            seed = Constants.default_seed
            destroy_ops = [DestroyOperators.Mutation_25, None]
            repair_ops = [RepairOperators.Repair]
            selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=5, num_repair=1,
                                   learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
            accept = HillClimbing()
            stop = MaxIterations(5)

            # Solver
            balans = Balans(destroy_ops, repair_ops, selector, accept, stop, seed, mip_solver=BaseTest.mip_solver)

    def test_invalid_repair_op(self):
        with self.assertRaises(TypeError):
            # Parameters
            seed = Constants.default_seed
            destroy_ops = [DestroyOperators.Mutation_25]
            repair_ops = [RepairOperators.Repair, "NOT VALID"]
            selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=5, num_repair=1,
                                   learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
            accept = HillClimbing()
            stop = MaxIterations(5)

            # Solver
            balans = Balans(destroy_ops, repair_ops, selector, accept, stop, seed, mip_solver=BaseTest.mip_solver)

    def test_invalid_repair_op_none(self):
        with self.assertRaises(TypeError):
            # Parameters
            seed = Constants.default_seed
            destroy_ops = [DestroyOperators.Mutation_25]
            repair_ops = [RepairOperators.Repair, None]
            selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=5, num_repair=1,
                                   learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
            accept = HillClimbing()
            stop = MaxIterations(5)

            # Solver
            balans = Balans(destroy_ops, repair_ops, selector, accept, stop, seed, mip_solver=BaseTest.mip_solver)

    def test_invalid_seed(self):
        with self.assertRaises(ValueError):
            # Parameters
            seed = -1
            destroy_ops = [DestroyOperators.Mutation_25]
            repair_ops = [RepairOperators.Repair]
            selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=5, num_repair=1,
                                   learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
            accept = HillClimbing()
            stop = MaxIterations(5)

            # Solver
            balans = Balans(destroy_ops, repair_ops, selector, accept, stop, seed, mip_solver=BaseTest.mip_solver)

    def test_invalid_selector(self):
        with self.assertRaises(TypeError):
            # Parameters
            seed = -1
            destroy_ops = [DestroyOperators.Mutation_25]
            repair_ops = [RepairOperators.Repair]
            selector = "NOT VALID" # None is OK we default to sth
            accept = HillClimbing()
            stop = MaxIterations(5)

            # Solver
            balans = Balans(destroy_ops, repair_ops, selector, accept, stop, seed, mip_solver=BaseTest.mip_solver)

    def test_invalid_accept(self):
        with self.assertRaises(TypeError):
            # Parameters
            seed = -1
            destroy_ops = [DestroyOperators.Mutation_25]
            repair_ops = [RepairOperators.Repair]
            selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=5, num_repair=1,
                                   learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
            accept = MaxIterations(5)
            stop = MaxIterations(5)

            # Solver
            balans = Balans(destroy_ops, repair_ops, selector, accept, stop, seed, mip_solver=BaseTest.mip_solver)

    def test_invalid_stop(self):
        with self.assertRaises(TypeError):
            # Parameters
            seed = -1
            destroy_ops = [DestroyOperators.Mutation_25]
            repair_ops = [RepairOperators.Repair]
            selector = None
            accept = HillClimbing()
            stop = "NOT VALID" # None is OK we default to sth

            # Solver
            balans = Balans(destroy_ops, repair_ops, selector, accept, stop, seed, mip_solver=BaseTest.mip_solver)

    def test_invalid_solve(self):
        with self.assertRaises(ValueError):
            # Parameters
            seed = Constants.default_seed
            destroy_ops = [DestroyOperators.Mutation_25]
            repair_ops = [RepairOperators.Repair]
            selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=5, num_repair=1,
                                   learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
            accept = HillClimbing()
            stop = MaxIterations(5)

            # Solver
            balans = Balans(destroy_ops, repair_ops, selector, accept, stop, seed, mip_solver=BaseTest.mip_solver)

            balans.solve("")

    def test_invalid_solve_no_file(self):
        with self.assertRaises(ValueError):
            # Parameters
            seed = Constants.default_seed
            destroy_ops = [DestroyOperators.Mutation_25]
            repair_ops = [RepairOperators.Repair]
            selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=5, num_repair=1,
                                   learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
            accept = HillClimbing()
            stop = MaxIterations(5)

            # Solver
            balans = Balans(destroy_ops, repair_ops, selector, accept, stop, seed, mip_solver=BaseTest.mip_solver)

            balans.solve("does_not_exist.mip")
