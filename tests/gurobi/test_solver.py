import json
import os
import tempfile

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
from balans.utils import Constants, ConfigFactory
from tests.test_base import BaseTest


class SolverTest(BaseTest):

    BaseTest.mip_solver = Constants.gurobi_solver

    def test_balans_default(self):
        # Input
        instance = "model.lp"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        # Solver
        balans = Balans()

        # Run
        result = balans.solve(instance_path)
        print("Best solution:", result.best_state.objective())

        self.assertEqual(result.best_state.objective(), 4)

    def test_balans_t1(self):
        # Input
        instance = "model.lp"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        # Parameters
        seed = 123456
        destroy_ops = [DestroyOperators.Dins,
                       DestroyOperators.Crossover,
                       DestroyOperators.Proximity_05,
                       DestroyOperators.Mutation_50,
                       DestroyOperators.Local_Branching_10,
                       DestroyOperators.Rins_25,
                       DestroyOperators.Rens_25]

        repair_ops = [RepairOperators.Repair]

        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=7, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))

        accept = HillClimbing()
        stop = MaxIterations(1)

        # Solver
        balans = Balans(destroy_ops, repair_ops, selector, accept, stop, seed, mip_solver=BaseTest.mip_solver)

        # Run
        result = balans.solve(instance_path)
        print("Best solution:", result.best_state.objective())

        self.assertEqual(result.best_state.objective(), 4)

    def test_balans_t2(self):
        # Input
        instance = "model2.lp"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        # Parameters
        seed = Constants.default_seed
        destroy_ops = [DestroyOperators.Dins,
                       DestroyOperators.Proximity_05,
                       DestroyOperators.Mutation_50,
                       DestroyOperators.Local_Branching_10,
                       DestroyOperators.Rins_25,
                       DestroyOperators.Rens_25,
                       DestroyOperators.Crossover]
        repair_ops = [RepairOperators.Repair]

        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=7, num_repair=1,
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

    def test_balans_t3(self):
        # Input
        instance = "model2.lp"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        # Parameters
        seed = 123456
        destroy_ops = [DestroyOperators.Dins,
                       DestroyOperators.Proximity_05,
                       DestroyOperators.Mutation_50,
                       DestroyOperators.Local_Branching_10,
                       DestroyOperators.Rins_25,
                       DestroyOperators.Rens_25,
                       DestroyOperators.Crossover]

        mip = create_mip_solver(instance_path, seed, mip_solver=BaseTest.mip_solver)
        instance = _Instance(mip)

        index_to_val = {0: -0.0, 1: 10.0, 2: 10.0, 3: 20.0, 4: 20.0}
        print("initial index to val:", index_to_val)

        initial2 = _State(instance, index_to_val, -30)

        # Initial solution
        initial_index_to_val, initial_obj_val = instance.initial_solve()

        # Create ALNS and add one or more destroy and repair operators
        alns = ALNS(np.random.default_rng(seed))
        for i in destroy_ops:
            alns.add_destroy_operator(i)
        alns.add_repair_operator(RepairOperators.Repair)

        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=7, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
        accept = HillClimbing()
        stop = MaxIterations(1)

        # Run the ALNS algorithm
        result = alns.iterate(initial2, selector, accept, stop)
        # Retrieve the final solution
        best = result.best_state
        print(f"Best heuristic solution objective is {best.objective()}.")

        self.is_not_worse(-30, result.best_state.objective(), "minimize")

    def test_balans_t4(self):
        # Input
        instance = "model2.lp"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        # Parameters
        seed = 123456
        destroy_ops = [DestroyOperators.Dins,
                       DestroyOperators.Proximity_05,
                       DestroyOperators.Mutation_50,
                       DestroyOperators.Local_Branching_10,
                       DestroyOperators.Rins_25,
                       DestroyOperators.Rens_25,
                       DestroyOperators.Crossover]

        mip = create_mip_solver(instance_path, seed, mip_solver=BaseTest.mip_solver)
        instance = _Instance(mip)

        # Initial solution
        initial_index_to_val, initial_obj_val = instance.initial_solve()

        initial_index_to_val = {0: -0.0, 1: 10.0, 2: 10.0, 3: 20.0, 4: 20.0}
        print("initial index to val:", initial_index_to_val)

        initial2 = _State(instance, initial_index_to_val, -30)

        # Create ALNS and add one or more destroy and repair operators
        alns = ALNS(np.random.default_rng(seed))
        for i in destroy_ops:
            alns.add_destroy_operator(i)
        alns.add_repair_operator(RepairOperators.Repair)

        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=7, num_repair=1,
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
        self.is_not_worse(-30, result.best_state.objective(), "minimize")

    def test_balans_t5(self):
        # Input
        instance = "model3.lp"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        # Parameters
        seed = 123456
        destroy_ops = [DestroyOperators.Dins,
                       DestroyOperators.Proximity_05,
                       DestroyOperators.Mutation_50,
                       DestroyOperators.Local_Branching_10,
                       DestroyOperators.Rins_25,
                       DestroyOperators.Rens_25,
                       DestroyOperators.Crossover]

        mip = create_mip_solver(instance_path, seed, mip_solver=BaseTest.mip_solver)
        instance = _Instance(mip)

        index_to_val = {0: 1.0, 1: 0.0, 2: 0.0, 3: 10.0, 4: 10.0, 5: 20.0, 6: 20.0}
        print("initial index to val:", index_to_val)

        initial2 = _State(instance, index_to_val, -40)

        # Initial solution
        initial_index_to_val, initial_obj_val = instance.initial_solve()

        # Create ALNS and add one or more destroy and repair operators
        alns = ALNS(np.random.default_rng(seed))
        for i in destroy_ops:
            alns.add_destroy_operator(i)
        alns.add_repair_operator(RepairOperators.Repair)

        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=7, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
        accept = HillClimbing()
        stop = MaxIterations(5)

        # Run the ALNS algorithm
        result = alns.iterate(initial2, selector, accept, stop)
        # Retrieve the final solution
        best = result.best_state
        print(f"Best heuristic solution objective is {best.objective()}.")
        self.is_not_worse(-40, result.best_state.objective(), "minimize")

    def test_balans_t6(self):
        # Input
        instance = "model.lp"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        # Parameters
        seed = 123456
        destroy_ops = [DestroyOperators.Dins,
                       DestroyOperators.Proximity_05,
                       DestroyOperators.Mutation_50,
                       DestroyOperators.Local_Branching_10,
                       DestroyOperators.Rins_25,
                       DestroyOperators.Rens_25,
                       DestroyOperators.Crossover]

        repair_ops = [RepairOperators.Repair]

        selector = MABSelector(scores=[5, 2, 1, 0.5], num_destroy=7, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))

        accept = HillClimbing()
        stop = MaxIterations(1)

        # Solver
        balans = Balans(destroy_ops, repair_ops, selector, accept, stop, seed, mip_solver=BaseTest.mip_solver)
        alns = ALNS(np.random.RandomState(seed))

        # Run
        result = balans.solve(instance_path)
        print("Best solution:", result.best_state.objective())

        self.assertEqual(result.best_state.objective(), 4)

    # ------------------------------------------------------------------
    # Config-based construction tests
    # ------------------------------------------------------------------

    def test_balans_default_config(self):
        """Balans() with no args should produce the same result as Balans(config=default.json)."""
        b_default = Balans()
        b_config = Balans(config=ConfigFactory.DEFAULT_CONFIG_PATH)

        self.assertEqual(b_default.seed, b_config.seed)
        self.assertEqual(b_default.n_mip_jobs, b_config.n_mip_jobs)
        self.assertEqual(b_default.mip_solver_str, b_config.mip_solver_str)
        self.assertEqual(len(b_default.destroy_ops), len(b_config.destroy_ops))
        self.assertEqual(len(b_default.repair_ops), len(b_config.repair_ops))
        self.assertEqual(type(b_default.selector).__name__, type(b_config.selector).__name__)
        self.assertEqual(type(b_default.accept).__name__, type(b_config.accept).__name__)
        self.assertEqual(type(b_default.stop).__name__, type(b_config.stop).__name__)

    def test_balans_default_config_positional(self):
        """Balans(config='configs/default.json') should work."""
        b = Balans(config=ConfigFactory.DEFAULT_CONFIG_PATH)
        self.assertEqual(b.seed, 1283)
        self.assertEqual(len(b.destroy_ops), 10)

    def test_balans_config_custom(self):
        """Balans(config='test_config.json') should load operators, selector, accept, stop, and constants."""
        config_path = os.path.normpath(os.path.join(Constants.DATA_TEST, '..', 'configs', 'test_config.json'))
        b = Balans(config=config_path)

        # Verify operators
        self.assertEqual(len(b.destroy_ops), 3)
        self.assertEqual(b.destroy_ops[0].__name__, "crossover")
        self.assertEqual(b.destroy_ops[1].__name__, "mutation_50")
        self.assertEqual(b.destroy_ops[2].__name__, "rins_25")

        # Verify selector
        self.assertIsInstance(b.selector, MABSelector)
        self.assertEqual(b.selector.scores, [5, 2, 1, 0])

        # Verify acceptance
        self.assertIsInstance(b.accept, HillClimbing)

        # Verify stop
        self.assertIsInstance(b.stop, MaxIterations)
        self.assertEqual(b.stop.max_iterations, 1)

        # Verify scalars
        self.assertEqual(b.seed, 42)


    def test_balans_config_with_explicit_override(self):
        """Explicit keyword args should override config values."""
        config_path = os.path.normpath(os.path.join(Constants.DATA_TEST, '..', 'configs', 'test_config.json'))
        b = Balans(config=config_path, seed=99999, stop=MaxIterations(5))

        # Seed overridden
        self.assertEqual(b.seed, 99999)
        # Stop overridden
        self.assertEqual(b.stop.max_iterations, 5)
        # Operators still from config
        self.assertEqual(len(b.destroy_ops), 3)

    def test_config_factory_load(self):
        """ConfigFactory.load should return a well-formed dict."""
        cfg = ConfigFactory.load(ConfigFactory.DEFAULT_CONFIG_PATH)
        self.assertIn('destroy_ops', cfg)
        self.assertIn('repair_ops', cfg)
        self.assertIn('selector', cfg)
        self.assertIn('accept', cfg)
        self.assertIn('stop', cfg)
        self.assertIn('seed', cfg)
        self.assertEqual(len(cfg['destroy_ops']), 10)

    def test_balans_config_invalid_operator(self):
        """Config with an unknown operator name should raise ValueError."""
        bad_config = {
            "destroy_ops": ["NonExistentOperator"],
            "repair_ops": ["Repair"],
            "stop": {"type": "MaxIterations", "max_iterations": 1}
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(bad_config, f)
            f.flush()
            with self.assertRaises(ValueError):
                Balans(config=f.name)
        os.unlink(f.name)

    def test_balans_backward_compat_positional(self):
        """Old-style Balans(destroy_list, repair_list, ...) should still work."""
        destroy_ops = [DestroyOperators.Crossover, DestroyOperators.Mutation_50]
        repair_ops = [RepairOperators.Repair]
        selector = MABSelector(scores=[1, 1, 0, 0], num_destroy=2, num_repair=1,
                               learning_policy=LearningPolicy.ThompsonSampling())
        accept = HillClimbing()
        stop = MaxIterations(1)
        seed = 42

        b = Balans(destroy_ops, repair_ops, selector, accept, stop, seed, mip_solver="gurobi")
        self.assertEqual(b.seed, 42)
        self.assertEqual(len(b.destroy_ops), 2)
        self.assertIsInstance(b.accept, HillClimbing)

