import json
import os
import tempfile
import unittest

from alns.accept import (AlwaysAccept, GreatDeluge, HillClimbing,
                          RandomAccept, RecordToRecordTravel, SimulatedAnnealing)
from alns.select import AlphaUCB, MABSelector, RandomSelect, RouletteWheel
from alns.stop import MaxIterations, MaxRuntime, NoImprovement
from mabwiser.mab import LearningPolicy

from balans.solver import Balans, DestroyOperators, RepairOperators
from balans.utils import Constants, ConfigFactory
from tests.test_base import BaseTest


class BalansConstructGurobiTest(BaseTest):
    """Tests for Balans constructor (construction) correctness using the Gurobi backend."""

    mip_solver = Constants.gurobi_solver

    # ==================================================================
    # 1. Default construction
    # ==================================================================

    def test_default_construction(self):
        """Balans() with no arguments should succeed using default config."""
        b = Balans()
        self.assertIsNotNone(b)
        self.assertIsInstance(b.seed, int)
        self.assertIsInstance(b.n_mip_jobs, int)
        self.assertIsInstance(b.mip_solver_str, str)
        self.assertIsNotNone(b.destroy_ops)
        self.assertIsNotNone(b.repair_ops)
        self.assertIsNotNone(b.selector)
        self.assertIsNotNone(b.accept)
        self.assertIsNotNone(b.stop)

    def test_default_construction_values(self):
        """Balans() should match default.json values."""
        b = Balans()
        self.assertEqual(b.seed, 1283)
        self.assertEqual(b.n_mip_jobs, 1)
        self.assertEqual(b.mip_solver_str, "scip")
        self.assertEqual(len(b.destroy_ops), 10)
        self.assertEqual(len(b.repair_ops), 1)

    def test_default_selector_type(self):
        """Default selector should be MABSelector."""
        b = Balans()
        self.assertIsInstance(b.selector, MABSelector)

    def test_default_accept_type(self):
        """Default acceptance should be HillClimbing."""
        b = Balans()
        self.assertIsInstance(b.accept, HillClimbing)

    def test_default_stop_type(self):
        """Default stop should be MaxRuntime (as defined in default.json)."""
        b = Balans()
        self.assertIsInstance(b.stop, MaxRuntime)

    # ==================================================================
    # 2. Explicit positional construction (backward compatibility)
    # ==================================================================

    def test_explicit_positional_construction(self):
        """Old-style Balans(destroy, repair, selector, accept, stop, seed) should work."""
        destroy_ops = [DestroyOperators.Crossover, DestroyOperators.Mutation_50]
        repair_ops = [RepairOperators.Repair]
        selector = MABSelector(scores=[1, 1, 0, 0], num_destroy=2, num_repair=1,
                               learning_policy=LearningPolicy.ThompsonSampling())
        accept = HillClimbing()
        stop = MaxIterations(5)
        seed = 42

        b = Balans(destroy_ops, repair_ops, selector, accept, stop, seed, mip_solver=self.mip_solver)
        self.assertEqual(b.seed, 42)
        self.assertEqual(len(b.destroy_ops), 2)
        self.assertEqual(len(b.repair_ops), 1)
        self.assertIsInstance(b.accept, HillClimbing)
        self.assertIsInstance(b.stop, MaxIterations)
        self.assertEqual(b.stop.max_iterations, 5)
        self.assertEqual(b.mip_solver_str, "gurobi")

    def test_explicit_all_destroy_families(self):
        """Construction with one operator from each destroy family."""
        destroy_ops = [
            DestroyOperators.Crossover,
            DestroyOperators.Dins,
            DestroyOperators.Local_Branching_25,
            DestroyOperators.Mutation_50,
            DestroyOperators.Proximity_05,
            DestroyOperators.Rens_25,
            DestroyOperators.Rins_25,
            DestroyOperators.Random_Objective,
        ]
        repair_ops = [RepairOperators.Repair]
        selector = MABSelector(scores=[1, 1, 0, 0], num_destroy=len(destroy_ops), num_repair=1,
                               learning_policy=LearningPolicy.ThompsonSampling())
        accept = HillClimbing()
        stop = MaxIterations(1)

        b = Balans(destroy_ops, repair_ops, selector, accept, stop, seed=1283, mip_solver=self.mip_solver)
        self.assertEqual(len(b.destroy_ops), 8)
        # Verify each operator function name
        op_names = [op.__name__ for op in b.destroy_ops]
        self.assertIn("crossover", op_names)
        self.assertIn("dins", op_names)
        self.assertIn("local_branching_25", op_names)
        self.assertIn("mutation_50", op_names)
        self.assertIn("proximity_05", op_names)
        self.assertIn("rens_25", op_names)
        self.assertIn("rins_25", op_names)
        self.assertIn("random_objective", op_names)

    def test_explicit_single_destroy_op(self):
        """Construction with a single destroy operator."""
        destroy_ops = [DestroyOperators.Mutation_25]
        repair_ops = [RepairOperators.Repair]
        selector = MABSelector(scores=[1, 1, 0, 0], num_destroy=1, num_repair=1,
                               learning_policy=LearningPolicy.ThompsonSampling())
        accept = HillClimbing()
        stop = MaxIterations(1)

        b = Balans(destroy_ops, repair_ops, selector, accept, stop, seed=1283, mip_solver=self.mip_solver)
        self.assertEqual(len(b.destroy_ops), 1)
        self.assertEqual(b.destroy_ops[0].__name__, "mutation_25")

    # ==================================================================
    # 3. Selector types
    # ==================================================================

    def test_selector_mab_thompson(self):
        """MABSelector with ThompsonSampling."""
        destroy_ops = [DestroyOperators.Crossover, DestroyOperators.Mutation_50]
        repair_ops = [RepairOperators.Repair]
        selector = MABSelector(scores=[1, 1, 0, 0], num_destroy=2, num_repair=1,
                               learning_policy=LearningPolicy.ThompsonSampling())
        b = Balans(destroy_ops, repair_ops, selector, HillClimbing(), MaxIterations(1),
                   seed=1283, mip_solver=self.mip_solver)
        self.assertIsInstance(b.selector, MABSelector)

    def test_selector_mab_epsilon_greedy(self):
        """MABSelector with EpsilonGreedy."""
        destroy_ops = [DestroyOperators.Crossover, DestroyOperators.Mutation_50]
        repair_ops = [RepairOperators.Repair]
        selector = MABSelector(scores=[5, 2, 1, 0], num_destroy=2, num_repair=1,
                               learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15))
        b = Balans(destroy_ops, repair_ops, selector, HillClimbing(), MaxIterations(1),
                   seed=1283, mip_solver=self.mip_solver)
        self.assertIsInstance(b.selector, MABSelector)
        self.assertEqual(b.selector.scores, [5, 2, 1, 0])

    def test_selector_mab_softmax(self):
        """MABSelector with Softmax."""
        destroy_ops = [DestroyOperators.Crossover, DestroyOperators.Mutation_50]
        repair_ops = [RepairOperators.Repair]
        selector = MABSelector(scores=[3, 2, 1, 0], num_destroy=2, num_repair=1,
                               learning_policy=LearningPolicy.Softmax(tau=1.5))
        b = Balans(destroy_ops, repair_ops, selector, HillClimbing(), MaxIterations(1),
                   seed=1283, mip_solver=self.mip_solver)
        self.assertIsInstance(b.selector, MABSelector)

    def test_selector_mab_ucb1(self):
        """MABSelector with UCB1."""
        destroy_ops = [DestroyOperators.Crossover, DestroyOperators.Mutation_50]
        repair_ops = [RepairOperators.Repair]
        selector = MABSelector(scores=[3, 2, 1, 0], num_destroy=2, num_repair=1,
                               learning_policy=LearningPolicy.UCB1(alpha=1.0))
        b = Balans(destroy_ops, repair_ops, selector, HillClimbing(), MaxIterations(1),
                   seed=1283, mip_solver=self.mip_solver)
        self.assertIsInstance(b.selector, MABSelector)

    def test_selector_roulette_wheel(self):
        """RouletteWheel selector."""
        destroy_ops = [DestroyOperators.Crossover, DestroyOperators.Mutation_50]
        repair_ops = [RepairOperators.Repair]
        selector = RouletteWheel(scores=[1, 1, 0, 0], num_destroy=2, num_repair=1, decay=0.5)
        b = Balans(destroy_ops, repair_ops, selector, HillClimbing(), MaxIterations(1),
                   seed=1283, mip_solver=self.mip_solver)
        self.assertIsInstance(b.selector, RouletteWheel)

    def test_selector_random_select(self):
        """RandomSelect selector."""
        destroy_ops = [DestroyOperators.Crossover, DestroyOperators.Mutation_50]
        repair_ops = [RepairOperators.Repair]
        selector = RandomSelect(num_destroy=2, num_repair=1)
        b = Balans(destroy_ops, repair_ops, selector, HillClimbing(), MaxIterations(1),
                   seed=1283, mip_solver=self.mip_solver)
        self.assertIsInstance(b.selector, RandomSelect)

    def test_selector_alpha_ucb(self):
        """AlphaUCB selector."""
        destroy_ops = [DestroyOperators.Crossover, DestroyOperators.Mutation_50]
        repair_ops = [RepairOperators.Repair]
        selector = AlphaUCB(scores=[1, 1, 0, 0], num_destroy=2, num_repair=1, alpha=1.0)
        b = Balans(destroy_ops, repair_ops, selector, HillClimbing(), MaxIterations(1),
                   seed=1283, mip_solver=self.mip_solver)
        self.assertIsInstance(b.selector, AlphaUCB)

    # ==================================================================
    # 4. Acceptance criteria types
    # ==================================================================

    def test_accept_hill_climbing(self):
        """HillClimbing acceptance."""
        b = Balans(
            destroy_ops=[DestroyOperators.Mutation_50],
            repair_ops=[RepairOperators.Repair],
            selector=MABSelector(scores=[1, 1, 0, 0], num_destroy=1, num_repair=1,
                                 learning_policy=LearningPolicy.ThompsonSampling()),
            accept=HillClimbing(),
            stop=MaxIterations(1), seed=1283, mip_solver=self.mip_solver)
        self.assertIsInstance(b.accept, HillClimbing)

    def test_accept_simulated_annealing(self):
        """SimulatedAnnealing acceptance."""
        sa = SimulatedAnnealing(start_temperature=20, end_temperature=1, step=0.1)
        b = Balans(
            destroy_ops=[DestroyOperators.Mutation_50],
            repair_ops=[RepairOperators.Repair],
            selector=MABSelector(scores=[1, 1, 0, 0], num_destroy=1, num_repair=1,
                                 learning_policy=LearningPolicy.ThompsonSampling()),
            accept=sa,
            stop=MaxIterations(1), seed=1283, mip_solver=self.mip_solver)
        self.assertIsInstance(b.accept, SimulatedAnnealing)
        self.assertEqual(b.accept.start_temperature, 20)
        self.assertEqual(b.accept.end_temperature, 1)
        self.assertAlmostEqual(b.accept.step, 0.1)

    def test_accept_always_accept(self):
        """AlwaysAccept acceptance."""
        b = Balans(
            destroy_ops=[DestroyOperators.Mutation_50],
            repair_ops=[RepairOperators.Repair],
            selector=MABSelector(scores=[1, 1, 0, 0], num_destroy=1, num_repair=1,
                                 learning_policy=LearningPolicy.ThompsonSampling()),
            accept=AlwaysAccept(),
            stop=MaxIterations(1), seed=1283, mip_solver=self.mip_solver)
        self.assertIsInstance(b.accept, AlwaysAccept)

    def test_accept_record_to_record_travel(self):
        """RecordToRecordTravel acceptance."""
        rrt = RecordToRecordTravel(start_threshold=10, end_threshold=1, step=0.5)
        b = Balans(
            destroy_ops=[DestroyOperators.Mutation_50],
            repair_ops=[RepairOperators.Repair],
            selector=MABSelector(scores=[1, 1, 0, 0], num_destroy=1, num_repair=1,
                                 learning_policy=LearningPolicy.ThompsonSampling()),
            accept=rrt,
            stop=MaxIterations(1), seed=1283, mip_solver=self.mip_solver)
        self.assertIsInstance(b.accept, RecordToRecordTravel)

    def test_accept_great_deluge(self):
        """GreatDeluge acceptance."""
        gd = GreatDeluge(alpha=1.01, beta=0.5)
        b = Balans(
            destroy_ops=[DestroyOperators.Mutation_50],
            repair_ops=[RepairOperators.Repair],
            selector=MABSelector(scores=[1, 1, 0, 0], num_destroy=1, num_repair=1,
                                 learning_policy=LearningPolicy.ThompsonSampling()),
            accept=gd,
            stop=MaxIterations(1), seed=1283, mip_solver=self.mip_solver)
        self.assertIsInstance(b.accept, GreatDeluge)

    def test_accept_random_accept(self):
        """RandomAccept acceptance."""
        ra = RandomAccept(start_prob=0.8, end_prob=0.1, step=0.05)
        b = Balans(
            destroy_ops=[DestroyOperators.Mutation_50],
            repair_ops=[RepairOperators.Repair],
            selector=MABSelector(scores=[1, 1, 0, 0], num_destroy=1, num_repair=1,
                                 learning_policy=LearningPolicy.ThompsonSampling()),
            accept=ra,
            stop=MaxIterations(1), seed=1283, mip_solver=self.mip_solver)
        self.assertIsInstance(b.accept, RandomAccept)

    # ==================================================================
    # 5. Stopping criteria types
    # ==================================================================

    def test_stop_max_iterations(self):
        """MaxIterations stopping criterion."""
        b = Balans(
            destroy_ops=[DestroyOperators.Mutation_50],
            repair_ops=[RepairOperators.Repair],
            selector=MABSelector(scores=[1, 1, 0, 0], num_destroy=1, num_repair=1,
                                 learning_policy=LearningPolicy.ThompsonSampling()),
            accept=HillClimbing(),
            stop=MaxIterations(10), seed=1283, mip_solver=self.mip_solver)
        self.assertIsInstance(b.stop, MaxIterations)
        self.assertEqual(b.stop.max_iterations, 10)

    def test_stop_max_runtime(self):
        """MaxRuntime stopping criterion."""
        b = Balans(
            destroy_ops=[DestroyOperators.Mutation_50],
            repair_ops=[RepairOperators.Repair],
            selector=MABSelector(scores=[1, 1, 0, 0], num_destroy=1, num_repair=1,
                                 learning_policy=LearningPolicy.ThompsonSampling()),
            accept=HillClimbing(),
            stop=MaxRuntime(600), seed=1283, mip_solver=self.mip_solver)
        self.assertIsInstance(b.stop, MaxRuntime)
        self.assertEqual(b.stop.max_runtime, 600)

    def test_stop_no_improvement(self):
        """NoImprovement stopping criterion."""
        b = Balans(
            destroy_ops=[DestroyOperators.Mutation_50],
            repair_ops=[RepairOperators.Repair],
            selector=MABSelector(scores=[1, 1, 0, 0], num_destroy=1, num_repair=1,
                                 learning_policy=LearningPolicy.ThompsonSampling()),
            accept=HillClimbing(),
            stop=NoImprovement(100), seed=1283, mip_solver=self.mip_solver)
        self.assertIsInstance(b.stop, NoImprovement)

    # ==================================================================
    # 6. Seed and scalar parameters
    # ==================================================================

    def test_seed_stored_correctly(self):
        """Seed should be stored as-is on the Balans instance."""
        b = Balans(
            destroy_ops=[DestroyOperators.Mutation_50],
            repair_ops=[RepairOperators.Repair],
            selector=MABSelector(scores=[1, 1, 0, 0], num_destroy=1, num_repair=1,
                                 learning_policy=LearningPolicy.ThompsonSampling()),
            accept=HillClimbing(),
            stop=MaxIterations(1), seed=99999, mip_solver=self.mip_solver)
        self.assertEqual(b.seed, 99999)

    def test_n_mip_jobs_stored_correctly(self):
        """n_mip_jobs should be stored as-is."""
        b = Balans(
            destroy_ops=[DestroyOperators.Mutation_50],
            repair_ops=[RepairOperators.Repair],
            selector=MABSelector(scores=[1, 1, 0, 0], num_destroy=1, num_repair=1,
                                 learning_policy=LearningPolicy.ThompsonSampling()),
            accept=HillClimbing(),
            stop=MaxIterations(1), seed=1283, n_mip_jobs=4, mip_solver=self.mip_solver)
        self.assertEqual(b.n_mip_jobs, 4)

    def test_mip_solver_gurobi(self):
        """mip_solver_str should be 'gurobi'."""
        b = Balans(
            destroy_ops=[DestroyOperators.Mutation_50],
            repair_ops=[RepairOperators.Repair],
            selector=MABSelector(scores=[1, 1, 0, 0], num_destroy=1, num_repair=1,
                                 learning_policy=LearningPolicy.ThompsonSampling()),
            accept=HillClimbing(),
            stop=MaxIterations(1), seed=1283, mip_solver=self.mip_solver)
        self.assertEqual(b.mip_solver_str, "gurobi")

    # ==================================================================
    # 7. Config-based construction
    # ==================================================================

    def test_config_default_json(self):
        """Balans(config=default.json) should match Balans() defaults."""
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

    def test_config_custom(self):
        """Balans(config='test_config.json') should load all fields correctly."""
        config_path = os.path.normpath(os.path.join(Constants.DATA_TEST, '..', 'configs', 'test_config.json'))
        b = Balans(config=config_path)

        # Operators
        self.assertEqual(len(b.destroy_ops), 3)
        self.assertEqual(b.destroy_ops[0].__name__, "crossover")
        self.assertEqual(b.destroy_ops[1].__name__, "mutation_50")
        self.assertEqual(b.destroy_ops[2].__name__, "rins_25")

        # Selector
        self.assertIsInstance(b.selector, MABSelector)
        self.assertEqual(b.selector.scores, [5, 2, 1, 0])

        # Acceptance
        self.assertIsInstance(b.accept, HillClimbing)

        # Stop
        self.assertIsInstance(b.stop, MaxIterations)
        self.assertEqual(b.stop.max_iterations, 1)

        # Scalars
        self.assertEqual(b.seed, 42)

    def test_config_with_explicit_seed_override(self):
        """Explicit seed kwarg should override config seed."""
        config_path = os.path.normpath(os.path.join(Constants.DATA_TEST, '..', 'configs', 'test_config.json'))
        b = Balans(config=config_path, seed=99999)
        self.assertEqual(b.seed, 99999)
        # Operators still from config
        self.assertEqual(len(b.destroy_ops), 3)

    def test_config_with_explicit_stop_override(self):
        """Explicit stop kwarg should override config stop."""
        config_path = os.path.normpath(os.path.join(Constants.DATA_TEST, '..', 'configs', 'test_config.json'))
        b = Balans(config=config_path, stop=MaxIterations(20))
        self.assertEqual(b.stop.max_iterations, 20)

    def test_config_with_explicit_accept_override(self):
        """Explicit accept kwarg should override config accept."""
        config_path = os.path.normpath(os.path.join(Constants.DATA_TEST, '..', 'configs', 'test_config.json'))
        sa = SimulatedAnnealing(start_temperature=50, end_temperature=5, step=0.2)
        b = Balans(config=config_path, accept=sa)
        self.assertIsInstance(b.accept, SimulatedAnnealing)
        self.assertEqual(b.accept.start_temperature, 50)

    def test_config_with_explicit_destroy_override(self):
        """Explicit destroy_ops kwarg should override config operators."""
        config_path = os.path.normpath(os.path.join(Constants.DATA_TEST, '..', 'configs', 'test_config.json'))
        custom_ops = [DestroyOperators.Rens_75, DestroyOperators.Rins_50]
        b = Balans(destroy_ops=custom_ops, config=config_path)
        self.assertEqual(len(b.destroy_ops), 2)
        self.assertEqual(b.destroy_ops[0].__name__, "rens_75")
        self.assertEqual(b.destroy_ops[1].__name__, "rins_50")

    def test_config_with_explicit_repair_override(self):
        """Explicit repair_ops kwarg should override config operators."""
        config_path = os.path.normpath(os.path.join(Constants.DATA_TEST, '..', 'configs', 'test_config.json'))
        b = Balans(repair_ops=[RepairOperators.Repair], config=config_path)
        self.assertEqual(len(b.repair_ops), 1)

    def test_config_with_explicit_mip_solver_override(self):
        """Explicit mip_solver kwarg should override config mip_solver."""
        config_path = os.path.normpath(os.path.join(Constants.DATA_TEST, '..', 'configs', 'test_config.json'))
        b = Balans(config=config_path, mip_solver=self.mip_solver)
        self.assertEqual(b.mip_solver_str, "gurobi")

    # ==================================================================
    # 8. Config-based construction from temp JSON files
    # ==================================================================

    def test_config_roulette_wheel_selector(self):
        """Config with RouletteWheel selector should build correctly."""
        cfg = {
            "mip_solver": "gurobi",
            "seed": 5000,
            "destroy_ops": ["Crossover", "Mutation_25"],
            "repair_ops": ["Repair"],
            "selector": {
                "type": "RouletteWheel",
                "scores": [3, 2, 1, 0],
                "decay": 0.8
            },
            "accept": {"type": "HillClimbing"},
            "stop": {"type": "MaxIterations", "max_iterations": 3}
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(cfg, f)
            f.flush()
            b = Balans(config=f.name)
        os.unlink(f.name)

        self.assertIsInstance(b.selector, RouletteWheel)
        self.assertEqual(b.seed, 5000)
        self.assertEqual(b.mip_solver_str, "gurobi")
        self.assertEqual(len(b.destroy_ops), 2)

    def test_config_random_select_selector(self):
        """Config with RandomSelect selector should build correctly."""
        cfg = {
            "mip_solver": "gurobi",
            "destroy_ops": ["Mutation_50", "Rins_25"],
            "repair_ops": ["Repair"],
            "selector": {"type": "RandomSelect"},
            "accept": {"type": "AlwaysAccept"},
            "stop": {"type": "MaxIterations", "max_iterations": 2}
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(cfg, f)
            f.flush()
            b = Balans(config=f.name)
        os.unlink(f.name)

        self.assertIsInstance(b.selector, RandomSelect)
        self.assertIsInstance(b.accept, AlwaysAccept)

    def test_config_alpha_ucb_selector(self):
        """Config with AlphaUCB selector should build correctly."""
        cfg = {
            "mip_solver": "gurobi",
            "destroy_ops": ["Mutation_50"],
            "repair_ops": ["Repair"],
            "selector": {
                "type": "AlphaUCB",
                "scores": [1, 1, 0, 0],
                "alpha": 0.8
            },
            "accept": {"type": "HillClimbing"},
            "stop": {"type": "MaxRuntime", "max_runtime": 60}
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(cfg, f)
            f.flush()
            b = Balans(config=f.name)
        os.unlink(f.name)

        self.assertIsInstance(b.selector, AlphaUCB)
        self.assertIsInstance(b.stop, MaxRuntime)
        self.assertEqual(b.stop.max_runtime, 60)

    def test_config_simulated_annealing_accept(self):
        """Config with SimulatedAnnealing acceptance should build correctly."""
        cfg = {
            "mip_solver": "gurobi",
            "destroy_ops": ["Crossover"],
            "repair_ops": ["Repair"],
            "selector": {"type": "RandomSelect"},
            "accept": {
                "type": "SimulatedAnnealing",
                "start_temperature": 100,
                "end_temperature": 5,
                "step": 0.5
            },
            "stop": {"type": "MaxIterations", "max_iterations": 1}
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(cfg, f)
            f.flush()
            b = Balans(config=f.name)
        os.unlink(f.name)

        self.assertIsInstance(b.accept, SimulatedAnnealing)
        self.assertEqual(b.accept.start_temperature, 100)
        self.assertEqual(b.accept.end_temperature, 5)
        self.assertAlmostEqual(b.accept.step, 0.5)

    def test_config_record_to_record_travel_accept(self):
        """Config with RecordToRecordTravel acceptance should build correctly."""
        cfg = {
            "mip_solver": "gurobi",
            "destroy_ops": ["Crossover"],
            "repair_ops": ["Repair"],
            "selector": {"type": "RandomSelect"},
            "accept": {
                "type": "RecordToRecordTravel",
                "start_threshold": 10,
                "end_threshold": 1,
                "step": 0.5
            },
            "stop": {"type": "MaxIterations", "max_iterations": 1}
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(cfg, f)
            f.flush()
            b = Balans(config=f.name)
        os.unlink(f.name)

        self.assertIsInstance(b.accept, RecordToRecordTravel)

    def test_config_great_deluge_accept(self):
        """Config with GreatDeluge acceptance should build correctly."""
        cfg = {
            "mip_solver": "gurobi",
            "destroy_ops": ["Crossover"],
            "repair_ops": ["Repair"],
            "selector": {"type": "RandomSelect"},
            "accept": {
                "type": "GreatDeluge",
                "alpha": 1.01,
                "beta": 0.5
            },
            "stop": {"type": "MaxIterations", "max_iterations": 1}
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(cfg, f)
            f.flush()
            b = Balans(config=f.name)
        os.unlink(f.name)

        self.assertIsInstance(b.accept, GreatDeluge)

    def test_config_random_accept(self):
        """Config with RandomAccept acceptance should build correctly."""
        cfg = {
            "mip_solver": "gurobi",
            "destroy_ops": ["Crossover"],
            "repair_ops": ["Repair"],
            "selector": {"type": "RandomSelect"},
            "accept": {
                "type": "RandomAccept",
                "start_prob": 0.9,
                "end_prob": 0.1,
                "step": 0.01
            },
            "stop": {"type": "MaxIterations", "max_iterations": 1}
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(cfg, f)
            f.flush()
            b = Balans(config=f.name)
        os.unlink(f.name)

        self.assertIsInstance(b.accept, RandomAccept)

    def test_config_no_improvement_stop(self):
        """Config with NoImprovement stop should build correctly."""
        cfg = {
            "mip_solver": "gurobi",
            "destroy_ops": ["Crossover"],
            "repair_ops": ["Repair"],
            "selector": {"type": "RandomSelect"},
            "accept": {"type": "HillClimbing"},
            "stop": {"type": "NoImprovement", "max_iterations": 50}
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(cfg, f)
            f.flush()
            b = Balans(config=f.name)
        os.unlink(f.name)

        self.assertIsInstance(b.stop, NoImprovement)

    def test_config_constants_override(self):
        """Timelimit/M overrides in config.json should be stored on the instance, not in global Constants."""
        cfg = {
            "mip_solver": "gurobi",
            "destroy_ops": ["Crossover"],
            "repair_ops": ["Repair"],
            "selector": {"type": "RandomSelect"},
            "accept": {"type": "HillClimbing"},
            "stop": {"type": "MaxIterations", "max_iterations": 1},
            "timelimit_first_solution": 5,
            "timelimit_alns_iteration": 15,
            "timelimit_crossover_random_feasible": 10,
            "big_m": 2000
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(cfg, f)
            f.flush()
            b = Balans(config=f.name)
        os.unlink(f.name)

        # Effective values are stored as public instance attrs
        self.assertEqual(b.timelimit_first_solution, 5)
        self.assertEqual(b.timelimit_alns_iteration, 15)
        self.assertEqual(b.timelimit_crossover_random_feasible, 10)
        self.assertEqual(b.big_m, 2000)

        # The global Constants class must NOT be permanently mutated (Issue 5 fix)
        self.assertEqual(Constants.timelimit_first_solution, 20)
        self.assertEqual(Constants.timelimit_alns_iteration, 60)
        self.assertEqual(Constants.timelimit_crossover_random_feasible, 20)
        self.assertEqual(Constants.M, 1000)

    def test_config_learning_policy_epsilon_greedy(self):
        """Config with EpsilonGreedy learning policy should build correctly."""
        cfg = {
            "mip_solver": "gurobi",
            "destroy_ops": ["Crossover", "Mutation_50"],
            "repair_ops": ["Repair"],
            "selector": {
                "type": "MABSelector",
                "scores": [5, 2, 1, 0],
                "learning_policy": {
                    "type": "EpsilonGreedy",
                    "epsilon": 0.25
                }
            },
            "accept": {"type": "HillClimbing"},
            "stop": {"type": "MaxIterations", "max_iterations": 1}
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(cfg, f)
            f.flush()
            b = Balans(config=f.name)
        os.unlink(f.name)

        self.assertIsInstance(b.selector, MABSelector)
        self.assertEqual(b.selector.scores, [5, 2, 1, 0])

    def test_config_learning_policy_softmax(self):
        """Config with Softmax learning policy should build correctly."""
        cfg = {
            "mip_solver": "gurobi",
            "destroy_ops": ["Crossover"],
            "repair_ops": ["Repair"],
            "selector": {
                "type": "MABSelector",
                "scores": [3, 2, 1, 0],
                "learning_policy": {
                    "type": "Softmax",
                    "tau": 2.0
                }
            },
            "accept": {"type": "HillClimbing"},
            "stop": {"type": "MaxIterations", "max_iterations": 1}
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(cfg, f)
            f.flush()
            b = Balans(config=f.name)
        os.unlink(f.name)

        self.assertIsInstance(b.selector, MABSelector)

    def test_config_learning_policy_ucb1(self):
        """Config with UCB1 learning policy should build correctly."""
        cfg = {
            "mip_solver": "gurobi",
            "destroy_ops": ["Crossover"],
            "repair_ops": ["Repair"],
            "selector": {
                "type": "MABSelector",
                "scores": [3, 2, 1, 0],
                "learning_policy": {
                    "type": "UCB1",
                    "alpha": 1.5
                }
            },
            "accept": {"type": "HillClimbing"},
            "stop": {"type": "MaxIterations", "max_iterations": 1}
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(cfg, f)
            f.flush()
            b = Balans(config=f.name)
        os.unlink(f.name)

        self.assertIsInstance(b.selector, MABSelector)

    # ==================================================================
    # 9. ConfigFactory unit tests
    # ==================================================================

    def test_config_factory_load_keys(self):
        """ConfigFactory.load should return all expected keys."""
        cfg = ConfigFactory.load(ConfigFactory.DEFAULT_CONFIG_PATH)
        self.assertIn('destroy_ops', cfg)
        self.assertIn('repair_ops', cfg)
        self.assertIn('selector', cfg)
        self.assertIn('accept', cfg)
        self.assertIn('stop', cfg)
        self.assertIn('seed', cfg)
        self.assertIn('n_mip_jobs', cfg)
        self.assertIn('mip_solver', cfg)
        self.assertIn('timelimit_first_solution', cfg)
        self.assertIn('timelimit_alns_iteration', cfg)
        self.assertIn('timelimit_local_branching_iteration', cfg)
        self.assertIn('timelimit_crossover_random_feasible', cfg)
        self.assertIn('big_m', cfg)

    def test_config_factory_load_operator_count(self):
        """Default config should have 16 destroy operators and 1 repair operator."""
        cfg = ConfigFactory.load(ConfigFactory.DEFAULT_CONFIG_PATH)
        self.assertEqual(len(cfg['destroy_ops']), 10)
        self.assertEqual(len(cfg['repair_ops']), 1)

    def test_config_factory_build_learning_policy_thompson(self):
        """build_learning_policy should produce ThompsonSampling."""
        lp = ConfigFactory.build_learning_policy({"type": "ThompsonSampling"})
        self.assertIsInstance(lp, LearningPolicy.ThompsonSampling)

    def test_config_factory_build_learning_policy_epsilon(self):
        """build_learning_policy should produce EpsilonGreedy."""
        lp = ConfigFactory.build_learning_policy({"type": "EpsilonGreedy", "epsilon": 0.2})
        self.assertIsInstance(lp, LearningPolicy.EpsilonGreedy)

    def test_config_factory_build_learning_policy_softmax(self):
        """build_learning_policy should produce Softmax."""
        lp = ConfigFactory.build_learning_policy({"type": "Softmax", "tau": 1.5})
        self.assertIsInstance(lp, LearningPolicy.Softmax)

    def test_config_factory_build_learning_policy_ucb1(self):
        """build_learning_policy should produce UCB1."""
        lp = ConfigFactory.build_learning_policy({"type": "UCB1", "alpha": 1.0})
        self.assertIsInstance(lp, LearningPolicy.UCB1)

    def test_config_factory_build_learning_policy_unknown(self):
        """build_learning_policy should raise ValueError on unknown type."""
        with self.assertRaises(ValueError):
            ConfigFactory.build_learning_policy({"type": "UnknownPolicy"})

    def test_config_factory_build_selector_mab(self):
        """build_selector should produce MABSelector."""
        sel = ConfigFactory.build_selector(
            {"type": "MABSelector", "scores": [1, 1, 0, 0],
             "learning_policy": {"type": "ThompsonSampling"}},
            num_destroy=3, num_repair=1)
        self.assertIsInstance(sel, MABSelector)

    def test_config_factory_build_selector_roulette(self):
        """build_selector should produce RouletteWheel."""
        sel = ConfigFactory.build_selector(
            {"type": "RouletteWheel", "scores": [1, 1, 0, 0], "decay": 0.5},
            num_destroy=2, num_repair=1)
        self.assertIsInstance(sel, RouletteWheel)

    def test_config_factory_build_selector_random(self):
        """build_selector should produce RandomSelect."""
        sel = ConfigFactory.build_selector(
            {"type": "RandomSelect"},
            num_destroy=2, num_repair=1)
        self.assertIsInstance(sel, RandomSelect)

    def test_config_factory_build_selector_alpha_ucb(self):
        """build_selector should produce AlphaUCB."""
        sel = ConfigFactory.build_selector(
            {"type": "AlphaUCB", "scores": [1, 1, 0, 0], "alpha": 0.5},
            num_destroy=2, num_repair=1)
        self.assertIsInstance(sel, AlphaUCB)

    def test_config_factory_build_selector_unknown(self):
        """build_selector should raise ValueError on unknown type."""
        with self.assertRaises(ValueError):
            ConfigFactory.build_selector({"type": "UnknownSelector"}, num_destroy=1, num_repair=1)

    def test_config_factory_build_acceptance_sa(self):
        """build_acceptance should produce SimulatedAnnealing."""
        acc = ConfigFactory.build_acceptance(
            {"type": "SimulatedAnnealing", "start_temperature": 20, "end_temperature": 1, "step": 0.1})
        self.assertIsInstance(acc, SimulatedAnnealing)

    def test_config_factory_build_acceptance_hc(self):
        """build_acceptance should produce HillClimbing."""
        acc = ConfigFactory.build_acceptance({"type": "HillClimbing"})
        self.assertIsInstance(acc, HillClimbing)

    def test_config_factory_build_acceptance_rrt(self):
        """build_acceptance should produce RecordToRecordTravel."""
        acc = ConfigFactory.build_acceptance(
            {"type": "RecordToRecordTravel", "start_threshold": 10, "end_threshold": 1, "step": 0.5})
        self.assertIsInstance(acc, RecordToRecordTravel)

    def test_config_factory_build_acceptance_gd(self):
        """build_acceptance should produce GreatDeluge."""
        acc = ConfigFactory.build_acceptance({"type": "GreatDeluge", "alpha": 1.01, "beta": 0.5})
        self.assertIsInstance(acc, GreatDeluge)

    def test_config_factory_build_acceptance_ra(self):
        """build_acceptance should produce RandomAccept."""
        acc = ConfigFactory.build_acceptance(
            {"type": "RandomAccept", "start_prob": 0.8, "end_prob": 0.1, "step": 0.05})
        self.assertIsInstance(acc, RandomAccept)

    def test_config_factory_build_acceptance_always(self):
        """build_acceptance should produce AlwaysAccept."""
        acc = ConfigFactory.build_acceptance({"type": "AlwaysAccept"})
        self.assertIsInstance(acc, AlwaysAccept)

    def test_config_factory_build_acceptance_unknown(self):
        """build_acceptance should raise ValueError on unknown type."""
        with self.assertRaises(ValueError):
            ConfigFactory.build_acceptance({"type": "UnknownAcceptance"})

    def test_config_factory_build_stop_max_iter(self):
        """build_stop should produce MaxIterations."""
        s = ConfigFactory.build_stop({"type": "MaxIterations", "max_iterations": 10})
        self.assertIsInstance(s, MaxIterations)
        self.assertEqual(s.max_iterations, 10)

    def test_config_factory_build_stop_max_runtime(self):
        """build_stop should produce MaxRuntime."""
        s = ConfigFactory.build_stop({"type": "MaxRuntime", "max_runtime": 300})
        self.assertIsInstance(s, MaxRuntime)
        self.assertEqual(s.max_runtime, 300)

    def test_config_factory_build_stop_no_improvement(self):
        """build_stop should produce NoImprovement."""
        s = ConfigFactory.build_stop({"type": "NoImprovement", "max_iterations": 50})
        self.assertIsInstance(s, NoImprovement)

    def test_config_factory_build_stop_unknown(self):
        """build_stop should raise ValueError on unknown type."""
        with self.assertRaises(ValueError):
            ConfigFactory.build_stop({"type": "UnknownStop"})

    def test_config_factory_resolve_operators_destroy(self):
        """resolve_operators should map destroy operator names to functions."""
        from balans.solver import _DESTROY_OP_MAP
        ops = ConfigFactory.resolve_operators(["Crossover", "Mutation_50"], _DESTROY_OP_MAP, kind="destroy operator")
        self.assertEqual(len(ops), 2)
        self.assertEqual(ops[0].__name__, "crossover")
        self.assertEqual(ops[1].__name__, "mutation_50")

    def test_config_factory_resolve_operators_repair(self):
        """resolve_operators should map repair operator names to functions."""
        from balans.solver import _REPAIR_OP_MAP
        ops = ConfigFactory.resolve_operators(["Repair"], _REPAIR_OP_MAP, kind="repair operator")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0].__name__, "repair")

    def test_config_factory_resolve_operators_unknown(self):
        """resolve_operators should raise ValueError for unknown operator names."""
        from balans.solver import _DESTROY_OP_MAP
        with self.assertRaises(ValueError):
            ConfigFactory.resolve_operators(["DoesNotExist"], _DESTROY_OP_MAP, kind="destroy operator")

    # ==================================================================
    # 10. Invalid construction tests
    # ==================================================================

    def test_invalid_destroy_type_string(self):
        """Passing a string as destroy op should raise TypeError."""
        with self.assertRaises(TypeError):
            Balans(
                destroy_ops=[DestroyOperators.Mutation_25, "INVALID"],
                repair_ops=[RepairOperators.Repair],
                selector=MABSelector(scores=[1, 1, 0, 0], num_destroy=2, num_repair=1,
                                     learning_policy=LearningPolicy.ThompsonSampling()),
                accept=HillClimbing(),
                stop=MaxIterations(1), seed=1283, mip_solver=self.mip_solver)

    def test_invalid_destroy_type_none(self):
        """Passing None as a destroy op should raise TypeError."""
        with self.assertRaises(TypeError):
            Balans(
                destroy_ops=[DestroyOperators.Mutation_25, None],
                repair_ops=[RepairOperators.Repair],
                selector=MABSelector(scores=[1, 1, 0, 0], num_destroy=2, num_repair=1,
                                     learning_policy=LearningPolicy.ThompsonSampling()),
                accept=HillClimbing(),
                stop=MaxIterations(1), seed=1283, mip_solver=self.mip_solver)

    def test_invalid_repair_type_string(self):
        """Passing a string as repair op should raise TypeError."""
        with self.assertRaises(TypeError):
            Balans(
                destroy_ops=[DestroyOperators.Mutation_25],
                repair_ops=[RepairOperators.Repair, "INVALID"],
                selector=MABSelector(scores=[1, 1, 0, 0], num_destroy=1, num_repair=2,
                                     learning_policy=LearningPolicy.ThompsonSampling()),
                accept=HillClimbing(),
                stop=MaxIterations(1), seed=1283, mip_solver=self.mip_solver)

    def test_invalid_selector_type(self):
        """Passing a string as selector should raise TypeError."""
        with self.assertRaises(TypeError):
            Balans(
                destroy_ops=[DestroyOperators.Mutation_25],
                repair_ops=[RepairOperators.Repair],
                selector="INVALID",
                accept=HillClimbing(),
                stop=MaxIterations(1), seed=1283, mip_solver=self.mip_solver)

    def test_invalid_accept_type(self):
        """Passing a wrong type as accept should raise TypeError."""
        with self.assertRaises(TypeError):
            Balans(
                destroy_ops=[DestroyOperators.Mutation_25],
                repair_ops=[RepairOperators.Repair],
                selector=MABSelector(scores=[1, 1, 0, 0], num_destroy=1, num_repair=1,
                                     learning_policy=LearningPolicy.ThompsonSampling()),
                accept=MaxIterations(5),  # wrong type
                stop=MaxIterations(1), seed=1283, mip_solver=self.mip_solver)

    def test_invalid_stop_type(self):
        """Passing a string as stop should raise TypeError."""
        with self.assertRaises(TypeError):
            Balans(
                destroy_ops=[DestroyOperators.Mutation_25],
                repair_ops=[RepairOperators.Repair],
                selector=None,
                accept=HillClimbing(),
                stop="INVALID", seed=1283, mip_solver=self.mip_solver)

    def test_invalid_mip_solver_string(self):
        """Invalid mip_solver string should raise ValueError."""
        with self.assertRaises(ValueError):
            Balans(
                destroy_ops=[DestroyOperators.Mutation_25],
                repair_ops=[RepairOperators.Repair],
                selector=MABSelector(scores=[1, 1, 0, 0], num_destroy=1, num_repair=1,
                                     learning_policy=LearningPolicy.ThompsonSampling()),
                accept=HillClimbing(),
                stop=MaxIterations(1), seed=1283, mip_solver="invalid_solver")

    def test_invalid_config_operator_name(self):
        """Config with unknown destroy operator name should raise ValueError."""
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

    def test_invalid_config_unknown_constant(self):
        """Config with unknown constant key should raise ValueError."""
        bad_config = {
            "destroy_ops": ["Crossover"],
            "repair_ops": ["Repair"],
            "accept": {"type": "HillClimbing"},
            "stop": {"type": "MaxIterations", "max_iterations": 1},
            "nonexistent_constant": 999
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(bad_config, f)
            f.flush()
            with self.assertRaises(ValueError):
                Balans(config=f.name)
        os.unlink(f.name)

    # ==================================================================
    # 11. __str__ representation
    # ==================================================================

    def test_str_contains_solver_name(self):
        """__str__ should contain MIP solver name."""
        b = Balans(
            destroy_ops=[DestroyOperators.Mutation_50],
            repair_ops=[RepairOperators.Repair],
            selector=MABSelector(scores=[1, 1, 0, 0], num_destroy=1, num_repair=1,
                                 learning_policy=LearningPolicy.ThompsonSampling()),
            accept=HillClimbing(),
            stop=MaxIterations(1), seed=1283, mip_solver=self.mip_solver)
        s = str(b)
        self.assertIn("gurobi", s)

    def test_str_contains_operator_count(self):
        """__str__ should show correct destroy operator count."""
        b = Balans(
            destroy_ops=[DestroyOperators.Mutation_50, DestroyOperators.Crossover],
            repair_ops=[RepairOperators.Repair],
            selector=MABSelector(scores=[1, 1, 0, 0], num_destroy=2, num_repair=1,
                                 learning_policy=LearningPolicy.ThompsonSampling()),
            accept=HillClimbing(),
            stop=MaxIterations(1), seed=1283, mip_solver=self.mip_solver)
        s = str(b)
        self.assertIn("2 Destroy Operators", s)
        self.assertIn("1 Repair Operators", s)

    def test_str_contains_selector_type(self):
        """__str__ should include the selector type name."""
        b = Balans(
            destroy_ops=[DestroyOperators.Mutation_50],
            repair_ops=[RepairOperators.Repair],
            selector=MABSelector(scores=[1, 1, 0, 0], num_destroy=1, num_repair=1,
                                 learning_policy=LearningPolicy.ThompsonSampling()),
            accept=HillClimbing(),
            stop=MaxIterations(1), seed=1283, mip_solver=self.mip_solver)
        s = str(b)
        self.assertIn("MABSelector", s)

    def test_str_contains_acceptance_type(self):
        """__str__ should include the acceptance criterion type name."""
        b = Balans(
            destroy_ops=[DestroyOperators.Mutation_50],
            repair_ops=[RepairOperators.Repair],
            selector=MABSelector(scores=[1, 1, 0, 0], num_destroy=1, num_repair=1,
                                 learning_policy=LearningPolicy.ThompsonSampling()),
            accept=SimulatedAnnealing(start_temperature=20, end_temperature=1, step=0.1),
            stop=MaxIterations(1), seed=1283, mip_solver=self.mip_solver)
        s = str(b)
        self.assertIn("SimulatedAnnealing", s)
        self.assertIn("Start Temperature", s)

    def test_str_contains_stop_type(self):
        """__str__ should include the stopping criterion type name."""
        b = Balans(
            destroy_ops=[DestroyOperators.Mutation_50],
            repair_ops=[RepairOperators.Repair],
            selector=MABSelector(scores=[1, 1, 0, 0], num_destroy=1, num_repair=1,
                                 learning_policy=LearningPolicy.ThompsonSampling()),
            accept=HillClimbing(),
            stop=MaxIterations(7), seed=1283, mip_solver=self.mip_solver)
        s = str(b)
        self.assertIn("MaxIterations", s)

    def test_str_contains_seed(self):
        """__str__ should include the seed value."""
        b = Balans(
            destroy_ops=[DestroyOperators.Mutation_50],
            repair_ops=[RepairOperators.Repair],
            selector=MABSelector(scores=[1, 1, 0, 0], num_destroy=1, num_repair=1,
                                 learning_policy=LearningPolicy.ThompsonSampling()),
            accept=HillClimbing(),
            stop=MaxIterations(1), seed=77777, mip_solver=self.mip_solver)
        s = str(b)
        self.assertIn("77777", s)

    def test_str_contains_operator_names(self):
        """__str__ should list individual operator function names."""
        b = Balans(
            destroy_ops=[DestroyOperators.Crossover, DestroyOperators.Rens_50],
            repair_ops=[RepairOperators.Repair],
            selector=MABSelector(scores=[1, 1, 0, 0], num_destroy=2, num_repair=1,
                                 learning_policy=LearningPolicy.ThompsonSampling()),
            accept=HillClimbing(),
            stop=MaxIterations(1), seed=1283, mip_solver=self.mip_solver)
        s = str(b)
        self.assertIn("crossover", s)
        self.assertIn("rens_50", s)
        self.assertIn("repair", s)

    # ==================================================================
    # 12. Property accessors before solve
    # ==================================================================

    def test_instance_none_before_solve(self):
        """instance property should be None before solve() is called."""
        b = Balans()
        self.assertIsNone(b.instance)

    def test_initial_index_to_val_none_before_solve(self):
        """initial_index_to_val should be None before solve() is called."""
        b = Balans()
        self.assertIsNone(b.initial_index_to_val)

    def test_initial_obj_val_none_before_solve(self):
        """initial_obj_val should be None before solve() is called."""
        b = Balans()
        self.assertIsNone(b.initial_obj_val)

    def test_alns_none_before_solve(self):
        """alns attribute should be None before solve() is called."""
        b = Balans()
        self.assertIsNone(b.alns)

    # ==================================================================
    # 13. Destroy operator coverage
    # ==================================================================

    def test_local_branching_operators_range(self):
        """All Local_Branching operators should be constructible."""
        lb_ops = [
            DestroyOperators.Local_Branching_05, DestroyOperators.Local_Branching_10,
            DestroyOperators.Local_Branching_15, DestroyOperators.Local_Branching_20,
            DestroyOperators.Local_Branching_25, DestroyOperators.Local_Branching_30,
            DestroyOperators.Local_Branching_35, DestroyOperators.Local_Branching_40,
            DestroyOperators.Local_Branching_45, DestroyOperators.Local_Branching_50,
            DestroyOperators.Local_Branching_55, DestroyOperators.Local_Branching_60,
            DestroyOperators.Local_Branching_65, DestroyOperators.Local_Branching_70,
            DestroyOperators.Local_Branching_75, DestroyOperators.Local_Branching_80,
            DestroyOperators.Local_Branching_85, DestroyOperators.Local_Branching_90,
            DestroyOperators.Local_Branching_95,
        ]
        repair_ops = [RepairOperators.Repair]
        selector = MABSelector(scores=[1, 1, 0, 0], num_destroy=len(lb_ops), num_repair=1,
                               learning_policy=LearningPolicy.ThompsonSampling())
        b = Balans(lb_ops, repair_ops, selector, HillClimbing(), MaxIterations(1),
                   seed=1283, mip_solver=self.mip_solver)
        self.assertEqual(len(b.destroy_ops), 19)

    def test_mutation_operators_range(self):
        """All Mutation operators should be constructible."""
        mut_ops = [
            DestroyOperators.Mutation_05, DestroyOperators.Mutation_10,
            DestroyOperators.Mutation_15, DestroyOperators.Mutation_20,
            DestroyOperators.Mutation_25, DestroyOperators.Mutation_30,
            DestroyOperators.Mutation_35, DestroyOperators.Mutation_40,
            DestroyOperators.Mutation_45, DestroyOperators.Mutation_50,
            DestroyOperators.Mutation_55, DestroyOperators.Mutation_60,
            DestroyOperators.Mutation_65, DestroyOperators.Mutation_70,
            DestroyOperators.Mutation_75, DestroyOperators.Mutation_80,
            DestroyOperators.Mutation_85, DestroyOperators.Mutation_90,
            DestroyOperators.Mutation_95,
        ]
        repair_ops = [RepairOperators.Repair]
        selector = MABSelector(scores=[1, 1, 0, 0], num_destroy=len(mut_ops), num_repair=1,
                               learning_policy=LearningPolicy.ThompsonSampling())
        b = Balans(mut_ops, repair_ops, selector, HillClimbing(), MaxIterations(1),
                   seed=1283, mip_solver=self.mip_solver)
        self.assertEqual(len(b.destroy_ops), 19)

    def test_proximity_operators_range(self):
        """All Proximity operators should be constructible."""
        prox_ops = [
            DestroyOperators.Proximity_005, DestroyOperators.Proximity_010,
            DestroyOperators.Proximity_015, DestroyOperators.Proximity_020,
            DestroyOperators.Proximity_025, DestroyOperators.Proximity_030,
            DestroyOperators.Proximity_035, DestroyOperators.Proximity_040,
            DestroyOperators.Proximity_045, DestroyOperators.Proximity_05,
            DestroyOperators.Proximity_055, DestroyOperators.Proximity_060,
            DestroyOperators.Proximity_065, DestroyOperators.Proximity_070,
            DestroyOperators.Proximity_075, DestroyOperators.Proximity_080,
            DestroyOperators.Proximity_085, DestroyOperators.Proximity_090,
            DestroyOperators.Proximity_095, DestroyOperators.Proximity_10,
        ]
        repair_ops = [RepairOperators.Repair]
        selector = MABSelector(scores=[1, 1, 0, 0], num_destroy=len(prox_ops), num_repair=1,
                               learning_policy=LearningPolicy.ThompsonSampling())
        b = Balans(prox_ops, repair_ops, selector, HillClimbing(), MaxIterations(1),
                   seed=1283, mip_solver=self.mip_solver)
        self.assertEqual(len(b.destroy_ops), 20)

    def test_rens_operators_range(self):
        """All Rens operators should be constructible."""
        rens_ops = [
            DestroyOperators.Rens_05, DestroyOperators.Rens_10,
            DestroyOperators.Rens_15, DestroyOperators.Rens_20,
            DestroyOperators.Rens_25, DestroyOperators.Rens_30,
            DestroyOperators.Rens_35, DestroyOperators.Rens_40,
            DestroyOperators.Rens_45, DestroyOperators.Rens_50,
            DestroyOperators.Rens_55, DestroyOperators.Rens_60,
            DestroyOperators.Rens_65, DestroyOperators.Rens_70,
            DestroyOperators.Rens_75, DestroyOperators.Rens_80,
            DestroyOperators.Rens_85, DestroyOperators.Rens_90,
            DestroyOperators.Rens_95,
        ]
        repair_ops = [RepairOperators.Repair]
        selector = MABSelector(scores=[1, 1, 0, 0], num_destroy=len(rens_ops), num_repair=1,
                               learning_policy=LearningPolicy.ThompsonSampling())
        b = Balans(rens_ops, repair_ops, selector, HillClimbing(), MaxIterations(1),
                   seed=1283, mip_solver=self.mip_solver)
        self.assertEqual(len(b.destroy_ops), 19)

    def test_rins_operators_range(self):
        """All Rins operators should be constructible."""
        rins_ops = [
            DestroyOperators.Rins_05, DestroyOperators.Rins_10,
            DestroyOperators.Rins_15, DestroyOperators.Rins_20,
            DestroyOperators.Rins_25, DestroyOperators.Rins_30,
            DestroyOperators.Rins_35, DestroyOperators.Rins_40,
            DestroyOperators.Rins_45, DestroyOperators.Rins_50,
            DestroyOperators.Rins_55, DestroyOperators.Rins_60,
            DestroyOperators.Rins_65, DestroyOperators.Rins_70,
            DestroyOperators.Rins_75, DestroyOperators.Rins_80,
            DestroyOperators.Rins_85, DestroyOperators.Rins_90,
            DestroyOperators.Rins_95,
        ]
        repair_ops = [RepairOperators.Repair]
        selector = MABSelector(scores=[1, 1, 0, 0], num_destroy=len(rins_ops), num_repair=1,
                               learning_policy=LearningPolicy.ThompsonSampling())
        b = Balans(rins_ops, repair_ops, selector, HillClimbing(), MaxIterations(1),
                   seed=1283, mip_solver=self.mip_solver)
        self.assertEqual(len(b.destroy_ops), 19)

    # ==================================================================
    # 14. Main example construction
    # ==================================================================

    def test_main_example_construction(self):
        """The main.py example configuration should construct successfully."""
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
        repair_ops = [RepairOperators.Repair]
        selector = MABSelector(scores=[1, 1, 0, 0],
                               num_destroy=len(destroy_ops),
                               num_repair=len(repair_ops),
                               learning_policy=LearningPolicy.ThompsonSampling())
        accept = SimulatedAnnealing(start_temperature=20, end_temperature=1, step=0.1)
        stop = MaxRuntime(600)

        b = Balans(destroy_ops=destroy_ops,
                   repair_ops=repair_ops,
                   selector=selector,
                   accept=accept,
                   stop=stop)
        self.assertEqual(len(b.destroy_ops), 16)
        self.assertEqual(len(b.repair_ops), 1)
        self.assertIsInstance(b.selector, MABSelector)
        self.assertIsInstance(b.accept, SimulatedAnnealing)
        self.assertIsInstance(b.stop, MaxRuntime)
        self.assertEqual(b.stop.max_runtime, 600)


if __name__ == '__main__':
    unittest.main()

