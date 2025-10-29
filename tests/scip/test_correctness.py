import os
import unittest

from alns.accept import *
from alns.select import *
from alns.stop import *
from mabwiser.mab import LearningPolicy

from balans.solver import Balans, DestroyOperators, RepairOperators
from balans.utils import Constants
from tests.test_base import BaseTest


class CorrectnessTest(BaseTest):

    BaseTest.mip_solver = Constants.scip_solver

    is_skip = True

    # TODO: implement the exact configs/runs below to run on these instances.
    # Best configs of Balans_Softmax and Balans_TS from the paper ran for 1 hour.
    @unittest.skipIf(is_skip, "Skipping correctness 1")
    def test_correctness1(self):
        # Input
        instance = "50v-10.mps"
        instance_path = os.path.join(Constants.DATA_TEST, instance)
        # Balans
        balans = Balans(destroy_ops=[DestroyOperators.Crossover,
                                     DestroyOperators.Dins,
                                     DestroyOperators.Mutation_25,
                                     DestroyOperators.Local_Branching_10,
                                     DestroyOperators.Rins_25,
                                     DestroyOperators.Proximity_05,
                                     DestroyOperators.Rens_25,
                                     DestroyOperators.Random_Objective],
                        repair_ops=[RepairOperators.Repair],
                        selector=MABSelector(scores=[5, 2, 1, 0.5], num_destroy=8, num_repair=1,
                                             learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.50)),
                        accept=HillClimbing(),
                        stop=MaxIterations(5),
                        seed=Constants.default_seed,
                        mip_solver=BaseTest.mip_solver)

        # Run
        result = balans.solve(instance_path)
        objective = result.best_state.objective()
        self.assertLess(objective, 4000)  # iter_30 --> 3500
        self.assertGreater(objective, 3311)

    @unittest.skipIf(is_skip, "Skipping correctness 2")
    def test_correctness2(self):
        # Input
        instance = "30n20b8.mps"
        instance_path = os.path.join(Constants.DATA_TEST, instance)
        # Balans
        balans = Balans(destroy_ops=[DestroyOperators.Crossover,
                                     DestroyOperators.Dins,
                                     DestroyOperators.Mutation_25,
                                     DestroyOperators.Local_Branching_10,
                                     DestroyOperators.Rins_25,
                                     DestroyOperators.Proximity_05,
                                     DestroyOperators.Rens_25,
                                     DestroyOperators.Random_Objective],
                        repair_ops=[RepairOperators.Repair],
                        selector=MABSelector(scores=[5, 2, 1, 0.5], num_destroy=8, num_repair=1,
                                             learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.50)),
                        accept=HillClimbing(),
                        stop=MaxIterations(15),
                        seed=Constants.default_seed,
                        mip_solver=BaseTest.mip_solver)

        # Run
        result = balans.solve(instance_path)
        objective = result.best_state.objective()
        self.assertLess(objective, 530)  # iter_5 --> 553, iter_6 --> 453, iter_15 --> 302
        self.assertGreater(objective, 301)

    @unittest.skipIf(is_skip, "Skipping correctness 3")
    def test_correctness3(self):
        # Input
        instance = "b1c1s1.mps"
        instance_path = os.path.join(Constants.DATA_TEST, instance)
        # Balans
        balans = Balans(destroy_ops=[DestroyOperators.Crossover,
                                     DestroyOperators.Dins,
                                     DestroyOperators.Mutation_25,
                                     DestroyOperators.Local_Branching_10,
                                     DestroyOperators.Rins_25,
                                     DestroyOperators.Proximity_05,
                                     DestroyOperators.Rens_25],
                        repair_ops=[RepairOperators.Repair],
                        selector=MABSelector(scores=[5, 2, 1, 0.5], num_destroy=7, num_repair=1,
                                             learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.50)),
                        accept=HillClimbing(),
                        stop=MaxIterations(15),
                        seed=Constants.default_seed,
                        mip_solver=BaseTest.mip_solver)

        # SK: iter_5 >>> FINISH objective: 69333.52 (no change)
        # SK: iter_10 >>> FINISH objective: 68020
        # SK: iter_15 --> 66569.81131420168

        # Run
        result = balans.solve(instance_path)
        objective = result.best_state.objective()
        self.assertLess(objective, 68550)
        self.assertGreater(objective, 24544)

    @unittest.skipIf(is_skip, "Skipping correctness 1")
    def test_correctness_gisp_102(self):
        # gisp_102.lp
        # Minimize
        # SCIP: -2246
        # Balans_Softmax: -2318
        # Balans_TS: -2322
        #
        pass

    @unittest.skipIf(is_skip, "Skipping correctness 1")
    def test_correctness_mis_2(self):
        # mis_2.lp
        # Maximize
        # SCIP: 3602
        # Balans_Softmax: 3732
        # Balans_TS: 3757
        pass

    @unittest.skipIf(is_skip, "Skipping correctness 1")
    def test_correctness_mk_5(self):
        # mk_5.lp
        # Maximize
        # SCIP: 3718
        # Balans_Softmax: 3749
        # Balans_TS: 3752
        pass

    @unittest.skipIf(is_skip, "Skipping correctness 1")
    def test_correctness_mvc_8(self):
        # mvc_8.lp
        # Minimize
        # SCIP: 2377
        # Balans_Softmax: 2365
        # Balans_TS: 2373
        pass

    @unittest.skipIf(is_skip, "Skipping correctness 1")
    def test_correctness_sc_3(self):
        # sc_3.lp
        # Minimize
        # SCIP: 172
        # Balans_Softmax: 171
        # Balans_TS: 171
        pass
