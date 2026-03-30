import os
import unittest
import pickle

from balans.solver import ParBalans
from balans.utils import Constants
from tests.test_base import BaseTest


class ParBalansTest(BaseTest):
    BaseTest.mip_solver = Constants.scip_solver

    is_skip = True
    @unittest.skipIf(is_skip, "Skipping ParBalans test")
    def test_parbalans(self):
        # Input
        instance = "noswot.mps"
        instance_path = os.path.join(Constants.DATA_TEST, instance)

        parbalans = ParBalans(n_jobs=2,
                              n_mip_jobs=1,
                              mip_solver=BaseTest.mip_solver,
                              output_dir="./tests/scip/parbalans/",
                              balans_generator=ParBalans._generate_random_balans)
        result = parbalans.run(instance_path)

        print("Best solution[0]:", result[0])
        print("Best solution[1]:", result[1])
        # self.is_not_worse(-41, result[0], "minimize")

        with open("./tests/scip/parbalans/result_0.pkl", "rb") as file:
            result0 = pickle.load(file)
        with open("./tests/scip/parbalans/result_1.pkl", "rb") as file:
            result1 = pickle.load(file)