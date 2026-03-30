import unittest

from balans.utils import Constants


class BaseTest(unittest.TestCase):

    def assertIsBetter(self, before, after, sense):
        if sense == Constants.minimize:
            self.assertLess(after, before)
        else:
            self.assertGreater(after, before)

    def is_not_worse(self, before, after, sense):
        if sense == Constants.minimize:
            self.assertLessEqual(after, before)
        else:
            self.assertGreaterEqual(after, before)
