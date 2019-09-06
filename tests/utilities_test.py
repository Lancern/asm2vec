import unittest as ut

import asm2vec.internal.util as utilities


class PermutationTest(ut.TestCase):
    def test_permute(self):
        v = [10, 20, 30, 40, 50]
        p = [2, 4, 1, 0, 3]
        pv = utilities.permute(v, p)
        self.assertListEqual([30, 50, 20, 10, 40], pv, 'Permutated vectors not equal.')

    def test_inv_permute(self):
        v = [30, 50, 20, 10, 40]
        p = [2, 4, 1, 0, 3]
        pv = utilities.inverse_permute(v, p)
        self.assertListEqual([10, 20, 30, 40, 50], pv, 'Inverse permutated vectors not equal.')
