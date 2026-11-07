import unittest
import numpy as np
from uncertainty import force_disagreement


class TestForceDisagreement(unittest.TestCase):
    def test_per_atom_vector_rms(self):
        forces = np.array([[[1, 0, 0], [0, 2, 0]],
                           [[-1, 0, 0], [0, 2, 0]],
                           [[1, 0, 0], [0, 2, 0]],
                           [[-1, 0, 0], [0, 2, 0]]], dtype=float)
        np.testing.assert_allclose(force_disagreement(forces), [1, 0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
