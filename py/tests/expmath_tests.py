import unittest


class RunningAverageTest(unittest.TestCase):
    def test_updating_running_average(self):
        from syndicato import expmath
        input = ((2, 0.0, 1.0), (4, -1.0, 1.0), (7, 3.0, 3.0))
        expected = (0.5, -0.5, 3.0)

        for idx, (count, prev_average, new_value) in enumerate(input):
            output = expmath.update_running_average(count, prev_average, new_value)
            self.assertEquals(expected[idx], output)
