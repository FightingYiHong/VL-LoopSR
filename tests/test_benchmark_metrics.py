import math
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_metrics import (  # noqa: E402
    expression_complexity,
    pareto_front_indices,
    pareto_knee_index,
    regression_metrics,
    srbench_formula_recovery,
    strict_formula_recovery,
)


class RegressionMetricsTest(unittest.TestCase):
    def test_perfect_prediction(self):
        metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertEqual(metrics["mse"], 0.0)
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["nmse"], 0.0)
        self.assertEqual(metrics["nrmse"], 0.0)
        self.assertEqual(metrics["r2"], 1.0)

    def test_nmse_matches_one_minus_r2(self):
        target = np.asarray([-1.0, 0.0, 1.0, 2.0])
        prediction = np.asarray([-0.5, 0.1, 0.8, 1.7])
        metrics = regression_metrics(target, prediction)
        self.assertTrue(math.isclose(metrics["nmse"], 1.0 - metrics["r2"]))
        self.assertTrue(math.isclose(metrics["nrmse"] ** 2, metrics["nmse"]))
        self.assertTrue(math.isclose(metrics["rmse"] ** 2, metrics["mse"]))

    def test_constant_target_marks_normalized_metrics_undefined(self):
        metrics = regression_metrics([2.0, 2.0], [2.0, 2.0])
        self.assertEqual(metrics["mse"], 0.0)
        self.assertIsNone(metrics["r2"])
        self.assertIsNone(metrics["nmse"])
        self.assertIsNone(metrics["nrmse"])


class SymbolicMetricsTest(unittest.TestCase):
    def test_complexity_counts_all_expression_tree_nodes(self):
        metrics = expression_complexity("x + 2*y", ["x", "y"])
        self.assertEqual(metrics["expr_complexity"], 5)
        self.assertEqual(metrics["expr_sympy_ops"], 2)

    def test_strict_recovery_accepts_algebraic_equivalence(self):
        recovered = strict_formula_recovery(
            "(x + 1)**2",
            "x**2 + 2*x + 1",
            ["x"],
        )
        self.assertIs(recovered, True)

    def test_strict_recovery_rejects_numerical_near_match(self):
        recovered = strict_formula_recovery("1.001*x", "x", ["x"])
        self.assertIs(recovered, False)

    def test_strict_recovery_normalizes_zero_indexed_aliases(self):
        recovered = strict_formula_recovery(
            "x0*x1",
            "x1*x2",
            ["x1", "x2"],
        )
        self.assertIs(recovered, True)

    def test_srbench_recovery_accepts_additive_constant(self):
        self.assertIs(srbench_formula_recovery("x + 3", "x", ["x"]), True)
        self.assertIs(strict_formula_recovery("x + 3", "x", ["x"]), False)

    def test_srbench_recovery_accepts_nonzero_constant_ratio(self):
        self.assertIs(srbench_formula_recovery("2*x", "x", ["x"]), True)

    def test_srbench_recovery_rejects_structural_mismatch(self):
        self.assertIs(srbench_formula_recovery("x**2", "x", ["x"]), False)


class ParetoMetricsTest(unittest.TestCase):
    def test_front_excludes_dominated_points(self):
        errors = [1.0, 0.8, 0.9, 0.7]
        complexities = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(pareto_front_indices(errors, complexities), [0, 1, 3])

    def test_knee_prefers_diminishing_return_point(self):
        errors = [10.0, 4.0, 3.0, 2.9]
        complexities = [1.0, 2.0, 4.0, 10.0]
        self.assertEqual(pareto_knee_index(errors, complexities), 2)


if __name__ == "__main__":
    unittest.main()
