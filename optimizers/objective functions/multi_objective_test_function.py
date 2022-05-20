"""
Synthetic functions created by myself
"""

from botorch.test_functions.multi_objective import MultiObjectiveTestProblem
from typing import Optional
from torch import Tensor
import torch
import math


class Spherical(MultiObjectiveTestProblem):
    r"""Spherical test problem.
    """

    _ref_val = 0.0

    def __init__(
            self,
            dim: int,
            num_objectives: int = 2,
            shift_factor: Optional[float] = None,
            noise_std: Optional[float] = None,
            negate: bool = False,
    ) -> None:

        self.num_objectives = num_objectives
        self.shift_factor = shift_factor
        self.dim = dim
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        self._ref_point = [self._ref_val for _ in range(num_objectives)]
        self.combinations = torch.combinations(torch.tensor([1 , -1]), r=self.dim, with_replacement=True)
        super().__init__(noise_std=noise_std, negate=negate)

    @property
    def _max_hv(self) -> float:
        pass

    def evaluate_true(self, X: Tensor) -> Tensor:

        fs = []
        for i in range(self.num_objectives):
            obj_shifting_coords = self.combinations[i]
            X_obj_i = torch.zeros(X.shape)
            for xi in range(self.dim):
                obj_shift_factor = 0.5 + self.shift_factor * obj_shifting_coords[xi]

                X_obj_i[...,xi] = X[..., xi] - obj_shift_factor

            f_i = torch.sum(X_obj_i**2, dim=-1)
            fs.append(f_i)
        return torch.stack(fs, dim=-1)
