import logging
import sys
from typing import Optional

import torch
from botorch.generation import gen_candidates_torch
from botorch.optim import gen_batch_initial_conditions
from botorch.optim import optimize_acqf
from torch import Tensor
from botorch.generation.gen import gen_candidates_scipy
from .baseoptimizer import BaseOptimizer
from .utils import timeit

LOG_FORMAT = (
    "%(asctime)s - %(name)s:%(funcName)s:%(lineno)s - %(levelname)s:  %(message)s"
)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class BaseBOOptimizer(BaseOptimizer):
    def __init__(
        self,
        testfun,
        acquisitionfun,
        lb: Tensor,
        ub: Tensor,
        n_max: int,
        n_init: int = 20,
        optional: Optional[dict[str, int]] = None,
    ):
        """
        kernel_str: string, SE or Matern
        n_ms: int, number of multi starts in Adam
        adam_iters: number of iterations for each atam ruin
        """

        self.acquisition_fun = acquisitionfun
        super().__init__(testfun, lb, ub, n_max, n_init, ns0=n_init)

        if optional is None:
            self.optional = {
                "OPTIMIZER": "Default",
                "NOISE_OBJECTIVE": None,
                "RAW_SAMPLES": 80,
                "NUM_RESTARTS": 5,
            }
        else:
            if optional["RAW_SAMPLES"] is None:
                optional["RAW_SAMPLES"] = 80

            if optional["NUM_RESTARTS"] is None:
                optional["NUM_RESTARTS"] = 5

            if optional["OPTIMIZER"] is None:
                optional["OPTIMIZER"] = "Default"

            if optional["NOISE_OBJECTIVE"] is False:
                optional["NOISE_OBJECTIVE"] = False

            self.optional = optional

    @timeit
    def _sgd_optimize_aqc_fun(self, acq_fun: callable, **kwargs) -> Tensor:
        """Use multi-start Adam SGD over multiple seeds"""

        bounds_normalized = torch.vstack([torch.zeros(self.dim), torch.ones(self.dim)])

        X_plot = torch.rand((100,1,2))
        KG_plot_vals = []
        for x in X_plot:
            KG_plot_vals.append(acq_fun.evaluate(x, bounds=bounds_normalized).detach().numpy())


        x_best, _ = optimize_acqf(
            acq_function=acq_fun,
            bounds=bounds_normalized,
            q=1,
            num_restarts=self.optional["NUM_RESTARTS"],
            raw_samples=self.optional["RAW_SAMPLES"],
            return_full_tree=False
        )


        import matplotlib.pyplot as plt
        x_best = x_best.squeeze().detach()
        plt.scatter(X_plot[:,0], X_plot[:,1], c=KG_plot_vals)
        plt.scatter(x_best[0], x_best[0])
        plt.show()
        # KG_val = acq_fun.evaluate(x_best, bounds=bounds_normalized)

        print("x_best", x_best, "value",_)
        raise
        return x_best.squeeze(dim=-2).detach()
