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

        with torch.no_grad():

            X_initial_conditions_raw, _, _ = acq_fun._initialize_maKG_parameters(model=self.model)

            mu_val_initial_conditions_raw = acq_fun.forward(X_initial_conditions_raw)

            best_k_indeces = torch.argsort(mu_val_initial_conditions_raw, descending=True)[
                             : self.optional["NUM_RESTARTS"]
                             ].squeeze()

            X_initial_conditions = X_initial_conditions_raw[best_k_indeces, :]

        # This optimizer uses "L-BFGS-B" by default. If specified, optimizer is Adam.
        if self.optional["OPTIMIZER"] == "Adam":
            x_best, _ = gen_candidates_torch(
                initial_conditions=X_initial_conditions.unsqueeze(dim=-2),
                acquisition_function=acq_fun,
                lower_bounds=bounds_normalized[0, :],
                upper_bounds=bounds_normalized[1, :],
                optimizer=torch.optim.Adam,
            )
        else:
            x_best, _ = gen_candidates_scipy(
                acquisition_function=acq_fun,
                initial_conditions=X_initial_conditions.unsqueeze(dim=-2),
                lower_bounds=torch.zeros(self.dim),
                upper_bounds=torch.ones(self.dim)
            )

        # print("X_initial_conditions", X_initial_conditions, "x_best",x_best, "_")
        # posterior_best = self.model.posterior(x_best)
        # mean_best = posterior_best.mean.squeeze().detach().numpy()
        # print("mean_best",mean_best, "_", _)
        # raise
        # with torch.no_grad():
        #     plot_X = torch.rand((1000, 1, 1, 4))
        #     posterior = self.model.posterior(plot_X)
        #     mean = posterior.mean.squeeze().detach().numpy()
        #     is_feas = (mean[:, 2] <= 0)
        #     import matplotlib.pyplot as plt
        #     plt.scatter(mean[is_feas, 0], mean[is_feas, 1], c=mean[is_feas, 2])
        #     plt.show()
        #
        #     acq_vals = acq_fun.forward(plot_X).squeeze().detach().numpy()
        #     posterior_best = self.model.posterior(x_best)
        #     mean_best = posterior_best.mean.squeeze().detach().numpy()
        #     print("mean_best", mean_best)
        #     plt.scatter(mean[:, 0], mean[:, 1], c=acq_vals)
        #     plt.scatter(mean_best[0], mean_best[1], color="red")
        #     plt.show()
        #     raise
        print("x_best", x_best, "value", _)
        return x_best.squeeze(dim=-2).detach()
