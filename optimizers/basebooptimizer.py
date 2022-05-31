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
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions
from .acquisition_functions.Knowledge_gradient import evaluate, _get_best_xstar

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
            n_pairs: int = 0,
            n_init: int = 20,
            optional: Optional[dict[str, int]] = None,
    ):
        """
        kernel_str: string, SE or Matern
        n_ms: int, number of multi starts in Adam
        adam_iters: number of iterations for each atam ruin
        """

        self.acquisition_fun = acquisitionfun
        super().__init__(testfun, lb, ub, n_max, n_init, n_pairs, ns0=n_init)

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
    def _sgd_optimize_aqc_fun(self, acq_fun: callable, **kwargs) -> [Tensor, Tensor]:
        """Use multi-start Adam SGD over multiple seeds"""

        bounds_normalized = torch.vstack([torch.zeros(self.dim), torch.ones(self.dim)])

        raw_initial_conditions = gen_one_shot_kg_initial_conditions(
            acq_function=acq_fun,
            bounds=bounds_normalized,
            q=1,
            num_restarts=self.optional["NUM_RESTARTS"],
            raw_samples=self.optional["RAW_SAMPLES"])


        x_recommended_expanded = _get_best_xstar(model=acq_fun.model,
                                                 sampler=acq_fun.sampler,
                                                 objective=acq_fun.objective,
                                                 posterior_transform=acq_fun.posterior_transform,
                                                 inner_sampler=acq_fun.inner_sampler,
                                                 X=self.x_recommended,
                                                 bounds=bounds_normalized)


        ic = torch.cat([raw_initial_conditions, x_recommended_expanded ])

        x_best, x_best_value = optimize_acqf(
            acq_function=acq_fun,
            bounds=bounds_normalized,
            batch_initial_conditions=ic,
            q=1,
            num_restarts=self.optional["NUM_RESTARTS"],
            raw_samples=self.optional["RAW_SAMPLES"],
            # optimizer=torch.optim.Adam,
            return_full_tree=False,
        )


        # x_best_value = evaluate(model=acq_fun.model,
        #                         sampler=acq_fun.sampler,
        #                         objective=acq_fun.objective,
        #                         posterior_transform=acq_fun.posterior_transform,
        #                         inner_sampler=acq_fun.inner_sampler,
        #                         X=x_best,
        #                         bounds=bounds_normalized,
        #                         x_current_best=self.x_recommended)

        # with torch.no_grad():
        #     X_random_initial_conditions_raw = torch.rand((1000, 1, self.dim))
        #
        #     X_initial_conditions_raw = X_random_initial_conditions_raw
        #
        #     from .acquisition_functions.VoI_simulator import ExpectedPosteriorMean
        #     expected_posterior_mean_objective = ExpectedPosteriorMean(model=self.model,
        #                                                               objective=acq_fun.objective)
        #     print("x_best", self.x_recommended)
        #     print("OBJECTIVE ACQ VALUE ", expected_posterior_mean_objective(self.x_recommended))
        #     x_train_posterior_mean = expected_posterior_mean_objective.forward(X_initial_conditions_raw).squeeze()
        #
        #     import matplotlib.pyplot as plt
        #     plt.scatter(X_initial_conditions_raw[:, 0, 0], X_initial_conditions_raw[:, 0, 1], c=x_train_posterior_mean)
        #     plt.scatter(self.x_train.numpy()[:, 0], self.x_train.numpy()[:, 1], color="red", label="sampled points")
        #
        #     plt.scatter(
        #         x_best.numpy()[:, 0],
        #         x_best.numpy()[:, 1],
        #         color="black",
        #         label="one-shot kg $x^{*}$",
        #         marker="^",
        #     )
        #
        #     plt.scatter(
        #         self.x_recommended[:, 0],
        #         self.x_recommended[:, 1],
        #         color="red",
        #         label="x recommended$",
        #         marker="^",
        #     )
        #     plt.title("Design Space")
        #     plt.show()
        #
        #
        #     posterior = self.model.posterior(X=X_initial_conditions_raw)
        #     posterior_mean = posterior.mean.squeeze().detach()
        #
        #     posterior_best = self.model.posterior(X=x_best)
        #     posterior_best_mean = posterior_best.mean.detach()
        #
        #     posterior_best_recommended = self.model.posterior(X=self.x_recommended)
        #     posterior_best_recommended_mean = posterior_best_recommended.mean.detach()
        #     plt.title("Objective Space")
        #     plt.scatter(posterior_mean[:, 0], posterior_mean[:, 1], c=x_train_posterior_mean)
        #     plt.scatter(posterior_best_mean.numpy()[:, 0],
        #                 posterior_best_mean.numpy()[:, 1], color="black",
        #                 label="one-shot kg $x^{*}$",
        #                 marker="^",
        #                 )
        #     plt.scatter(posterior_best_recommended_mean[:, 0],
        #                 posterior_best_recommended_mean[:, 1], color="red",
        #                 label="best posterior mean",
        #                 marker="x",
        #                 )
        #     plt.show()
        # raise
        return x_best.squeeze(dim=-2).detach(), x_best_value

    @timeit
    def optimize_sampled_pairs_fun(self, acq_fun: callable, **kwargs) -> [Tensor, Tensor]:
        """
        obtains the best pair to get to the decision maker. Uses acquisition function acq_fun

        """

        # compute all possible pairs from sampled data
        num_y_train = len(self.y_train)
        y_train_idx = torch.arange(num_y_train)
        pairs_idx = torch.combinations(y_train_idx, r=2)  # (num_possible_combinations, 2)

        # check dominancy between pairs.
        pairs_lst = []
        pairs_idx_lst = []
        for p in pairs_idx:
            pair = self.y_train[p]  # [option_1 index, option_2 index]

            non_dominated_individual_options = is_non_dominated(Y=pair)
            if torch.sum(non_dominated_individual_options) == 2:
                pairs_lst.append(pair)
                pairs_idx_lst.append(p)


        # Filter already sampled pairs
        unsampled_pairs_lst = []
        unsampled_pairs_idx_lst = []

        for i, pairs_idx in enumerate(pairs_idx_lst):
            include = True
            for j, sampled_pairs in enumerate(self.index_pairs_sampled):

                if torch.all(pairs_idx == sampled_pairs):
                    include = False

            if include:
                pair = self.y_train[pairs_idx]
                unsampled_pairs_lst.append(pair)
                unsampled_pairs_idx_lst.append(pairs_idx)


        # Compute best pair.
        if len(unsampled_pairs_lst)==0:
            return [torch.Tensor([])],  [torch.Tensor([]), torch.Tensor([])], -torch.inf

        best_pair_idx, best_pair, voi_dm = acq_fun.find_best_pair(pairs=unsampled_pairs_lst,
                                                                  pairs_idx=unsampled_pairs_idx_lst)
        return best_pair_idx, best_pair, voi_dm
