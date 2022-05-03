import os
import pickle as pkl
from typing import Optional

import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.multi_objective.scalarization import (
    get_chebyshev_scalarization,
    get_linear_scalarization,
)
from botorch.utils.sampling import sample_simplex
from botorch.utils.transforms import unnormalize, normalize
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from torch import Tensor
from botorch.utils import standardize

from .basebooptimizer import BaseBOOptimizer
from .utils import timeit, ParetoFrontApproximation, _compute_expected_utility


class Optimizer(BaseBOOptimizer):
    def __init__(
        self,
        testfun,
        acquisitionfun,
        lb,
        ub,
        utility_model_name: str,
        num_scalarizations: int,
        n_max: int,
        n_init: int = 20,
        kernel_str: str = None,
        nz: int = 5,
        base_seed: Optional[int] = 0,
        save_folder: Optional[str] = None,
        optional: Optional[dict[str, int]] = None,
    ):

        super().__init__(
            testfun,
            acquisitionfun,
            lb,
            ub,
            n_max=n_max,
            n_init=n_init,
            optional=optional,
        )

        torch.manual_seed(base_seed)
        self.base_seed = base_seed
        self.nz = nz
        self.save_folder = save_folder
        self.bounds = testfun.bounds
        self.num_scalarisations = num_scalarizations
        if kernel_str == "RBF":
            self.covar_module = ScaleKernel(
                RBFKernel(ard_num_dims=self.dim),
            )
        elif kernel_str == "Matern":
            self.covar_module = ScaleKernel(
                RBFKernel(ard_num_dims=self.dim),
            )
        else:
            raise Exception("Expected RBF or Matern Kernel")

        if utility_model_name == "Tche":
            self.utility_model = get_chebyshev_scalarization

        elif utility_model_name == "Lin":
            self.utility_model = get_linear_scalarization

    @timeit
    def evaluate_objective(self, x: Tensor, **kwargs) -> Tensor:
        x = torch.atleast_2d(x)
        x = unnormalize(X=x, bounds=self.bounds)
        objective = self.f(x)
        return objective

    def evaluate_constraints(self, x: Tensor, **kwargs) -> Tensor:
        x = torch.atleast_2d(x)
        x = unnormalize(X=x, bounds=self.bounds)
        constraints = -self.f.evaluate_slack(x)
        return constraints

    def _update_model(self, X_train: Tensor, Y_train: Tensor, C_train: Tensor):

        Y_train_standarized = standardize(Y_train)
        train_joint_YC = torch.cat([Y_train_standarized, C_train], dim=-1)

        models = []
        for i in range(train_joint_YC.shape[-1]):
            models.append(
                SingleTaskGP(X_train, train_joint_YC[..., i : i + 1])
            )
        self.model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll)


    def policy(self, num_scalarizations:int):

        # print(self.x_train, self.y_train, self.c_train)
        self._update_model(
            X_train=self.x_train, Y_train=self.y_train, C_train=self.c_train
        )
        x_rec = self.best_model_posterior_mean(model=self.model, num_scalarizations=num_scalarizations)
        return x_rec

    def best_model_posterior_mean(self, model, num_scalarizations):
        """find the highest predicted x to return to the user"""

        assert self.y_train is not None
        "Include data to find best posterior mean"

        bounds_normalized = torch.vstack([torch.zeros(self.dim), torch.ones(self.dim)])

        # sample random weights
        weights = sample_simplex(
            n=num_scalarizations, d=self.f.num_objectives
        ).squeeze()

        X_pareto_solutions, _ = ParetoFrontApproximation(
            model=model,
            objective_dim=self.y_train.shape[1],
            scalatization_fun=self.utility_model,
            input_dim=self.dim,
            bounds=bounds_normalized,
            y_train=self.y_train,
            weights=weights,
            optional=self.optional,
        )

        return X_pareto_solutions, weights

    def get_next_point(self):
        self._update_model(X_train=self.x_train, Y_train=self.y_train, C_train=self.c_train)
        acquisition_function = self.acquisition_fun(self.model)
        x_new = self._sgd_optimize_aqc_fun(
            acquisition_function, log_time=self.method_time
        )
        return x_new

    def save(self):
        # save the output

        self.gp_likelihood_noise = [
            self.model.likelihood.likelihoods[n].noise_covar.noise
            for n in range(self.model.num_outputs)
        ]

        self.gp_lengthscales = [
            self.model.models[n].covar_module.base_kernel.lengthscale.detach()
            for n in range(self.model.num_outputs)
        ]

        self.kernel_name = str(
            self.model.models[0].covar_module.base_kernel.__class__.__name__
        )

        output = {
            "problem": self.f.problem,
            "method_times": self.method_time,
            "OC_GP": self.GP_performance,
            "OC_sampled": self.sampled_performance,
            "x": self.x_train,
            "y": self.y_train,
            "c": self.c_train,
            "x_pareto_recommended": self.pareto_set_recommended,
            "weights": self.weights,
            "kernel": self.kernel_name,
            "gp_lik_noise": self.gp_likelihood_noise,
            "gp_lengthscales": self.gp_lengthscales,
            "base_seed": self.base_seed,
            "cwd": os.getcwd(),
            "savefile": self.save_folder,
        }

        if self.save_folder is not None:
            if os.path.isdir(self.save_folder) == False:
                os.makedirs(self.save_folder)

            with open(self.save_folder + "/" + str(self.base_seed) + ".pkl", "wb") as f:
                pkl.dump(output, f)

    def test(self):
        """
        test and saves performance measures
        """
        x_rec, weights = self.policy(num_scalarizations=self.x_train.shape[0])
        self.pareto_set_recommended = x_rec
        self.weights = weights

        y_pareto_values = torch.vstack([self.evaluate_objective(x_i) for x_i in x_rec])
        c_pareto_values = torch.vstack(
            [self.evaluate_constraints(x_i) for x_i in x_rec]
        )

        expected_sampled_utility = _compute_expected_utility(
            scalatization_fun=self.utility_model,
            y_values=self.y_train,
            c_values=self.c_train,
            weights=weights,
        )

        expected_PF_utility = _compute_expected_utility(
            scalatization_fun=self.utility_model,
            y_values=y_pareto_values,
            c_values=c_pareto_values,
            weights=weights,
        )

        n = len(self.y_train) * 1.0
        self.GP_performance = torch.vstack(
            [self.GP_performance, torch.Tensor([n, expected_PF_utility ])]
        )

        self.sampled_performance = torch.vstack(
            [self.sampled_performance , torch.Tensor([n, expected_sampled_utility])]
        )
        self.save()
