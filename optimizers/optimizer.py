import os
import pickle as pkl
from typing import Optional

import torch
from botorch.acquisition import PosteriorMean
from botorch.fit import fit_gpytorch_model
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.utils import standardize
from botorch.utils.transforms import unnormalize, normalize
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from .basebooptimizer import BaseBOOptimizer
from .utils import timeit


class Optimizer(BaseBOOptimizer):
    def __init__(
        self,
        testfun,
        acquisitionfun,
        lb,
        ub,
        n_max: int,
        n_init: int = 20,
        kernel_str: str = None,
        nz: int = 5,
        base_seed: Optional[int] = 0,
        save_folder: Optional[str] = None,
        optional: Optional[dict[str, int]] = None,
    ):

        super().__init__(
            testfun=testfun,
            acquisitionfun=acquisitionfun,
            lb=lb,
            ub=ub,
            n_max=n_max,
            n_init=n_init,
            optional=optional,
        )

        torch.manual_seed(base_seed)
        self.base_seed = base_seed
        self.nz = nz
        self.save_folder = save_folder
        self.bounds = testfun.bounds
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

    @timeit
    def evaluate_objective(self, x: Tensor, **kwargs) -> Tensor:
        x = unnormalize(X=x, bounds=self.bounds)
        y = self.f(x)
        return y

    def _update_model(self, X_train: Tensor, Y_train: Tensor):
        X_train_normalized = normalize(X=X_train, bounds=self.bounds)
        Y_train_standarized = standardize(Y_train)

        if self.optional["NOISE_OBJECTIVE"]:
            self.model = SingleTaskGP(
                train_X=X_train_normalized,
                train_Y=Y_train_standarized,
                covar_module=self.covar_module,
            )
        else:
            NOISE_VAR = torch.Tensor([1e-4])

            self.model = FixedNoiseGP(
                train_X=X_train_normalized,
                train_Y=Y_train_standarized,
                covar_module=self.covar_module,
                train_Yvar=NOISE_VAR.expand_as(Y_train_standarized),
            )

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll)

    def policy(self):

        self._update_model(self.x_train, self.y_train)
        x_rec = self.best_model_posterior_mean(model=self.model)
        return x_rec

    def best_model_posterior_mean(self, model):
        """find the highest predicted x to return to the user"""

        assert self.y_train is not None
        "Include data to find best posterior mean"

        bounds_normalized = torch.vstack([torch.zeros(self.dim), torch.ones(self.dim)])
        print(bounds_normalized)
        raise
        # generate initialisation points
        batch_initial_conditions = gen_batch_initial_conditions(
            acq_function=PosteriorMean(model),
            bounds=bounds_normalized,
            q=1,
            num_restarts=self.optional["NUM_RESTARTS"],
            raw_samples=self.optional["RAW_SAMPLES"],
        )

        # making sure that the posterior mean is at least higher when compared to the sampled solutions.
        x_train_normalized = normalize(X=self.x_train, bounds=self.bounds)
        x_train_posterior_mean = PosteriorMean(model).forward(
            x_train_normalized[:, None, :]
        )
        argmax_sampled_pmean = x_train_normalized[
            x_train_posterior_mean.argmax(), :
        ].clone()

        x_candidates = torch.cat(
            (argmax_sampled_pmean[None, None, :], batch_initial_conditions), dim=0
        )

        argmax_pmean, _ = optimize_acqf(
            acq_function=PosteriorMean(model),
            bounds=bounds_normalized,
            batch_initial_conditions=x_candidates,
            q=1,
            num_restarts=self.optional["NUM_RESTARTS"],
            raw_samples=self.optional["RAW_SAMPLES"],
        )

        return argmax_pmean

    def get_next_point_simulator(self):
        self._update_model(self.x_train, self.y_train)
        acquisition_function = self.acquisition_fun(self.model)
        x_new = self._sgd_optimize_aqc_fun(
            acquisition_function, log_time=self.method_time
        )
        return x_new

    def save(self):
        # save the output
        ynoise = torch.unique(self.model.likelihood.noise_covar.noise)
        gp_likelihood_noise = torch.Tensor([ynoise])
        gp_lengthscales = self.model.covar_module.base_kernel.lengthscale.detach()
        self.gp_likelihood_noise = torch.cat(
            [self.gp_likelihood_noise, gp_likelihood_noise]
        )
        self.gp_lengthscales = torch.cat([self.gp_lengthscales, gp_lengthscales])
        self.kernel_name = str(self.model.covar_module.base_kernel.__class__.__name__)

        output = {
            "problem": self.f.problem,
            "method_times": self.method_time,
            "OC": self.performance,
            "x": self.x_train,
            "y": self.y_train,
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
