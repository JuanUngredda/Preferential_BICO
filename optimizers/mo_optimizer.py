import os
import pickle as pkl
from typing import Optional

import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
# from botorch.utils.multi_objective.scalarization import (
#     get_chebyshev_scalarization
# )
from .scalarization_functions.vectorised_scalarizations import get_chebyshev_scalarization
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.utils.sampling import sample_simplex
from botorch.utils.transforms import unnormalize, normalize
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from torch import Tensor
from botorch.utils import standardize
from optimizers.utils import timeit
from .basebooptimizer import BaseBOOptimizer
from botorch.acquisition import PosteriorMean
from botorch.optim import optimize_acqf
from .acquisition_functions.VoI_preferences import simulate_decision_maker_responses, check_parameter
from botorch.utils.sampling import draw_sobol_samples
from .acquisition_functions.VoI_simulator import ExpectedPosteriorMean
from botorch.acquisition.objective import GenericMCObjective
from .acquisition_functions.VoI_simulator import integrated_utility_objective


class Optimizer(BaseBOOptimizer):
    def __init__(
            self,
            testfun,
            acquisitionfun,
            recommenderfun,
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
        self.recommenderfun = recommenderfun
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

        # initialize true underlying parameter
        self.true_parameter = sample_simplex(n=1, d=self.f.num_objectives, qmc=True).squeeze()

        bounds_normalized = torch.vstack(
            [torch.zeros((1, self.dim)), torch.ones((1, self.dim))]
        )
        XDISC_normalization = draw_sobol_samples(bounds=bounds_normalized, n=1000, q=1)
        self.utility_normalizing_vectors_true_fun = self.evaluate_objective(XDISC_normalization).squeeze()

    @timeit
    def evaluate_objective(self, x: Tensor, **kwargs) -> Tensor:
        x = torch.atleast_2d(x)
        x = unnormalize(X=x, bounds=self.bounds)
        objective = self.f(x)
        return objective

    def evaluate_decision_maker(self, option_1: Tensor, option_2: Tensor, **kwargs) -> tuple[Tensor, Tensor]:

        utility_model = self.utility_model(weights=self.true_parameter,
                                           Y=self.utility_normalizing_vectors_true_fun)
        utility_option_2 = utility_model(option_2)
        utility_option_1 = utility_model(option_1)

        if utility_option_1 >= utility_option_2:
            return option_1, option_2
        else:
            return option_2, option_1

    def _update_model(self, X_train: Tensor, Y_train: Tensor):

        Y_train_standarized = Y_train  # standardize(Y_train)

        NOISE_VAR = torch.Tensor([1e-4])
        while True:
            try:
                models = []

                for i in range(Y_train_standarized.shape[-1]):
                    models.append(
                        FixedNoiseGP(X_train, Y_train_standarized[..., i: i + 1],
                                     train_Yvar=NOISE_VAR.expand_as(Y_train_standarized[..., i: i + 1])
                                     )
                    )
                self.model = ModelListGP(*models)
                mll = SumMarginalLogLikelihood(self.model.likelihood, self.model)
                fit_gpytorch_model(mll)
                break
            except:
                print("update model: increased assumed fixed noise term")
                NOISE_VAR *= 10
                print("original noise var:", 1e-4, "updated noisevar:", NOISE_VAR)

    def policy(self):

        # print(self.x_train, self.y_train, self.c_train)
        self._update_model(
            X_train=self.x_train, Y_train=self.y_train)
        x_rec = self.best_model_posterior_mean(model=self.model)
        return x_rec

    def best_model_posterior_mean(self, model):
        """find the highest predicted x to return to the user"""

        assert self.y_train is not None
        "Include data to find best posterior mean"

        bounds_normalized = torch.vstack(
            [torch.zeros((1, self.dim)), torch.ones((1, self.dim))]
        )

        # generate initialisation points
        X_random_initial_conditions_raw = torch.rand((self.optional["RAW_SAMPLES"], self.dim))
        X_sampled = self.x_train

        # print(x_GP_rec.shape, X_random_initial_conditions_raw.shape, X_sampled.shape)
        X_initial_conditions_raw = torch.concat([X_random_initial_conditions_raw, X_sampled])
        X_initial_conditions_raw = X_initial_conditions_raw.unsqueeze(dim=-2)


        expected_posterior_mean_objective = self.ExpectedUtilityFunction(utility_model=self.utility_model,
                                                                         weights=self.weights)

        with torch.no_grad():
            x_train_posterior_mean = expected_posterior_mean_objective.forward(X_initial_conditions_raw).squeeze()

        best_k_indeces = torch.argsort(x_train_posterior_mean, descending=True)[:self.optional["NUM_RESTARTS"]]
        X_initial_conditions = X_initial_conditions_raw[best_k_indeces, :]

        X_optimised, X_optimised_vals = optimize_acqf(
            acq_function=expected_posterior_mean_objective,
            bounds=bounds_normalized,
            q=1,
            batch_initial_conditions=X_initial_conditions,
            num_restarts=self.optional["NUM_RESTARTS"])

        x_best = X_optimised[torch.argmax(X_optimised_vals.squeeze())]

        # with torch.no_grad():
        #     x_best = torch.atleast_2d(x_best)
        #     X_random_initial_conditions_raw = torch.rand((1000, 1, self.dim))
        #
        #     X_initial_conditions_raw = X_random_initial_conditions_raw
        #     x_train_posterior_mean = expected_posterior_mean_objective.forward(X_initial_conditions_raw).squeeze()
        #
        #     print("x_best", x_best)
        #     print("BEST POST MEAN VALUE ", expected_posterior_mean_objective(x_best))
        #
        #     import matplotlib.pyplot as plt
        #     plt.scatter(X_initial_conditions_raw[:, 0, 0], X_initial_conditions_raw[:, 0, 1], c=x_train_posterior_mean)
        #     plt.scatter(self.x_train.numpy()[:, 0], self.x_train.numpy()[:, 1], color="red", label="sampled points")
        #
        #     print(x_best)
        #     plt.scatter(
        #         x_best.numpy()[:, 0],
        #         x_best.numpy()[:, 1],
        #         color="black",
        #         label="one-shot kg $x^{*}$",
        #         marker="^",
        #     )
        #     plt.title("Design Space")
        #     plt.show()
        #
        #     posterior = self.model.posterior(X=X_initial_conditions_raw)
        #     posterior_mean = posterior.mean.squeeze().detach()
        #
        #     posterior_best = self.model.posterior(X=x_best)
        #     posterior_best_mean = posterior_best.mean.detach()
        #
        #     plt.title("Objective Space")
        #     plt.scatter(posterior_mean[:, 0], posterior_mean[:, 1], c=x_train_posterior_mean)
        #     plt.scatter(posterior_best_mean.numpy()[:, 0],
        #                 posterior_best_mean.numpy()[:, 1], color="black",
        #                 label="one-shot kg $x^{*}$",
        #                 marker="^",
        #                 )
        #
        #     plt.show()

        return torch.atleast_2d(x_best)

    def get_next_point_simulator(self):

        self._update_model(X_train=self.x_train, Y_train=self.y_train)

        acquisition_function = self.acquisition_fun(model=self.model,
                                                    utility_model=self.utility_model,
                                                    weights=self.weights,
                                                    Y_sampled=self.utility_normalizing_vectors_true_fun,
                                                    current_best_value=self.x_recommended)

        x_new, voi_val = self._sgd_optimize_aqc_fun(
            acquisition_function, log_time=self.method_time
        )

        return x_new, voi_val

    def get_next_point_decision_maker(self) -> tuple[Tensor, list[Tensor, Tensor], Tensor]:
        self._update_preference_model()

        # find best pair

        # initialize acquisition function for dm
        if self.recommenderfun is None:
            best_pair_idx = [torch.Tensor([])]
            pair_new = [torch.Tensor([]), torch.Tensor([])]
            voi_dm = -torch.inf

        else:
            bounds_normalized = torch.vstack(
                [torch.zeros((1, self.dim)), torch.ones((1, self.dim))]
            )

            pairs_acquisition_function = self.recommenderfun(model=self.model,
                                                             utility_model=self.utility_model,
                                                             weights=self.weights,
                                                             bounds=bounds_normalized,
                                                             Y_sampled=self.utility_normalizing_vectors_true_fun,
                                                             current_best_value=self.x_recommended,
                                                             optional=self.optional)

            best_pair_idx, pair_new, voi_dm = self.optimize_sampled_pairs_fun(pairs_acquisition_function,
                                                                              log_time=self.method_time)

        return best_pair_idx, pair_new, voi_dm

    def _update_preference_model(self):
        """
        updates weights of preference model given pairwise data
        """
        bounds_normalized = torch.vstack(
            [torch.zeros((1, self.dim)), torch.ones((1, self.dim))]
        )
        XDISC_normalization = draw_sobol_samples(bounds=bounds_normalized, n=1000, q=1)
        posterior = self.model.posterior(XDISC_normalization)
        self.utility_normalizing_vectors = posterior.mean.squeeze()

        # if there is no training data then only retrieve prior weights
        if len(self.y_train_option_1) == 0:
            prior_weights = sample_simplex(n=self.num_scalarisations, d=self.f.num_objectives, qmc=True).squeeze()
            self.weights = prior_weights
        else:
            # iterate over prior weights until finding suitable weights that are compatible with the data

            weights = torch.zeros((self.num_scalarisations, self.f.num_objectives))
            idx = 0
            while True:
                prior_weight = sample_simplex(n=1, d=self.f.num_objectives, qmc=True).squeeze()
                include_point = check_parameter(utility_model=self.utility_model,
                                                winner_tensor=self.y_train_option_1,
                                                loser_tensor=self.y_train_option_2,
                                                weight=prior_weight.squeeze(),
                                                y_train=self.utility_normalizing_vectors_true_fun)

                if include_point == 1:
                    weights[idx] = prior_weight
                    idx += 1
                    if idx == self.num_scalarisations:
                        break

            self.weights = weights[:self.num_scalarisations, :]

            import matplotlib.pyplot as plt
            plt.hist(weights[:, 0])
            plt.xlim(torch.min(weights[:, 0]), torch.max(weights[:, 0]))
            plt.show()
            # raise

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

    def ExpectedUtilityFunction(self,
                                utility_model,
                                weights):

        integrated_objective = integrated_utility_objective(utility_model=utility_model,
                                                            weights=weights,
                                                            Y_sampled=self.utility_normalizing_vectors_true_fun)

        Expected_Utility = GenericMCObjective(integrated_objective)
        expected_posterior_mean_objective = ExpectedPosteriorMean(model=self.model,
                                                                  objective=Expected_Utility)

        return expected_posterior_mean_objective



    def _optimize_true_underlying_performance(self,true_weight):

        bounds_normalized = torch.vstack(
            [torch.zeros((1, self.dim)), torch.ones((1, self.dim))]
        )

        # generate initialisation points
        X_random_initial_conditions_raw = torch.rand((self.optional["RAW_SAMPLES"], self.dim))
        X_sampled = self.x_train

        # print(x_GP_rec.shape, X_random_initial_conditions_raw.shape, X_sampled.shape)
        X_initial_conditions_raw = torch.concat([X_random_initial_conditions_raw, X_sampled])
        X_initial_conditions_raw = torch.concat([X_initial_conditions_raw , self.x_recommended])
        X_initial_conditions_raw = X_initial_conditions_raw.unsqueeze(dim=-2)

        def objective(X):
            utility = self.utility_model(weights=true_weight, Y=self.utility_normalizing_vectors_true_fun)
            objective_values = self.evaluate_objective(X)
            uvals = utility(objective_values )
            return uvals

        with torch.no_grad():
            x_train_posterior_mean = objective(X_initial_conditions_raw).squeeze()

        best_k_indeces = torch.argsort(x_train_posterior_mean, descending=True)[:self.optional["NUM_RESTARTS"]]
        X_initial_conditions = X_initial_conditions_raw[best_k_indeces, :]

        X_optimised, X_optimised_vals = optimize_acqf(
            acq_function=objective,
            bounds=bounds_normalized,
            q=1,
            batch_initial_conditions=X_initial_conditions,
            num_restarts=self.optional["NUM_RESTARTS"])

        x_best = X_optimised[torch.argmax(X_optimised_vals.squeeze())]
        x_value = torch.max(X_optimised_vals.squeeze())

        x_best = torch.atleast_2d(x_best)

        # with torch.no_grad():
        #     X_random_initial_conditions_raw = torch.rand((1000, 1, self.dim))
        #
        #     X_initial_conditions_raw = X_random_initial_conditions_raw
        #     x_train_posterior_mean = objective(X_initial_conditions_raw).squeeze()
        #
        #     print("x_best", x_best)
        #     print("BEST POST MEAN VALUE ", objective(x_best))
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
        #     plt.title("Design Space")
        #     plt.show()
        #
        #
        #     posterior_mean = self.evaluate_objective(X_initial_conditions_raw).squeeze()
        #
        #     posterior_best_mean =  self.evaluate_objective(x_best)
        #     print("posterior_mean",posterior_mean.shape)
        #     plt.title("Objective Space")
        #     plt.scatter(posterior_mean[:, 0],
        #                 posterior_mean[:, 1],
        #                 c=x_train_posterior_mean)
        #     print("posterior_best_mean",posterior_best_mean.shape)
        #     plt.scatter(posterior_best_mean.numpy()[:, 0],
        #                 posterior_best_mean.numpy()[:, 1], color="black",
        #                 label="one-shot kg $x^{*}$",
        #                 marker="^",
        #                 )
        #
        #     plt.show()
        return x_best, x_value

    def test(self):
        """
        test and saves performance measures
        """
        x_rec = self.policy()
        self.x_recommended = x_rec

        expectedutilityfunction = self.ExpectedUtilityFunction(utility_model=self.utility_model,
                                                               weights=self.true_parameter)

        true_sampled_utility = expectedutilityfunction(torch.atleast_2d(self.x_train[-1]))
        true_recommended_utility = expectedutilityfunction(torch.atleast_2d(x_rec))

        _, true_best_utility = self. _optimize_true_underlying_performance(true_weight=self.true_parameter)
        n = len(self.y_train) * 1.0

        OC_sampled = true_best_utility - true_sampled_utility
        OC_recommended = true_best_utility - true_recommended_utility
        print("OC_sampled", OC_sampled)
        print("OC_recommended", OC_recommended)
        self.sampled_performance = torch.vstack(
            [self.sampled_performance, torch.Tensor([n, OC_sampled])]
        )

        self.GP_performance = torch.vstack(
            [self.GP_performance, torch.Tensor([n, OC_recommended])]
        )

        self.save()
