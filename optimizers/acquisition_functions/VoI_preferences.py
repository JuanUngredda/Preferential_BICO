from typing import Dict, Tuple, Any, Callable, Optional
import torch
from torch import Tensor
from typing import Dict, Tuple, Any, Callable, Optional
import torch
from torch import Tensor
from botorch.models.model import Model
from botorch.acquisition import qKnowledgeGradient
from botorch.acquisition.objective import GenericMCObjective
from botorch.models.converter import (
    model_list_to_batched,
)
from botorch.models.model_list_gp_regression import ModelListGP
from .VoI_simulator import integrated_utility_objective
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from botorch.utils.transforms import (
    concatenate_pending_points,
    match_batch_shape,
    t_batch_mode_transform,
)
from .VoI_simulator import ExpectedPosteriorMean
from botorch.acquisition.objective import GenericMCObjective
from .VoI_simulator import integrated_utility_objective
from botorch.optim import optimize_acqf


def check_parameter(utility_model: Callable,
                    winner_tensor: Tensor,
                    loser_tensor: Tensor,
                    weight: Tensor,
                    y_train: Tensor,
                    **kwargs) -> int:
    assert winner_tensor.shape[-1] == y_train.shape[-1], "not same number of output dims Y and option2"
    assert loser_tensor.shape[-1] == y_train.shape[-1], "not same number of output dims Y and option1"

    utility_model = utility_model(weights=weight, Y=y_train)
    utility_option_1 = utility_model(winner_tensor)
    utility_option_2 = utility_model(loser_tensor)


    # winner tensor must be greater than loser tensor for all instances.
    num_simulated_winning_instances = torch.sum(utility_option_1 > utility_option_2)
    num_actual_winning_instances = len(winner_tensor)
    if num_simulated_winning_instances == num_actual_winning_instances:
        return 1
    else:
        return 0


def simulate_decision_maker_responses(utility_model: Callable,
                                      option_1: Tensor,
                                      option_2: Tensor,
                                      weight: Tensor,
                                      y_train: Tensor,
                                      **kwargs) -> int:
    assert len(option_1.shape) == 1, "only a single isntance of option 1 is accepted"
    assert len(option_2.shape) == 1, "only a single instance of option 2 is accepted"
    assert option_2.shape[-1] == y_train.shape[-1], "not same number of output dims Y and option2"
    assert option_1.shape[-1] == y_train.shape[-1], "not same number of output dims Y and option1"

    utility_model = utility_model(weights=weight, Y=y_train)
    utility_option_1 = utility_model(option_1)
    utility_option_2 = utility_model(option_2)

    if utility_option_1 > utility_option_2:
        return 1
    else:
        return 0


def compute_acceptance_probability(utility_model: Callable,
                                   option_1: Tensor,
                                   option_2: Tensor,
                                   weights: Tensor,
                                   y_train):
    """
    computes the probability of option 1 is better than option 2
    """
    simulated_response = []

    for w in weights:
        include_point = simulate_decision_maker_responses(utility_model=utility_model,
                                                          option_1=option_1,
                                                          option_2=option_2,
                                                          weight=w,
                                                          y_train=y_train)

        simulated_response.append(include_point)
    simulated_response = torch.Tensor(simulated_response)

    return torch.mean(simulated_response), simulated_response.bool()


class AcquisitionFunctionDecisionMaker(MCAcquisitionFunction):
    """
    Acquisition function for the decision maker
    """

    def __init__(self, model: Model,
                 weights: Tensor,
                 y_train: Tensor,
                 utility_model: Callable,
                 bounds: Tensor,
                 optional: dict,
                 objective: Optional[MCAcquisitionObjective] = None,
                 sampler: Optional[MCSampler] = None,
                 num_fantasies: Optional[int] = 64,
                 current_optimiser: Optional[Tensor] = None, ):

        if sampler is None:
            if num_fantasies is None:
                raise ValueError(
                    "Must specify `num_fantasies` if no `sampler` is provided."
                )
            # base samples should be fixed for joint optimization over X, X_fantasies
            sampler = SobolQMCNormalSampler(
                num_samples=num_fantasies, resample=False, collapse_batch_dims=True
            )
        elif num_fantasies is not None:
            if sampler.sample_shape != torch.Size([num_fantasies]):
                raise ValueError(
                    f"The sampler shape must match num_fantasies={num_fantasies}."
                )
        else:
            num_fantasies = sampler.sample_shape[0]
        super(MCAcquisitionFunction, self).__init__(model=model)
        # if not explicitly specified, we use the posterior mean for linear objs

        self.sampler = sampler
        self.objective = objective
        self.num_fantasies = num_fantasies
        self.current_optimiser = current_optimiser
        self.weights = weights
        self.y_train = y_train
        self.cache_voi = []
        self.utility_model = utility_model
        self.bounds = bounds
        self.optional = optional

    def forward(self, pairs: Tensor) -> Tensor:
        KG = torch.zeros(len(pairs))
        for idx, p in enumerate(pairs):
            option_1 = p[0]
            option_2 = p[1]
            KG[idx] = self._compute_voi_pair(option_1=option_1,
                                             option_2=option_2)

        return KG

    def find_best_pair(self, pairs: list, pairs_idx: list) -> Tuple[Tensor, Tensor, Tensor]:

        KG = torch.zeros(len(pairs))
        for idx, p in enumerate(pairs):
            option_1 = p[0]
            option_2 = p[1]

            KG[idx] = self._compute_voi_pair(option_1=option_1,
                                             option_2=option_2)

        best_idx = torch.argmax(KG)
        best_pair = pairs[best_idx]
        best_pair_idx = pairs_idx[best_idx]
        voi_dm = KG[best_idx]

        return best_pair_idx, best_pair, voi_dm

    def _compute_voi_pair(self, option_1: Tensor,
                          option_2: Tensor) -> Tensor:

        acceptance_prob, weight_mask = compute_acceptance_probability(utility_model=self.utility_model,
                                                                      option_1=option_1,
                                                                      option_2=option_2,
                                                                      weights=self.weights,
                                                                      y_train=self.y_train)

        if acceptance_prob == 1 or acceptance_prob == 0:
            voi_value = 0

        else:
            accepted_weights_option_1 = self.weights[weight_mask, :]

            voi_value_option_1 = self.optimise_posterior_mean(weights=accepted_weights_option_1)

            if self.current_optimiser is not None:
                current_best_value_option_1 = self.evaluate_expected_posterior(weights=accepted_weights_option_1,
                                                                               X=self.current_optimiser)
                voi_value_option_1 = voi_value_option_1 - current_best_value_option_1

            accepted_weights_option_2 = self.weights[torch.logical_not(weight_mask), :]
            voi_value_option_2 = self.optimise_posterior_mean(weights=accepted_weights_option_2)

            if self.current_optimiser is not None:
                current_best_value_option_2 = self.evaluate_expected_posterior(weights=accepted_weights_option_2,
                                                                               X=self.current_optimiser)
                voi_value_option_2 = voi_value_option_2 - current_best_value_option_2

            voi_value = acceptance_prob * voi_value_option_1 + (1 - acceptance_prob) * voi_value_option_2

        return voi_value

    def evaluate_expected_posterior(self, X: Tensor, weights: Tensor):

        integrated_objective = integrated_utility_objective(utility_model=self.utility_model,
                                                            weights=weights,
                                                            Y_sampled=self.y_train)

        Expected_Utility = GenericMCObjective(integrated_objective)

        expected_posterior_mean_objective = ExpectedPosteriorMean(model=self.model,
                                                                  objective=Expected_Utility)
        with torch.no_grad():
            x_train_posterior_mean = expected_posterior_mean_objective.forward(X).squeeze()
        return x_train_posterior_mean

    def optimise_posterior_mean(self, weights: Tensor):

        bounds_normalized = self.bounds
        dim = bounds_normalized.shape[-1]
        # generate initialisation points
        X_random_initial_conditions_raw = torch.rand((self.optional["RAW_SAMPLES"], dim))
        X_sampled = self.model.train_inputs[0][0].squeeze()


        X_initial_conditions_raw = torch.concat([X_random_initial_conditions_raw, X_sampled])
        X_initial_conditions_raw = X_initial_conditions_raw.unsqueeze(dim=-2)

        integrated_objective = integrated_utility_objective(utility_model=self.utility_model,
                                                            weights=weights,
                                                            Y_sampled=self.y_train)

        Expected_Utility = GenericMCObjective(integrated_objective)

        expected_posterior_mean_objective = ExpectedPosteriorMean(model=self.model,
                                                                  objective=Expected_Utility)
        with torch.no_grad():
            x_train_posterior_mean = expected_posterior_mean_objective.forward(X_initial_conditions_raw).squeeze()

        best_k_indeces = torch.argsort(x_train_posterior_mean, descending=True)[:self.optional["NUM_RESTARTS"]]
        X_initial_conditions = X_initial_conditions_raw[best_k_indeces, :]

        X_optimised, X_optimised_vals = optimize_acqf(
            acq_function=expected_posterior_mean_objective,
            bounds=bounds_normalized,
            batch_initial_conditions=X_initial_conditions,
            q=1,
            num_restarts=self.optional["NUM_RESTARTS"],
            raw_samples=self.optional["RAW_SAMPLES"],
        )

        x_best_val = torch.max(X_optimised_vals.squeeze())
        return x_best_val


def ValueOfInformationDecisionMaker(model: Model,
                                    utility_model: Callable,
                                    weights: Tensor,
                                    bounds: Tensor,
                                    Y_sampled: Tensor,
                                    x_best: Tensor,
                                    optional: dict):
    "Function wrapper to compute acquisition function for decision maker with integrated utility"

    acquisitionfun_decision_maker = AcquisitionFunctionDecisionMaker(model=model,
                                                                     utility_model=utility_model,
                                                                     bounds=bounds,
                                                                     weights=weights,
                                                                     y_train=Y_sampled,
                                                                     current_optimiser=x_best,
                                                                     optional=optional)

    return acquisitionfun_decision_maker
