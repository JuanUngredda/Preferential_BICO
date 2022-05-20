# Acquisition Function for simulator

from typing import Dict, Tuple, Any, Callable, Optional
import torch
from torch import Tensor
from botorch.models.model import Model
# from .Knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.objective import GenericMCObjective
from botorch.models.converter import (
    model_list_to_batched,
)
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.acquisition.objective import PosteriorTransform
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler

class ExpectedPosteriorMean(MCAcquisitionFunction):
    r"""Single-outcome Posterior Mean.

    Only supports the case of q=1. Requires the model's posterior to have a
    `mean` property. The model must be single-outcome.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> PM = PosteriorMean(model)
        >>> pm = PM(test_X)
    """

    def __init__(
        self,
        model: Model,
        objective: Optional[MCAcquisitionObjective] = None,
        sampler: Optional[MCSampler] = None
    ) -> None:
        r"""Single-outcome Posterior Mean.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem. Note
                that if `maximize=False`, the posterior mean is negated. As a
                consequence `optimize_acqf(PosteriorMean(gp, maximize=False))`
                does actually return -1 * minimum of the posterior mean.
        """
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective
        )



    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qExpectedImprovement on the candidate set `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of Expected Improvement values at the given
            design points `X`, where `batch_shape'` is the broadcasted batch shape of
            model and input `X`.
        """

        posterior = self.model.posterior(X=X)
        posterior_mean = posterior.mean.unsqueeze(dim=0)
        obj = self.objective(posterior_mean, X=X)
        return obj.mean(dim=0).squeeze(dim=1)


def integrated_utility_objective(utility_model: Callable,
                                 weights: Tensor,
                                 Y_sampled: Tensor):
    """Initialize a ExpectedUtilityMCObjective for VoI simulator"""
    weights = torch.atleast_2d(weights)

    # from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
    # utility_model_unvectorised = get_chebyshev_scalarization
    # utility_lst = []
    # for w in weights:
    #     ufun = utility_model_unvectorised(weights=w, Y=Y_sampled)
    #     print("uval_unvect", ufun(Y_sampled))
    #     utility_lst.append(ufun)

    ufun = utility_model(weights=weights, Y=Y_sampled)
    # uval = ufun(Y_sampled) # (num_Y_values, num_parameters)


    # print("uval", uval)
    def expected_utility(Z):
        # expected_utility_vals = utility_lst[0](Z)
        #
        # test_lst = []
        # for uvals in utility_lst[1:]:
        #     test_lst.append(uvals(Z))
        #     expected_utility_vals += uvals(Z)
        #
        # expected_utility_vals *= 1/len(utility_lst)
        expected_utility_vectorised = ufun(Z)
        # print("expected_utility_vals", test_lst)
        # print("expected_utility", expected_utility_vals, expected_utility_vals.shape)
        # print("expected_utility_vectorised",expected_utility_vectorised.mean(dim=-1), expected_utility_vectorised.mean(dim=-1).shape)
        # raise
        return expected_utility_vectorised.mean(dim=-1).unsqueeze(dim=-1)

    return expected_utility


def ValueOfInformationSimulator(model: Model,
                                utility_model: Callable,
                                weights: Tensor,
                                num_fantasies: int,
                                Y_sampled: Tensor,
                                x_best: Tensor):
    "Function wrapper to compute one-shot KG with integrated utility"
    # integrated utility objective
    integrated_objective = integrated_utility_objective(utility_model=utility_model,
                                                        weights=weights,
                                                        Y_sampled=Y_sampled)

    Expected_Utility = GenericMCObjective(integrated_objective)

    batched_model = (
        model_list_to_batched(model) if isinstance(model, ModelListGP) else model
    )

    expected_posterior_mean_objective = ExpectedPosteriorMean(model=model,
                                                              objective=Expected_Utility)

    best_expected_posterior_value = expected_posterior_mean_objective(x_best)
    # one-shot kg. Include current best integrated value
    inner_sampler = SobolQMCNormalSampler(
        num_samples=50, resample=False, collapse_batch_dims=True
    )
    qKG = qKnowledgeGradient(batched_model,
                             inner_sampler= inner_sampler,
                             objective=Expected_Utility ,
                             num_fantasies=num_fantasies,
                             current_value=best_expected_posterior_value)


    return qKG
