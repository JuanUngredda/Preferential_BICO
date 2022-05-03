# Acquisition Function for simulator

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

def integrated_utility_objective(utility_model: Callable, weights: Tensor, Y_sampled):
    """Initialize a ExpectedUtilityMCObjective for VoI simulator"""
    weights = torch.atleast_2d(weights)
    utility_lst = []
    for w in weights:
        utility_lst.append(utility_model(weights=w, Y=Y_sampled))

    def expected_utility(Z):

        expected_utility_vals = utility_lst[0](Z)
        for uvals in utility_lst[1:]:
            expected_utility_vals += uvals(Z)

        expected_utility_vals *= len(utility_lst)
        return expected_utility_vals

    return expected_utility


def ValueOfInformationSimulator(model: Model,
                                utility_model: Callable,
                                weights: Tensor,
                                num_fantasies: int,
                                Y_sampled: Tensor):
    "Function wrapper to compute one-shot KG with integrated utility"
    # integrated utility objective
    integrated_objective = integrated_utility_objective(utility_model=utility_model,
                                                        weights=weights,
                                                        Y_sampled=Y_sampled)

    Expected_Utility = GenericMCObjective(integrated_objective)

    batched_model = (
        model_list_to_batched(model) if isinstance(model, ModelListGP) else model
    )

    # one-shot kg. Include current best integrated value
    qKG = qKnowledgeGradient(batched_model,
                             objective=Expected_Utility,
                             num_fantasies=num_fantasies,
                             current_value=None)

    return qKG
