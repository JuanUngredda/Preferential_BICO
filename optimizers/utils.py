import time
from typing import Optional, Callable, Dict, Tuple

import torch
from torch import Tensor

from botorch.acquisition.analytic import AnalyticAcquisitionFunction, _construct_dist
from optimizers.acquisition_functions.VoI_simulator import ValueOfInformationSimulator
from optimizers.acquisition_functions.VoI_preferences import ValueOfInformationDecisionMaker
from botorch.generation.gen import gen_candidates_scipy
from botorch.models.model import Model
from botorch.utils import standardize
from botorch.utils.transforms import t_batch_mode_transform

from botorch.utils.multi_objective.scalarization import (
    get_chebyshev_scalarization
)

dtype = torch.double


#################################################################
#                                                               #
#                           LHC SAMPLERS                        #
#                                                               #
#################################################################


def lhc(
        n: int,
        dim: Optional[int] = None,
        lb: Optional[Tensor] = None,
        ub: Optional[Tensor] = None,
) -> Tensor:
    """
    Parameters
    ----------
    n: sample size
    dim: optional, dimenions of the cube
    lb: lower bound, Tensor
    ub: upper bound, Tensor
    Returns
    -------
    x: Tensor, shape (n, dim)
    """

    if dim is not None:
        assert (lb is None) and (ub is None), "give dim OR bounds"
        lb = torch.zeros(dim)
        ub = torch.ones(dim)

    else:
        assert (lb is not None) and (ub is not None), "give dim OR bounds"
        lb = lb.squeeze()
        ub = ub.squeeze()
        dim = len(lb)
        assert len(lb) == len(ub), f"bounds are not same shape:{len(lb)}!={len(ub)}"

    x = torch.zeros((n, dim))
    if n > 0:
        for d in range(dim):
            x[:, d] = (torch.randperm(n) + torch.rand(n)) * (1 / n)
            x[:, d] = (ub[d] - lb[d]) * x[:, d] + lb[d]

    return x


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            if name in kw["log_time"].keys():
                kw["log_time"][name].append((te - ts))
            else:
                kw["log_time"][name] = [te - ts]
        return result

    return timed


def test_function_handler(test_fun_str: str,
                          test_fun_dict: dict,
                          input_dim: int,
                          output_dim: int):
    if test_fun_str == "C2DTLZ2":
        synthetic_fun = test_fun_dict[test_fun_str](dim=input_dim,
                                                    num_objectives=output_dim,
                                                    negate=True)
    else:

        synthetic_fun = test_fun_dict[test_fun_str](negate=True)

    return synthetic_fun


def mo_acq_wrapper(
        method: str,
        utility_model_name=str,
        num_fantasies: Optional[int] = None
):
    def acquisition_function(model: method,
                             weights: Tensor,
                             Y_sampled: Tensor,
                             utility_model: Callable,
                             current_best_value: Tensor):

        voi_sim = ValueOfInformationSimulator(
            model=model,
            utility_model=utility_model,
            num_fantasies=num_fantasies,
            weights=weights,
            Y_sampled=Y_sampled,
            x_best=current_best_value)
        return voi_sim

    def recommender_function(model: method,
                             weights: Tensor,
                             utility_model: Callable,
                             bounds: Tensor,
                             Y_sampled: Tensor,
                             current_best_value: Tensor,
                             optional: dict):

        voi_dm = ValueOfInformationDecisionMaker(model=model,
                                                 utility_model=utility_model,
                                                 bounds=bounds,
                                                 weights=weights,
                                                 Y_sampled=Y_sampled,
                                                 x_best=current_best_value,
                                                 optional=optional)
        return voi_dm

    if method == "Interactive":

        return acquisition_function, recommender_function

    elif method == "VoISim":
        return acquisition_function, None

    else:
        print("Method Not Implement")
        raise


def _compute_expected_utility(
        scalatization_fun: Callable,
        y_values: Tensor,
        c_values: Tensor,
        weights: Tensor,
) -> Tensor:
    utility = torch.zeros((weights.shape[0], y_values.shape[0]))

    for idx, w in enumerate(weights):
        scalarization = scalatization_fun(weights=w, Y=torch.Tensor([]).view((0, y_values.shape[1])))
        utility_values = scalarization(y_values).squeeze()
        utility[idx, :] = utility_values

    print(weights.shape)
    print(y_values.shape)
    print(utility.shape)
    raise
    return expected_utility
