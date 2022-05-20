#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Batch Knowledge Gradient (KG) via one-shot optimization as introduced in
[Balandat2020botorch]_. For broader discussion of KG see also [Frazier2008knowledge]_
and [Wu2016parallelkg]_.

.. [Balandat2020botorch]
    M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, and
    E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization.
    Advances in Neural Information Processing Systems 33, 2020.

.. [Frazier2008knowledge]
    P. Frazier, W. Powell, and S. Dayanik. A Knowledge-Gradient policy for
    sequential information collection. SIAM Journal on Control and Optimization,
    2008.

.. [Wu2016parallelkg]
    J. Wu and P. Frazier. The parallel knowledge gradient method for batch
    bayesian optimization. NIPS 2016.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple, Type

import torch
from botorch import settings
from botorch.acquisition.acquisition import (
    AcquisitionFunction,
    OneShotAcquisitionFunction,
)
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.cost_aware import CostAwareUtility
from botorch.acquisition.monte_carlo import MCAcquisitionFunction, qSimpleRegret
from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from botorch.utils.transforms import (
    concatenate_pending_points,
    match_batch_shape,
    t_batch_mode_transform,
)
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from torch import Tensor

def _get_best_xstar(model:Model,
             sampler: Callable,
             objective: Callable,
             posterior_transform: Callable,
             inner_sampler: Callable,
             X: Tensor, bounds: Tensor,
             x_current_best:Optional[Tensor]=None,
             **kwargs: Any) -> Tensor:
    r"""Evaluate qKnowledgeGradient on the candidate set `X_actual` by
    solving the inner optimization problem.

    Args:
        X: A `b x q x d` Tensor with `b` t-batches of `q` design points
            each. Unlike `forward()`, this does not include solutions of the
            inner optimization problem.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of
            the solutions to the inner problem.
        kwargs: Additional keyword arguments. This includes the options for
            optimization of the inner problem, i.e. `num_restarts`, `raw_samples`,
            an `options` dictionary to be passed on to the optimization helpers, and
            a `scipy_options` dictionary to be passed to `scipy.minimize`.

    Returns:
        A Tensor of shape `b`. For t-batch b, the q-KG value of the design
            `X[b]` is averaged across the fantasy models.
            NOTE: If `current_value` is not provided, then this is not the
            true KG value of `X[b]`.
    """

    # construct the fantasy model of shape `num_fantasies x b`
    fantasy_model = model.fantasize(
        X=X, sampler=sampler, observation_noise=True
    )

    # get the value function
    value_function = _get_value_function(
        model=fantasy_model,
        objective=objective,
        posterior_transform=posterior_transform,
        sampler=inner_sampler,
        project=None,
    )

    from botorch.generation.gen import gen_candidates_scipy

    # optimize the inner problem
    from botorch.optim.initializers import gen_value_function_initial_conditions

    initial_conditions = gen_value_function_initial_conditions(
        acq_function=value_function,
        bounds=bounds,
        num_restarts=1,
        raw_samples=kwargs.get("raw_samples", 1024),
        current_model=model,
    )

    maximisers, values = gen_candidates_scipy(
        initial_conditions=initial_conditions,
        acquisition_function=value_function,
        lower_bounds=bounds[0],
        upper_bounds=bounds[1],
        options=kwargs.get("scipy_options"),
    )
    maximisers = maximisers.squeeze()
    X = X

    recommended_points = torch.vstack([X, maximisers]).unsqueeze(dim=0)

    return recommended_points


def evaluate(model:Model,
             sampler: Callable,
             objective: Callable,
             posterior_transform: Callable,
             inner_sampler: Callable,
             X: Tensor, bounds: Tensor,
             x_current_best:Optional[Tensor]=None,
             **kwargs: Any) -> Tensor:
    r"""Evaluate qKnowledgeGradient on the candidate set `X_actual` by
    solving the inner optimization problem.

    Args:
        X: A `b x q x d` Tensor with `b` t-batches of `q` design points
            each. Unlike `forward()`, this does not include solutions of the
            inner optimization problem.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of
            the solutions to the inner problem.
        kwargs: Additional keyword arguments. This includes the options for
            optimization of the inner problem, i.e. `num_restarts`, `raw_samples`,
            an `options` dictionary to be passed on to the optimization helpers, and
            a `scipy_options` dictionary to be passed to `scipy.minimize`.

    Returns:
        A Tensor of shape `b`. For t-batch b, the q-KG value of the design
            `X[b]` is averaged across the fantasy models.
            NOTE: If `current_value` is not provided, then this is not the
            true KG value of `X[b]`.
    """

    # construct the fantasy model of shape `num_fantasies x b`
    fantasy_model = model.fantasize(
        X=X, sampler=sampler, observation_noise=True
    )

    # get the value function
    value_function = _get_value_function(
        model=fantasy_model,
        objective=objective,
        posterior_transform=posterior_transform,
        sampler=inner_sampler,
        project=None,
    )

    from botorch.generation.gen import gen_candidates_scipy

    # optimize the inner problem
    from botorch.optim.initializers import gen_value_function_initial_conditions

    initial_conditions = gen_value_function_initial_conditions(
        acq_function=value_function,
        bounds=bounds,
        num_restarts=kwargs.get("num_restarts", 20),
        raw_samples=kwargs.get("raw_samples", 1024),
        current_model=model,
    )

    _, values = gen_candidates_scipy(
        initial_conditions=initial_conditions,
        acquisition_function=value_function,
        lower_bounds=bounds[0],
        upper_bounds=bounds[1],
        options=kwargs.get("scipy_options"),
    )
    # get the maximizer for each batch
    values, _ = torch.max(values, dim=0)

    if x_current_best is not None:

        values = values - value_function(x_current_best)

    # return average over the fantasy samples
    return values.mean(dim=0)

class ProjectedAcquisitionFunction(AcquisitionFunction):
    r"""
    Defines a wrapper around  an `AcquisitionFunction` that incorporates the project
    operator. Typically used to handle value functions in look-ahead methods.
    """

    def __init__(
        self,
        base_value_function: AcquisitionFunction,
        project: Callable[[Tensor], Tensor],
    ) -> None:
        super().__init__(base_value_function.model)
        self.base_value_function = base_value_function
        self.project = project
        self.objective = getattr(base_value_function, "objective", None)
        self.posterior_transform = base_value_function.posterior_transform
        self.sampler = getattr(base_value_function, "sampler", None)

    def forward(self, X: Tensor) -> Tensor:
        return self.base_value_function(self.project(X))


def _get_value_function(
    model: Model,
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    sampler: Optional[MCSampler] = None,
    project: Optional[Callable[[Tensor], Tensor]] = None,
    valfunc_cls: Optional[Type[AcquisitionFunction]] = None,
    valfunc_argfac: Optional[Callable[[Model, Dict[str, Any]]]] = None,
) -> AcquisitionFunction:
    r"""Construct value function (i.e. inner acquisition function)."""
    if valfunc_cls is not None:
        common_kwargs: Dict[str, Any] = {
            "model": model,
            "posterior_transform": posterior_transform,
        }
        if issubclass(valfunc_cls, MCAcquisitionFunction):
            common_kwargs["sampler"] = sampler
            common_kwargs["objective"] = objective
        kwargs = valfunc_argfac(model=model) if valfunc_argfac is not None else {}
        base_value_function = valfunc_cls(**common_kwargs, **kwargs)
    else:
        if objective is not None:
            base_value_function = qSimpleRegret(
                model=model,
                sampler=sampler,
                objective=objective,
                posterior_transform=posterior_transform,
            )
        else:
            base_value_function = PosteriorMean(
                model=model, posterior_transform=posterior_transform
            )

    if project is None:
        return base_value_function
    else:
        return ProjectedAcquisitionFunction(
            base_value_function=base_value_function,
            project=project,
        )


def _split_fantasy_points(X: Tensor, n_f: int) -> Tuple[Tensor, Tensor]:
    r"""Split a one-shot optimization input into actual and fantasy points

    Args:
        X: A `batch_shape x (q + n_f) x d`-dim tensor of actual and fantasy
            points

    Returns:
        2-element tuple containing

        - A `batch_shape x q x d`-dim tensor `X_actual` of input candidates.
        - A `n_f x batch_shape x 1 x d`-dim tensor `X_fantasies` of fantasy
            points, where `X_fantasies[i, batch_idx]` is the i-th fantasy point
            associated with the batch indexed by `batch_idx`.
    """
    if n_f > X.size(-2):
        raise ValueError(
            f"n_f ({n_f}) must be less than the q-batch dimension of X ({X.size(-2)})"
        )
    split_sizes = [X.size(-2) - n_f, n_f]
    X_actual, X_fantasies = torch.split(X, split_sizes, dim=-2)
    # X_fantasies is b x num_fantasies x d, needs to be num_fantasies x b x 1 x d
    # for batch mode evaluation with batch shape num_fantasies x b.
    # b x num_fantasies x d --> num_fantasies x b x d
    X_fantasies = X_fantasies.permute(-2, *range(X_fantasies.dim() - 2), -1)
    # num_fantasies x b x 1 x d
    X_fantasies = X_fantasies.unsqueeze(dim=-2)
    return X_actual, X_fantasies

