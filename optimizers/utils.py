import time
from typing import Optional, Callable, Dict, Tuple

import torch
from torch import Tensor

from botorch.acquisition.analytic import AnalyticAcquisitionFunction, _construct_dist
from botorch.acquisition.multi_objective.multi_attribute_constrained_kg import MultiAttributeConstrainedKG
from botorch.generation.gen import gen_candidates_scipy
from botorch.models.model import Model
from botorch.utils import standardize
from botorch.utils.transforms import t_batch_mode_transform

from botorch.utils.multi_objective.scalarization import (
    get_chebyshev_scalarization,
    get_linear_scalarization,
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


def mo_acq_wrapper(
        method: str,
        num_objectives: int,
        utility_model_name=str,
        bounds: Optional[Tensor] = None,
        num_fantasies: Optional[int] = None,
        num_scalarizations: Optional[int] = None,
        num_discrete_points: Optional[int] = None,
        num_restarts: Optional[int] = None,
        raw_samples: Optional[int] = None,
):
    if utility_model_name == "Tche":
        utility_model = get_chebyshev_scalarization

    elif utility_model_name == "Lin":
        utility_model = get_linear_scalarization

    def acquisition_function(model: method):
        if method == "macKG":
            KG_acq_fun = MultiAttributeConstrainedKG(
                model=model,
                bounds=bounds,
                utility_model= utility_model,
                num_objectives= num_objectives,
                num_fantasies=num_fantasies,
                num_scalarisations=num_scalarizations)

        elif method == "SMSEGO":
            pass

        else:
            raise Exception(
                "method does not exist. Specify implemented method: DISCKG (Discrete KG), "
                "MCKG (Monte Carlo KG), HYBRIDKG (Hybrid KG), and ONESHOTKG (One Shot KG)"
            )
        return KG_acq_fun

    return acquisition_function


class ConstrainedPosteriorMean(AnalyticAcquisitionFunction):
    r"""Constrained Posterior Mean (feasibility-weighted).

    Computes the analytic Posterior Mean for a Normal posterior
    distribution, weighted by a probability of feasibility. The objective and
    constraints are assumed to be independent and have Gaussian posterior
    distributions. Only supports the case `q=1`. The model should be
    multi-outcome, with the index of the objective and constraints passed to
    the constructor.
    """

    def __init__(
            self,
            model: Model,
            objective_index: int,
            maximize: bool = True,
            scalarization=Callable,
    ) -> None:
        r"""Analytic Constrained Expected Improvement.

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best feasible function value observed so far (assumed noiseless).
            objective_index: The index of the objective.
            constraints: A dictionary of the form `{i: [lower, upper]}`, where
                `i` is the output index, and `lower` and `upper` are lower and upper
                bounds on that output (resp. interpreted as -Inf / Inf if None)
            maximize: If True, consider the problem a maximization problem.
        """
        # use AcquisitionFunction constructor to avoid check for objective
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.objective = None
        self.maximize = maximize
        self.objective_index = objective_index
        self.scalarization = scalarization
        default_value = (None, 0)
        constraints = dict.fromkeys(
            range(model.num_outputs - self.objective_index), default_value
        )
        self._preprocess_constraint_bounds(constraints=constraints)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Constrained Expected Improvement on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Expected Improvement values at the given
            design points `X`.
        """
        X = X.to(dtype=torch.double)
        posterior = self.model.posterior(X=X)
        means = posterior.mean.squeeze(dim=-2)  # (b) x m
        sigmas = posterior.variance.squeeze(dim=-2).sqrt().clamp_min(1e-9)  # (b) x m

        # (b) x 1
        oi = self.objective_index
        mean_obj = means[..., :oi]
        scalarized_objective = self.scalarization(Y=mean_obj)

        mean_constraints = means[..., oi:]
        sigma_constraints = sigmas[..., oi:]

        prob_feas = self._compute_prob_feas(
            X=X.squeeze(dim=-2),
            means=mean_constraints.squeeze(dim=-2),
            sigmas=sigma_constraints.squeeze(dim=-2),
        ).double()

        constrained_posterior_mean = scalarized_objective.squeeze() * prob_feas.squeeze()

        # import matplotlib.pyplot as plt
        # val =constrained_posterior_mean.squeeze(dim=-1).detach().numpy()
        # plt.scatter(mean_obj.detach().numpy()[..., 0], mean_obj.detach().numpy()[..., 1], c=val)
        # plt.show()
        # raise

        return constrained_posterior_mean.squeeze(dim=-1).double()

    def _preprocess_constraint_bounds(
            self, constraints: Dict[int, Tuple[Optional[float], Optional[float]]]
    ) -> None:
        r"""Set up constraint bounds.

        Args:
            constraints: A dictionary of the form `{i: [lower, upper]}`, where
                `i` is the output index, and `lower` and `upper` are lower and upper
                bounds on that output (resp. interpreted as -Inf / Inf if None)
        """
        con_lower, con_lower_inds = [], []
        con_upper, con_upper_inds = [], []
        con_both, con_both_inds = [], []
        con_indices = list(constraints.keys())
        if len(con_indices) == 0:
            raise ValueError("There must be at least one constraint.")
        if self.objective_index in con_indices:
            raise ValueError(
                "Output corresponding to objective should not be a constraint."
            )
        for k in con_indices:
            if constraints[k][0] is not None and constraints[k][1] is not None:
                if constraints[k][1] <= constraints[k][0]:
                    raise ValueError("Upper bound is less than the lower bound.")
                con_both_inds.append(k)
                con_both.append([constraints[k][0], constraints[k][1]])
            elif constraints[k][0] is not None:
                con_lower_inds.append(k)
                con_lower.append(constraints[k][0])
            elif constraints[k][1] is not None:
                con_upper_inds.append(k)
                con_upper.append(constraints[k][1])
        # tensor-based indexing is much faster than list-based advanced indexing
        self.register_buffer("con_lower_inds", torch.tensor(con_lower_inds))
        self.register_buffer("con_upper_inds", torch.tensor(con_upper_inds))
        self.register_buffer("con_both_inds", torch.tensor(con_both_inds))
        # tensor indexing
        self.register_buffer("con_both", torch.tensor(con_both, dtype=torch.float))
        self.register_buffer("con_lower", torch.tensor(con_lower, dtype=torch.float))
        self.register_buffer("con_upper", torch.tensor(con_upper, dtype=torch.float))

    def _compute_prob_feas(self, X: Tensor, means: Tensor, sigmas: Tensor) -> Tensor:
        r"""Compute feasibility probability for each batch of X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.
            means: A `(b) x m`-dim Tensor of means.
            sigmas: A `(b) x m`-dim Tensor of standard deviations.
        Returns:
            A `(b) x 1`-dim tensor of feasibility probabilities

        Note: This function does case-work for upper bound, lower bound, and both-sided
        bounds. Another way to do it would be to use 'inf' and -'inf' for the
        one-sided bounds and use the logic for the both-sided case. But this
        causes an issue with autograd since we get 0 * inf.
        TODO: Investigate further.
        """
        output_shape = X.shape[:-2] + torch.Size([1])
        prob_feas = torch.ones(output_shape, device=X.device, dtype=X.dtype)

        if len(self.con_lower_inds) > 0:
            self.con_lower_inds = self.con_lower_inds.to(device=X.device)
            normal_lower = _construct_dist(means, sigmas, self.con_lower_inds)
            prob_l = 1 - normal_lower.cdf(self.con_lower)
            prob_feas = prob_feas.mul(torch.prod(prob_l, dim=-1, keepdim=True))
        if len(self.con_upper_inds) > 0:
            self.con_upper_inds = self.con_upper_inds.to(device=X.device)
            normal_upper = _construct_dist(means, sigmas, self.con_upper_inds)
            prob_u = normal_upper.cdf(self.con_upper)
            prob_feas = prob_feas.mul(torch.prod(prob_u, dim=-1, keepdim=True))
        if len(self.con_both_inds) > 0:
            self.con_both_inds = self.con_both_inds.to(device=X.device)
            normal_both = _construct_dist(means, sigmas, self.con_both_inds)
            prob_u = normal_both.cdf(self.con_both[:, 1])
            prob_l = normal_both.cdf(self.con_both[:, 0])
            prob_feas = prob_feas.mul(torch.prod(prob_u - prob_l, dim=-1, keepdim=True))
        return prob_feas


def ParetoFrontApproximation(
        model: Model,
        input_dim: int,
        objective_dim: int,
        scalatization_fun: Callable,
        bounds: Tensor,
        y_train: Tensor,
        weights: Tensor,
        optional: Optional[dict[str, int]] = None,
) -> tuple[Tensor, Tensor]:

    X_pareto_solutions = []
    X_pmean = []

    dummy_X = torch.rand((500, input_dim))
    posterior = model.posterior(dummy_X)
    dummy_mean = posterior.mean[..., :objective_dim]

    for w in weights:
        # normalizes scalarization between [0,1] given the training data.
        scalarization = scalatization_fun(weights=w, Y=dummy_mean)

        constrained_model = ConstrainedPosteriorMean(
            model=model,
            objective_index=y_train.shape[-1],
            scalarization=scalarization,
        )

        X_initial_conditions_raw = torch.rand((optional["RAW_SAMPLES"], 1, 1, input_dim))

        mu_val_initial_conditions_raw = constrained_model.forward(
            X_initial_conditions_raw
        )

        best_k_indeces = torch.argsort(mu_val_initial_conditions_raw, descending=True)[
                         : optional["NUM_RESTARTS"]
                         ]
        X_initial_conditions = X_initial_conditions_raw[best_k_indeces, :].double()

        top_x_initial_means, value_initial_means = gen_candidates_scipy(
            initial_conditions=X_initial_conditions,
            acquisition_function=constrained_model,
            lower_bounds=torch.zeros(input_dim),
            upper_bounds=torch.ones(input_dim))

        top_x = top_x_initial_means[torch.argmax(value_initial_means), ...]
        X_pareto_solutions.append(top_x)
        X_pmean.append(torch.max(value_initial_means))

    X_pareto_solutions = torch.vstack(X_pareto_solutions)
    X_pmean = torch.vstack(X_pmean)

    # plot_X = torch.rand((1000,3))
    # posterior = model.posterior(plot_X)
    # mean = posterior.mean.detach().numpy()
    # is_feas = (mean[:,2] <= 0)
    # print("mean", mean.shape)
    # import matplotlib.pyplot as plt
    # plt.scatter(mean[is_feas,0], mean[is_feas,1], c=mean[is_feas,2])
    #
    # Y_pareto_posterior = model.posterior(X_pareto_solutions)
    # Y_pareto_mean = Y_pareto_posterior.mean.detach().numpy()
    # print(Y_pareto_mean.shape)
    # plt.scatter(Y_pareto_mean[...,0], Y_pareto_mean[...,1], color="red")
    #
    # plt.show()
    # raise

    return X_pareto_solutions, X_pmean


def _compute_expected_utility(
        scalatization_fun: Callable,
        y_values: Tensor,
        c_values: Tensor,
        weights: Tensor,
) -> Tensor:

    utility = torch.zeros((weights.shape[0], y_values.shape[0]))

    for idx, w in enumerate(weights):
        scalarization = scalatization_fun(weights=w, Y=torch.Tensor([]).view((0,y_values.shape[1])))
        utility_values = scalarization(y_values).squeeze()
        utility[idx, :] = utility_values

    is_feas =  (c_values <= 0).squeeze()

    if is_feas.sum() == 0:
        expected_utility = torch.Tensor([-100])
        return expected_utility
    else:
        utility_feas = utility[:, is_feas]
        best_utility = torch.max(utility_feas , dim=1).values
        expected_utility = best_utility.mean()

        return expected_utility
