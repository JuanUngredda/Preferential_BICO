import logging
import sys
from abc import ABC, abstractmethod

import torch
from torch import Tensor

from .utils import lhc
from botorch.utils.sampling import sample_simplex

LOG_FORMAT = (
    "%(asctime)s - %(name)s:%(funcName)s:%(lineno)s - %(levelname)s:  %(message)s"
)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


###################################################################
##                                                               ##
##                         OPTIMIZERS                            ##
##                                                               ##
###################################################################

# All optimizers must have the following components:
#   - def __init__(test_fun, budget, *kwargs)
#   - def optimize(): tries to learn optima
#   - def get_next_point(): find (s,x) to evaluate for next iteration
#   - def policy(): returns recommended action
#   - def test(): evaluates policy() action averaged over many seeds.


class BaseOptimizer(ABC):
    """
    Randomly picks a new point to sample at each time step
    and at test time, when given a test state, it takes the nearest 10%
    of sample states, finds the state with the best y value and returns x.
    """

    def __init__(
            self,
            fun,
            lb: Tensor,
            ub: Tensor,
            n_max: int,
            n_init: int = 20,
            n_pairs: int=0,
            ns0: int = None,
    ):
        """
        ARGS:
            fun: expensive black box function: X x {1,2,3,..} -> R
            lb: np.ndarray, lower bounds on x
            ub: np.ndarray, upper bounds on x
            n_max: int, number of call to tet funtion
            n_init: int, number of samples to start the BO
            ns0: int, number of seeds to sample, default n_init
        RETURNS:
            Optimizer object
        """
        logger.info(f"Initializing {type(self)}")

        if ns0 is None:
            self.ns0 = n_init
        else:
            self.ns0 = ns0
        self.n_init = n_init
        self.n_pairs = n_pairs
        self.n_max = n_max
        self.f = fun
        self.lb = lb.squeeze(-1)
        self.ub = ub.squeeze(-1)
        self.dim = len(lb.squeeze())
        self.GP_performance = torch.zeros((0, 2))
        self.sampled_performance = torch.zeros((0, 2))
        self.method_time = {}
        self.gp_likelihood_noise = torch.Tensor([])
        self.gp_lengthscales = torch.Tensor([])
        # no need to test every step, 30 points will be enough for a results plot.
        self.testable_iters = torch.unique(
            torch.linspace(n_init, n_max, steps=31, dtype=int)
        )
        logger.info("Testable iters: %s", self.testable_iters)

    def optimize(self):

        logger.info(f"Starting optim, n_init: {self.n_init}")

        # initial random dataset
        self.y_train_option_1 = torch.zeros((0, self.dim))
        self.y_train_option_2 = torch.zeros((0, self.dim))
        self.index_pairs_sampled = []
        self.decisions = []

        self.x_train = lhc(n=self.n_init, dim=self.dim).to(dtype=torch.double)
        self.y_train = torch.vstack(
            [self.evaluate_objective(x_i) for x_i in self.x_train]
        ).to(dtype=torch.double)

        self._update_model(
            X_train=self.x_train, Y_train=self.y_train)
        self._update_preference_model()
        self.test()
        logger.info("Test GP performance:\n %s", self.GP_performance[-1, :])
        logger.info("Test sampled performance:\n %s", self.sampled_performance[-1, :])

        num_sim = 0
        num_dm = 0
        # start iterating until the budget is exhausted.
        # print("range(self.n_max - self.n_init)",range(self.n_max - self.n_init))
        schedule_dm_sampling = torch.linspace(start=0, end=self.n_max - self.n_init- 1, steps=self.n_pairs, dtype=torch.int)
        for it in range(self.n_max - self.n_init):


            # if voi simulator greater than dm then query simulator. Otherwise query the decision maker
            if it in schedule_dm_sampling:
                num_dm += 1
                self.random_pairs_evaluation_dm(n_pairs=1)
            else:
                num_sim +=1
                x_new, voi_sim = self.get_next_point_simulator()
                x_new = x_new.to(dtype=torch.double)
                y_new = self.evaluate_objective(x_new).to(dtype=torch.double)

                # update stored data
                self.x_train = torch.vstack([self.x_train, x_new.reshape(1, -1)])
                self.y_train = torch.vstack((self.y_train, y_new))
                self.decisions.append(1)

            logger.info(f"Running optim, n: {self.x_train.shape[0] + len(self.index_pairs_sampled)}")

            # test if necessary
            if torch.any(len(self.y_train) == self.testable_iters):
                self.test()
                logger.info("Test GP performance:\n %s", self.GP_performance[-1, :])
                logger.info("Test sampled performance:\n %s", self.sampled_performance[-1, :])

        # select final self.n_pairs pairs.

    def random_pairs_evaluation_dm(self, n_pairs):
        for it in range(n_pairs):
                pair_new_idx, y_1, y_2 = self.select_random_non_dominated_pair()

                if (y_1 is None) or (y_2 is None):
                    self._update_preference_model()
                    self.test()
                    logger.info("Test GP performance:\n %s", self.GP_performance[-1, :])
                    logger.info("Test sampled performance:\n %s", self.sampled_performance[-1, :])
                    break

                y_winner, y_loser = self.evaluate_decision_maker(option_1=y_1,
                                                     option_2=y_2)

                self.y_train_option_1 = torch.vstack([self.y_train_option_1, y_winner.reshape(1, -1)])
                self.y_train_option_2 = torch.vstack([self.y_train_option_2, y_loser.reshape(1, -1)])
                self.index_pairs_sampled.append(pair_new_idx)
                self.decisions.append(0)

                if it == range(n_pairs)[-1]:
                    self._update_preference_model()
                    self.test()
                    logger.info("Test GP performance:\n %s", self.GP_performance[-1, :])
                    logger.info("Test sampled performance:\n %s", self.sampled_performance[-1, :])


    def evaluate_objective(self, x: Tensor, **kwargs) -> Tensor:
        """
        evaluate objective function f(x)
        """

    def evaluate_decision_maker(self, option_1: Tensor,
                                option_2: Tensor, **kwargs) -> Tensor:
        """
        "Returns the decision maker preference given two distinct options"
        """

    @abstractmethod
    def get_next_point_simulator(self):
        """
        return next (x) point.
        """

    @abstractmethod
    def get_next_point_decision_maker(self):
        """
        return next pair of y point.
        """

    @abstractmethod
    def policy(self):
        """
        Return the recommended x value
        """

    def save(self):
        """
        saves intermediate results in directory.
        """

    def test(self):
        """
        test and saves performance measures
        """
