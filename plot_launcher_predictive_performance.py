import logging
import os
import pickle as pkl
import subprocess as sp
import sys
from itertools import product

import torch

from botorch.test_functions.multi_objective import BNH, SRN, CONSTR, ConstrainedBraninCurrin, C2DTLZ2, OSY, WeldedBeam
from optimizers.objective_functions.multi_objective_test_function import Spherical
from mo_config import CONFIG_DICT
from optimizers.mo_optimizer import Optimizer
from optimizers.utils import mo_acq_wrapper, test_function_handler

logger = logging.getLogger(__name__)

GIT_ROOT = sp.check_output(["git", "rev-parse", "--show-toplevel"]).decode()[:-1]

# set path to use copy of source
sys.path.insert(0, GIT_ROOT + "/experiment_scripts/")

HOSTNAME = sp.check_output(["hostname"], shell=True).decode()[:-1]

# set directory to save files
script_dir = os.path.dirname(os.path.abspath(__file__))

dtype = torch.double


def run_experiment(
        experiment_name: str,
        utility_model_str: str,
        problem: str,
        method: str,
        savefile: str,
        base_seed: int,
        n_init=4,
        n_max=50,
):
    """
    ARGS:
        problem: str, a key from the dict of problems
        method: str, name of optimizer
        savefile: str, locaiotn to write output results
        base_seed: int, generates testfun
        n_init: int, starting budfget of optimizer
        n_max: ending budget for optimizer
    """

    # print out all the passed arguments

    # instantiate the test problem
    testfun_dict = {
        "BNH": BNH,
        "SRN": SRN,
        "CONSTR": CONSTR,
        "Spherical": Spherical,
        "ConstrainedBraninCurrin": ConstrainedBraninCurrin,
        "C2DTLZ2": C2DTLZ2,
        "OSY": OSY,
        "WeldedBeam": WeldedBeam
    }

    CONFIG_NUMBER_FANTASIES = CONFIG_DICT[experiment_name]["num_fantasies"]
    # TODO: Change objective function parametrisation to not be hard-coded
    CONFIG_NUMBER_INPUT_DIM = CONFIG_DICT[experiment_name][
        "input_dim"
    ]
    CONFIG_NUMBER_OUTPUT_DIM = CONFIG_DICT[experiment_name][
        "output_dim"
    ]

    CONFIG_NUMBER_PAIRS = CONFIG_DICT[experiment_name]["number_pairs"]

    CONFIG_SHIFT_PARAMETER = CONFIG_DICT[experiment_name][
        "shift_parameter"
    ]

    testfun = test_function_handler(test_fun_str=problem,
                                    test_fun_dict=testfun_dict,
                                    shift_parameter=CONFIG_SHIFT_PARAMETER,
                                    input_dim=CONFIG_NUMBER_INPUT_DIM,
                                    output_dim=CONFIG_NUMBER_OUTPUT_DIM).to(dtype=dtype)

    dim = testfun.dim
    bounds = testfun.bounds  # Bounds tensor (2, d)
    lb, ub = bounds
    testfun.problem = problem
    bounds_normalized = torch.vstack([torch.zeros(dim), torch.ones(dim)])

    CONFIG_UTILITY_MODEL = utility_model_str

    CONFIG_NUMBER_FANTASIES = CONFIG_DICT[experiment_name]["num_fantasies"]
    CONFIG_NUMBER_OF_SCALARIZATIONS = CONFIG_DICT[experiment_name][
        "number_of_scalarizations"
    ]
    CONFIG_NUMBER_DISCRETE_POINTS = CONFIG_DICT[experiment_name]["num_discrete_points"]
    CONFIG_NUMBER_RESTARTS_INNER_OPT = CONFIG_DICT[experiment_name][
        "num_restarts_inner_optimizer"
    ]
    CONFIG_NUMBER_RAW_SAMPLES_INNER_OPT = CONFIG_DICT[experiment_name][
        "raw_samples_inner_optimizer"
    ]

    CONFIG_ACQ_OPTIMIZER = CONFIG_DICT[experiment_name]["acquisition_optimizer"]

    CONFIG_NUMBER_RESTARTS_ACQ_OPT = CONFIG_DICT[experiment_name][
        "num_restarts_acq_optimizer"
    ]
    CONFIG_NUMBER_RAW_SAMPLES_ACQ_OPT = CONFIG_DICT[experiment_name][
        "raw_samples_acq_optimizer"
    ]

    CONFIG_NUMBER_INITAL_DESIGN = CONFIG_DICT[experiment_name][
        "num_samples_initial_design"
    ]

    CONFIG_MAX_NUM_EVALUATIONS = CONFIG_DICT[experiment_name][
        "num_max_evaluatations"
    ]

    acquisition_function, recommender_function = mo_acq_wrapper(method=method,
                                                                utility_model_name=CONFIG_UTILITY_MODEL,
                                                                num_fantasies=CONFIG_NUMBER_FANTASIES)
    # instantiate the optimizer
    acquisition_function_optimizer = Optimizer

    optimizer = acquisition_function_optimizer(
        testfun=testfun,
        acquisitionfun=acquisition_function,
        recommenderfun=recommender_function,
        utility_model_name=CONFIG_UTILITY_MODEL,
        num_scalarizations=CONFIG_NUMBER_OF_SCALARIZATIONS,
        lb=lb,
        ub=ub,
        n_pairs=CONFIG_NUMBER_PAIRS,
        n_init=CONFIG_NUMBER_INITAL_DESIGN,  # n_init,
        n_max=CONFIG_MAX_NUM_EVALUATIONS,  # n_max,
        kernel_str="Matern",
        save_folder=savefile,
        base_seed=base_seed,
        optional={
            "NOISE_OBJECTIVE": False,
            "OPTIMIZER": CONFIG_ACQ_OPTIMIZER,
            "RAW_SAMPLES": CONFIG_NUMBER_RAW_SAMPLES_ACQ_OPT,
            "NUM_RESTARTS": CONFIG_NUMBER_RESTARTS_ACQ_OPT,
        },
    )

    from optimizers.utils import lhc

    optimizer.x_train = lhc(n=15, dim=optimizer.dim).to(dtype=torch.double)
    optimizer.y_train = torch.vstack(
        [optimizer.evaluate_objective(x_i) for x_i in optimizer.x_train]
    ).to(dtype=torch.double)

    optimizer._update_model(
        X_train=optimizer.x_train, Y_train=optimizer.y_train)
    optimizer._update_preference_model()

    bounds_normalized = torch.vstack(
        [torch.zeros((1, optimizer.dim)), torch.ones((1, optimizer.dim))]
    )
    optimizer.num_scalarisations = 1000

    plot_X = lhc(n=100, dim=optimizer.dim).to(dtype=torch.double)

    GPmodel = optimizer.model.posterior(plot_X)

    mean = GPmodel.mean.detach()

    import matplotlib.pyplot as plt

    plt.scatter(mean[:,0], mean[:,1])
    plt.show()






def main(exp_names, seed):
    # make table of experiment settings
    # seed += 7
    EXPERIMENT_NAME = exp_names
    PROBLEMS = CONFIG_DICT[EXPERIMENT_NAME]["problems"]
    ALGOS = CONFIG_DICT[EXPERIMENT_NAME]["method"]
    UTILITY = CONFIG_DICT[EXPERIMENT_NAME]["utility_model"]
    EXPERIMENTS = list(product(*[PROBLEMS, ALGOS, UTILITY]))
    logger.info(f"Running experiment: {seed} of {len(EXPERIMENTS)}")

    # run that badboy
    for idx, _ in enumerate(EXPERIMENTS):

        file_name = script_dir + "/results/" + EXPERIMENT_NAME + "/" + EXPERIMENTS[idx][0] + "/" + EXPERIMENTS[idx][
            1] + "/" + EXPERIMENTS[idx][2] + "/" + str(seed) + ".pkl"

        if os.path.isfile(file_name) == False:
            run_experiment(
                experiment_name=EXPERIMENT_NAME,
                problem=EXPERIMENTS[idx][0],
                method=EXPERIMENTS[idx][1],
                utility_model_str=EXPERIMENTS[idx][2],
                savefile=script_dir
                         + "/results/" +
                         EXPERIMENT_NAME + "/"
                         + EXPERIMENTS[idx][0]
                         + "/"
                         + EXPERIMENTS[idx][1] + "/"
                         + EXPERIMENTS[idx][2],
                base_seed=10,
            )


if __name__ == "__main__":
    main(sys.argv[1:])

    # parser = argparse.ArgumentParser(description="Run KG experiment")
    # parser.add_argument("--seed", type=int, help="base seed", default=0)
    # parser.add_argument("--exp_name", type=str, help="Experiment name in config file")
    # args = parser.parse_args()
