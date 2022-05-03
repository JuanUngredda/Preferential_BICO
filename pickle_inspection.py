import pickle

import matplotlib.pyplot as plt
import torch

# read python dict back from the file
from botorch.test_functions.multi_objective import C2DTLZ2

methods = ["macKG"]


for i, m in enumerate(methods):
    pkl_file = open(
        "/home/juan/Documents/Github_repos/botorch/experiment_scripts/results/C2DTLZ2/"
        + m
        + "/0.pkl",
        "rb",
    )
    mydict2 = pickle.load(pkl_file)
    pkl_file.close()

    X = mydict2["x"]
    Y = mydict2["y"]
    C = mydict2["c"]
    # print("mydict", mydict2)
    # raise
    # print("average_time_acq", mydict2[""])
    d = 4
    M = 2

    print(mydict2["method_times"])
    fun = C2DTLZ2(dim=d, num_objectives=M, negate=True)
    bounds = fun.bounds  # Bounds tensor (2, d)
    print(bounds)
    X_plot = torch.rand(10000, d) * (bounds[1] - bounds[0]) + bounds[0]
    Y_plot = fun(X_plot).numpy()
    C_plot = fun.evaluate_slack(X_plot).numpy()
    is_feas = -C_plot < 0
    is_feas = is_feas.reshape(-1)

    Y_feas = Y#[C.squeeze()<0]
    plt.scatter(Y_plot[is_feas,0], Y_plot[is_feas,1])
    plt.scatter(Y_feas[:, 0], Y_feas[:, 1])
    plt.show()


    plt.title(m + " performance from GP")
    plt.plot(mydict2["OC_GP"][:, 0], mydict2["OC_GP"][:, 1])
    plt.legend()
    plt.show()

    plt.title(m + " performance from sampled")
    plt.plot(mydict2["OC_sampled"][:, 0], mydict2["OC_sampled"][:, 1])
    plt.legend()
    plt.show()


# pkl_file = open(
#     "/home/juan/Documents/Github_repos/botorch/experiment_scripts/results/Branin/HYBRIDKG/0.pkl",
#     "rb",
# )
# mydict2 = pickle.load(pkl_file)
# pkl_file.close()
#
# print(mydict2["method_times"])
