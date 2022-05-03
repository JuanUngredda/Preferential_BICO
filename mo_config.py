# Available synthetic Problems:

# 2D problems:

# "Egg-holder"
# "Sum of Powers"
# "Branin"
# "Cosines"
# "Mccormick"
# "Goldstein"
# "Six-hump camel"
# "dropwave"
# "Rosenbrock"
# "beale"

CONFIG_DICT = {
    "C2DTLZ2_experiments": {
        "problems": ["C2DTLZ2"],
        "method": ["macKG"],
        "number_of_scalarizations": 10 ,
        "num_samples_initial_design": 10,
        "num_max_evaluatations": 100,
        "utility_model": ["Lin", "Tche"],
        "num_discrete_points": None,
        "num_fantasies": 5,
        "num_restarts_inner_optimizer": 1 ,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "Adam",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 1,
        "raw_samples_acq_optimizer": 100,
    },
    "BraninCurrin_experiments": {
        "problems": ["ConstrainedBraninCurrin"],
        "method": ["macKG"],
        "number_of_scalarizations": 10,
        "num_samples_initial_design": 10,
        "num_max_evaluatations": 100,
        "utility_model": ["Lin", "Tche"],
        "num_discrete_points": None,
        "num_fantasies": 5,
        "num_restarts_inner_optimizer": 1,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "Adam",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 1,
        "raw_samples_acq_optimizer": 100,
    }
}
