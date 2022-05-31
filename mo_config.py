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
    "Spherical_0.01_Interactive_experiments": {
        "problems": ["Spherical"],
        "method": ["Interactive"],  # Interactive or VoISim
        "output_dim": 2,
        "input_dim": 2,
        "shift_parameter": 0.01,
        "number_pairs": 0,
        "number_of_scalarizations": 30,
        "num_samples_initial_design": 6,
        "num_max_evaluatations": 30,
        "utility_model": ["Tche"],
        "num_discrete_points": 3,
        "num_fantasies": 20,
        "num_restarts_inner_optimizer": 1,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "Adam",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 5,
        "raw_samples_acq_optimizer": 100,
    },

    "Spherical_0.01_0_pair_experiments": {
        "problems": ["Spherical"],
        "method": ["VoISim"], #Interactive or VoISim
        "output_dim": 2,
        "input_dim": 2,
        "shift_parameter": 0.01,
        "number_pairs" : 0,
        "number_of_scalarizations": 30,
        "num_samples_initial_design": 6,
        "num_max_evaluatations": 30,
        "utility_model": ["Tche"],
        "num_discrete_points": 3,
        "num_fantasies": 20,
        "num_restarts_inner_optimizer": 1,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "Adam",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 5,
        "raw_samples_acq_optimizer": 100,
    },

    "Spherical_0.01_3_pair_experiments": {
        "problems": ["Spherical"],
        "method": ["VoISim"],  # Interactive or VoISim
        "output_dim": 2,
        "input_dim": 2,
        "shift_parameter": 0.01,
        "number_pairs": 3,
        "number_of_scalarizations": 30,
        "num_samples_initial_design": 6,
        "num_max_evaluatations": 27,
        "utility_model": ["Tche"],
        "num_discrete_points": 3,
        "num_fantasies": 20,
        "num_restarts_inner_optimizer": 1,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "Adam",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 5,
        "raw_samples_acq_optimizer": 100,
    },

    "Spherical_0.01_6_pair_experiments": {
        "problems": ["Spherical"],
        "method": ["VoISim"],  # Interactive or VoISim
        "output_dim": 2,
        "input_dim": 2,
        "shift_parameter": 0.01,
        "number_pairs": 3,
        "number_of_scalarizations": 30,
        "num_samples_initial_design": 6,
        "num_max_evaluatations": 24,
        "utility_model": ["Tche"],
        "num_discrete_points": 3,
        "num_fantasies": 20,
        "num_restarts_inner_optimizer": 1,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "Adam",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 5,
        "raw_samples_acq_optimizer": 100,
    },

    "Spherical_0.01_9_pair_experiments": {
        "problems": ["Spherical"],
        "method": ["VoISim"],  # Interactive or VoISim
        "output_dim": 2,
        "input_dim": 2,
        "shift_parameter": 0.01,
        "number_pairs": 3,
        "number_of_scalarizations": 30,
        "num_samples_initial_design": 6,
        "num_max_evaluatations": 21,
        "utility_model": ["Tche"],
        "num_discrete_points": 3,
        "num_fantasies": 20,
        "num_restarts_inner_optimizer": 1,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "Adam",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 5,
        "raw_samples_acq_optimizer": 100,
    },

    "Spherical_0.01_12_pair_experiments": {
        "problems": ["Spherical"],
        "method": ["VoISim"],  # Interactive or VoISim
        "output_dim": 2,
        "input_dim": 2,
        "shift_parameter": 0.01,
        "number_pairs": 3,
        "number_of_scalarizations": 30,
        "num_samples_initial_design": 6,
        "num_max_evaluatations": 18,
        "utility_model": ["Tche"],
        "num_discrete_points": 3,
        "num_fantasies": 20,
        "num_restarts_inner_optimizer": 1,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "Adam",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 5,
        "raw_samples_acq_optimizer": 100,
    },

    "Spherical_0.01_experiments": {
        "problems": ["Spherical"],
        "method": ["Interactive"],  # Interactive or VoISim
        "output_dim": 2,
        "input_dim": 2,
        "shift_parameter": 0.01,
        "number_pairs": 4,
        "number_of_scalarizations": 100,
        "num_samples_initial_design": 6,
        "num_max_evaluatations": 30,
        "utility_model": ["Tche"],
        "num_discrete_points": 3,
        "num_fantasies": 20,
        "num_restarts_inner_optimizer": 1,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "Adam",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 5,
        "raw_samples_acq_optimizer": 100,
    },

    "Spherical_0.1_Interactive_experiments": {
        "problems": ["Spherical"],
        "method": ["Interactive"],
        "output_dim": 2,
        "input_dim": 2,
        "shift_parameter": 0.1,
        "number_of_scalarizations": 30,
        "num_samples_initial_design": 6,
        "num_max_evaluatations": 30,
        "utility_model": ["Tche"],
        "num_discrete_points": 3,
        "num_fantasies": 20,
        "num_restarts_inner_optimizer": 1,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "Adam",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 5,
        "raw_samples_acq_optimizer": 100,
    },
    "Spherical_0.1_0_pair_experiments": {
            "problems": ["Spherical"],
            "method": ["VoISim"],
            "output_dim": 2,
            "input_dim": 2,
            "shift_parameter": 0.1,
            "number_pairs" : 0,
            "number_of_scalarizations": 30,
            "num_samples_initial_design": 6,
            "num_max_evaluatations": 30,
            "utility_model": ["Tche"],
            "num_discrete_points": 3,
            "num_fantasies": 20,
            "num_restarts_inner_optimizer": 1,
            "raw_samples_inner_optimizer": 100,
            "acquisition_optimizer":
                "Adam",  # "L-BFGS-B" or "Adam"
            "num_restarts_acq_optimizer": 5,
            "raw_samples_acq_optimizer": 100,
        },
    "Spherical_0.1_3_pair_experiments": {
            "problems": ["Spherical"],
            "method": ["VoISim"],
            "output_dim": 2,
            "input_dim": 2,
            "shift_parameter": 0.1,
            "number_pairs" : 3,
            "number_of_scalarizations": 30,
            "num_samples_initial_design": 6,
            "num_max_evaluatations": 27,
            "utility_model": ["Tche"],
            "num_discrete_points": 3,
            "num_fantasies": 20,
            "num_restarts_inner_optimizer": 1,
            "raw_samples_inner_optimizer": 100,
            "acquisition_optimizer":
                "Adam",  # "L-BFGS-B" or "Adam"
            "num_restarts_acq_optimizer": 5,
            "raw_samples_acq_optimizer": 100,
        },
    "Spherical_0.1_6_pair_experiments": {
            "problems": ["Spherical"],
            "method": ["VoISim"],
            "output_dim": 2,
            "input_dim": 2,
            "shift_parameter": 0.1,
            "number_pairs" : 6,
            "number_of_scalarizations": 30,
            "num_samples_initial_design": 6,
            "num_max_evaluatations": 24,
            "utility_model": ["Tche"],
            "num_discrete_points": 3,
            "num_fantasies": 20,
            "num_restarts_inner_optimizer": 1,
            "raw_samples_inner_optimizer": 100,
            "acquisition_optimizer":
                "Adam",  # "L-BFGS-B" or "Adam"
            "num_restarts_acq_optimizer": 5,
            "raw_samples_acq_optimizer": 100,
        },
    "Spherical_0.1_9_pair_experiments": {
            "problems": ["Spherical"],
            "method": ["VoISim"],
            "output_dim": 2,
            "input_dim": 2,
            "shift_parameter": 0.1,
            "number_pairs" : 9,
            "number_of_scalarizations": 30,
            "num_samples_initial_design": 6,
            "num_max_evaluatations": 21,
            "utility_model": ["Tche"],
            "num_discrete_points": 3,
            "num_fantasies": 20,
            "num_restarts_inner_optimizer": 1,
            "raw_samples_inner_optimizer": 100,
            "acquisition_optimizer":
                "Adam",  # "L-BFGS-B" or "Adam"
            "num_restarts_acq_optimizer": 5,
            "raw_samples_acq_optimizer": 100,
        },
    "Spherical_0.1_12_pair_experiments": {
            "problems": ["Spherical"],
            "method": ["VoISim"],
            "output_dim": 2,
            "input_dim": 2,
            "shift_parameter": 0.1,
            "number_pairs" : 12,
            "number_of_scalarizations": 30,
            "num_samples_initial_design": 6,
            "num_max_evaluatations": 18,
            "utility_model": ["Tche"],
            "num_discrete_points": 3,
            "num_fantasies": 20,
            "num_restarts_inner_optimizer": 1,
            "raw_samples_inner_optimizer": 100,
            "acquisition_optimizer":
                "Adam",  # "L-BFGS-B" or "Adam"
            "num_restarts_acq_optimizer": 5,
            "raw_samples_acq_optimizer": 100,
        },

    "Spherical_0.3_experiments": {
        "problems": ["Spherical"],
        "method": ["Interactive"],
        "output_dim": 2,
        "input_dim": 2,
        "shift_parameter": 0.3,
        "number_of_scalarizations": 100,
        "num_samples_initial_design": 6,
        "num_max_evaluatations": 30,
        "utility_model": ["Tche"],
        "num_discrete_points": 3,
        "num_fantasies": 20,
        "num_restarts_inner_optimizer": 1,
        "raw_samples_inner_optimizer": 100,
        "acquisition_optimizer":
            "Adam",  # "L-BFGS-B" or "Adam"
        "num_restarts_acq_optimizer": 5,
        "raw_samples_acq_optimizer": 100,
    }
}
