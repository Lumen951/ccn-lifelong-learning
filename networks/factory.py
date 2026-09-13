from networks.incremental import IncrementalNetworks, NormalizedIncrementalNetworks, Snap1
from networks.bptt import DenseLSTM, DenseLSTMRmsProp


def get_network(experiment_config):
    print("Layer size =", experiment_config.get_vector_param("features")[1])
    algorithm = experiment_config.get_string_param("algorithm")

    if algorithm == "constructive":
        network = IncrementalNetworks(
            experiment_config.get_float_param("step_size"),
            experiment_config.get_int_param("seed"),
            experiment_config.get_int_param("input_features"), 1,
            int(experiment_config.get_vector_param("features")[0]), 1,
            experiment_config.get_float_param("std_cap"),
            experiment_config.get_float_param("normalization_rate"))
    elif algorithm == "columnar":
        network = IncrementalNetworks(
            experiment_config.get_float_param("step_size"),
            experiment_config.get_int_param("seed"),
            experiment_config.get_int_param("input_features"), 1,
            int(experiment_config.get_vector_param("features")[0]),
            experiment_config.get_int_param("features"),
            experiment_config.get_float_param("std_cap"),
            experiment_config.get_float_param("normalization_rate"))
    elif algorithm == "hybrid":
        network = IncrementalNetworks(
            experiment_config.get_float_param("step_size"),
            experiment_config.get_int_param("seed"),
            experiment_config.get_int_param("input_features"), 1,
            int(experiment_config.get_vector_param("features")[0]),
            int(experiment_config.get_vector_param("features")[1]),
            experiment_config.get_float_param("std_cap"),
            experiment_config.get_float_param("normalization_rate"))
    elif algorithm == "hybrid_normalized":
        network = NormalizedIncrementalNetworks(
            experiment_config.get_float_param("step_size"),
            experiment_config.get_int_param("seed"),
            experiment_config.get_int_param("input_features"), 1,
            int(experiment_config.get_vector_param("features")[0]),
            int(experiment_config.get_vector_param("features")[1]),
            experiment_config.get_float_param("std_cap"),
            experiment_config.get_float_param("normalization_rate"))
    elif algorithm == "snap":
        network = Snap1(
            experiment_config.get_float_param("step_size"),
            experiment_config.get_int_param("seed"),
            experiment_config.get_int_param("input_features"), 1,
            int(experiment_config.get_vector_param("features")[0]),
            int(experiment_config.get_vector_param("features")[1]),
            experiment_config.get_float_param("std_cap"))
    elif algorithm == "tbptt":
        network = DenseLSTM(
            experiment_config.get_float_param("step_size"),
            experiment_config.get_int_param("seed"),
            int(experiment_config.get_vector_param("features_truncation")[1]),
            experiment_config.get_int_param("input_features"),
            int(experiment_config.get_vector_param("features_truncation")[0]))
    elif algorithm == "tbptt_rmsprop":
        network = DenseLSTMRmsProp(
            experiment_config.get_float_param("step_size"),
            experiment_config.get_int_param("seed"),
            int(experiment_config.get_vector_param("features_truncation")[1]),
            experiment_config.get_int_param("input_features"),
            int(experiment_config.get_vector_param("features_truncation")[0]),
            experiment_config.get_float_param("beta_2"),
            experiment_config.get_float_param("eps"))
    else:
        raise ValueError("Unknown algorithm: " + algorithm)

    return network
