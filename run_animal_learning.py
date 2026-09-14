import sys
from collections import deque

from envs.animal_learning import TracePatterning
from experiment.config import ExperimentJSON
from experiment.metric import Metric
from networks.factory import get_network


def main():
    my_experiment = ExperimentJSON(sys.argv[1:])

    error_metric = Metric(my_experiment.database_name, "error_table",
                          ["run", "step", "error"],
                          ["int", "int", "real"], ["run", "step"])

    avg_error = Metric(my_experiment.database_name, "predictions",
                       ["run", "step", "x0", "x1", "x2", "x3", "x4", "x5", "x6",
                        "pred", "target"],
                       ["int", "int", "real", "real", "real", "real", "real", "real",
                        "real", "real", "real"], ["run", "step"])

    print("Program started")

    ISI = (24, 36)
    ITI = (80, 120)
    env = TracePatterning(ISI, ITI, 5, my_experiment.get_int_param("seed"))

    network = get_network(my_experiment)
    K = 100000
    print("Network created")
    running_error = 0.0
    list_of_errors = deque()
    gamma = 0.90
    x = env.reset()
    old_pred = network.forward(x)
    layer = 0
    steps = my_experiment.get_int_param("steps")
    feature_increase = int(my_experiment.get_vector_param("features")[2])
    for i in range(steps):
        if i % feature_increase == feature_increase - 1:
            layer += 1
            print("Increasing layer")

        x = env.step()

        network.decay_gradient(my_experiment.get_float_param("lambda") * gamma)
        network.backward(layer)

        pred = network.forward(x)
        real_target = env.get_target(gamma)
        real_error = (real_target - pred) * (real_target - pred)
        target = env.get_US() + gamma * pred
        error = target - old_pred

        network.update_parameters(layer, error)
        old_pred = pred

        if i % 1000000 < 400:
            cur_error = [my_experiment.get_int_param("run"), i]
            for inner_c in range(7):
                cur_error.append(x[inner_c])
            cur_error.append(pred)
            cur_error.append(real_target)
            avg_error.record_value(cur_error)

        list_of_errors.append(real_error)
        running_error += real_error
        if len(list_of_errors) > K:
            running_error -= list_of_errors[0]
            list_of_errors.popleft()
        if len(list_of_errors) > K:
            print("Queue longer than 100000; shouldn't happen. Exiting")
            sys.exit(0)

        if i % 50000 == 20000 and i > K:
            error_metric.record_value(
                [my_experiment.get_int_param("run"), i, running_error / K])

        if i % 500000 == 0:
            print("Step =", i, "Error =", running_error / K)
            error_metric.commit_values()
            avg_error.commit_values()

    error_metric.commit_values()
    avg_error.commit_values()


if __name__ == "__main__":
    main()
