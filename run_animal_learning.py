import math
import sys
from collections import deque
from pathlib import Path

from envs.animal_learning import TracePatterning
from experiment.config import ExperimentJSON
from experiment.metric import Metric
from networks.factory import get_network
from experiment.checkpoint import Checkpoints, writer_lock


def main():
    my_experiment = ExperimentJSON(sys.argv[1:])
    my_experiment.validate_animal_learning()
    with writer_lock(Path('results') / my_experiment.database_name):
        run_experiment(my_experiment)


def run_experiment(my_experiment):
    my_experiment.check_output()
    network = None if my_experiment.resume else get_network(my_experiment)

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
    env = None if my_experiment.resume else TracePatterning(ISI, ITI, 5, my_experiment.get_int_param("seed"))

    checkpoints = Checkpoints(my_experiment, (error_metric, avg_error))
    state = checkpoints.restore(my_experiment.resume) if my_experiment.resume else None
    if state is None:
        my_experiment._write_runs_row()
    try:
        train(my_experiment, network, env, error_metric, avg_error, checkpoints, state)
    finally:
        error_metric.commit_values()
        avg_error.commit_values()


def train(my_experiment, network, env, error_metric, avg_error, checkpoints, state=None):
    print("Network created")
    if state is None:
        state = {'network': network, 'env': env, 'next_step': 0, 'layer': 0,
                 'old_pred': network.forward(env.reset()), 'running_error': 0.0,
                 'errors': deque()}
        checkpoints.save(state)
    try:
        train_steps(my_experiment, checkpoints, state, error_metric, avg_error)
    except BaseException:
        checkpoints.status('interrupted_or_failed', state['next_step'])
        raise


def train_steps(my_experiment, checkpoints, state, error_metric, avg_error):
    network, env = state['network'], state['env']
    running_error, list_of_errors = state['running_error'], state['errors']
    old_pred, layer = state['old_pred'], state['layer']
    steps = my_experiment.get_int_param('steps')
    feature_increase = int(my_experiment.get_vector_param('features')[2])
    gamma, K = .90, 100000
    for i in range(state['next_step'], steps):
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
        if not all(math.isfinite(v) for v in (pred, real_target, real_error, error)):
            raise FloatingPointError(f"Non-finite prediction or error at step {i}")

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
        state.update(next_step=i + 1, old_pred=old_pred, layer=layer,
                     running_error=running_error)
        if (i + 1) % my_experiment.checkpoint_every == 0 and i + 1 < steps:
            checkpoints.save(state)

    checkpoints.save(state, status='completed')


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, FloatingPointError) as exc:
        print(f"Experiment failed: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Experiment interrupted; recorded metrics have been flushed", file=sys.stderr)
        sys.exit(130)
