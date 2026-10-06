import argparse
import csv
import json
import math
import os
import re

_INT_PREFIX = re.compile(r"^[+-]?\d+")
_FLOAT_PREFIX = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")


def parse_params(argv):
    result = {}
    key = None
    for arg in argv:
        if arg.startswith("--"):
            key = arg[2:]
            result[key] = []
        elif key is not None:
            result[key].append(arg)
    return result


def _prod(values):
    prod = 1
    for v in values:
        prod *= v
    return prod


def _iter_items(j):
    # C++ nlohmann::json iterates objects alphabetically (std::map);
    # sorted() keeps run -> params mapping identical to the original
    if isinstance(j, dict):
        return [(k, j[k]) for k in sorted(j)]
    return list(enumerate(j))


def get_combinations(j):
    combination_list = []
    for _, value in _iter_items(j):
        if isinstance(value, list):
            object_combi = 0
            for item in value:
                if isinstance(item, dict):
                    object_combi += _prod(get_combinations(item))
                else:
                    object_combi += 1
            combination_list.append(object_combi)
        elif isinstance(value, dict):
            combination_list.append(_prod(get_combinations(value)))
        else:
            combination_list.append(1)
    return combination_list


def get_args_for_run(j, rank):
    my_map = {}
    freq = get_combinations(j)
    rank = rank % _prod(freq)
    for counter, (key, value) in enumerate(_iter_items(j)):
        temp_index = rank % freq[counter]
        rank //= freq[counter]
        if not isinstance(value, (dict, list)):
            my_map[key] = value
        elif isinstance(value, list) and not isinstance(value[0], dict):
            my_map[key] = value[temp_index]
        elif isinstance(value, dict):
            for k, v in get_args_for_run(value, temp_index).items():
                my_map.setdefault(k, v)
        else:
            inner_freq = get_combinations(value)
            inner_index = 0
            running = inner_freq[0]
            old_sum = 0
            while temp_index >= running:
                inner_index += 1
                old_sum = running
                running += inner_freq[inner_index]
            temp_index -= old_sum
            chosen = value[inner_index]
            if key not in my_map:
                my_map[key] = sorted(chosen)[0]
            for k, v in get_args_for_run(chosen, temp_index).items():
                my_map.setdefault(k, v)
    return my_map


def _parse_int_prefix(value):
    # std::stoi semantics: parse the leading integer prefix
    match = _INT_PREFIX.match(str(value).strip())
    if match is None:
        raise ValueError(f"Expected an integer, got {value!r}")
    return int(match.group(0))


def _parse_float_prefix(value):
    # std::stof semantics: parse the leading float prefix
    match = _FLOAT_PREFIX.match(str(value).strip())
    if match is None:
        raise ValueError(f"Expected a number, got {value!r}")
    return float(match.group(0))


def load_config(path):
    with open(path, encoding="utf-8-sig") as f:
        config = json.load(f)
    if not isinstance(config, dict) or not isinstance(config.get("experiment"), dict):
        raise ValueError("Config must contain an experiment object")
    experiment = config["experiment"]
    name = experiment.get("name")
    if not isinstance(name, str) or not name or name in (".", "..") or any(
            c in name for c in '/\\:<>"|?*') or name.endswith((".", " ")):
        raise ValueError("Experiment name must be a valid single directory name")
    if not isinstance(experiment.get("params"), dict) or not experiment["params"]:
        raise ValueError("Experiment params must be a nonempty object")

    def validate(value):
        if isinstance(value, list):
            if not value:
                raise ValueError("Parameter choices must not be empty")
            if any(isinstance(v, dict) for v in value) and not all(isinstance(v, dict) for v in value):
                raise ValueError("Parameter choices cannot mix objects and scalar values")
            for item in value:
                if isinstance(item, dict) and len(item) != 1:
                    raise ValueError("Each algorithm choice must contain one named branch")
                validate(item)
        elif isinstance(value, dict):
            for item in value.values():
                validate(item)
        elif not isinstance(value, (str, int, float)) or isinstance(value, bool):
            raise ValueError("Parameter values must be strings or numbers")
    validate(config)
    return config


class ExperimentJSON:
    def __init__(self, argv):
        parser = argparse.ArgumentParser(description="Run an animal-learning experiment")
        parser.add_argument("--config", required=True)
        parser.add_argument("--run", required=True, type=int)
        parser.add_argument("--checkpoint-every", type=int, default=50000)
        parser.add_argument("--resume", help="Path to a checkpoint from this experiment")
        args = parser.parse_args(argv)
        if args.checkpoint_every <= 0:
            raise ValueError('checkpoint-every must be positive')
        self.checkpoint_every = args.checkpoint_every
        self.resume = args.resume
        run = args.run
        j = load_config(args.config)
        total_combinations = _prod(get_combinations(j))
        if not 0 <= run < total_combinations:
            raise ValueError(f"Run must be between 0 and {total_combinations - 1}")
        print("TOTAL COMBINATIONS =", total_combinations)

        self.args_for_run = get_args_for_run(j, run)
        self.args_for_run["run"] = str(run)
        for key in sorted(self.args_for_run):
            print(key, self.args_for_run[key])

        self.run = run
        self.database_name = self.args_for_run["name"]
        self.run_columns = sorted({"run"}.union(*(get_args_for_run(j, i).keys()
                                                  for i in range(total_combinations))))

    def validate_animal_learning(self):
        algorithm = self.get_string_param("algorithm")
        if algorithm not in {"constructive", "columnar", "hybrid", "hybrid_normalized",
                             "snap", "tbptt", "tbptt_rmsprop"}:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        features = self.get_vector_param("features")
        if len(features) != 3 or any(int(v) <= 0 for v in features):
            raise ValueError("features must be three positive integers separated by ':'")
        if self.get_int_param("input_features") != 12:
            raise ValueError("TracePatterning requires input_features=12")
        if self.get_int_param("steps") <= 0 or self.get_int_param("seed") < 0:
            raise ValueError("steps must be positive and seed must be nonnegative")
        for key in ("step_size", "std_cap"):
            value = self.get_float_param(key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        value = self.get_float_param("lambda")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("lambda must be between 0 and 1")
        if algorithm in {"constructive", "columnar", "hybrid", "hybrid_normalized"}:
            value = self.get_float_param("normalization_rate")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("normalization_rate must be between 0 and 1")
        if algorithm.startswith("tbptt"):
            values = self.get_vector_param("features_truncation")
            if len(values) != 2 or any(int(v) <= 0 for v in values):
                raise ValueError("features_truncation must contain two positive integers")
        if algorithm == "tbptt_rmsprop":
            beta = self.get_float_param("beta_2")
            eps = self.get_float_param("eps")
            if not math.isfinite(beta) or not 0 <= beta < 1 or not math.isfinite(eps) or eps <= 0:
                raise ValueError("RMSProp requires 0 <= beta_2 < 1 and finite eps > 0")

    def check_output(self):
        path = os.path.join("results", self.database_name, "runs.csv")
        if os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames != self.run_columns:
                    raise ValueError("Existing runs.csv has incompatible columns; use a new experiment name")
                for row in reader:
                    if None in row or any(v is None for v in row.values()):
                        raise ValueError("Existing runs.csv contains malformed rows")
                    if row["run"] == str(self.run):
                        if self.resume:
                            expected = {k: str(self.args_for_run.get(k, '')) for k in self.run_columns}
                            if row != expected:
                                raise ValueError('Registered run parameters do not match')
                            return
                        raise ValueError(f"Run {self.run} already exists; use --resume or a new experiment name")
        if self.resume:
            raise ValueError('Cannot resume an unregistered run')

    def _write_runs_row(self):
        self.check_output()
        out_dir = os.path.join("results", self.database_name)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "runs.csv")
        keys = self.run_columns
        write_header = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(keys)
            writer.writerow([self.args_for_run.get(k, "") for k in keys])

    def get_int_param(self, param):
        if param not in self.args_for_run:
            raise ValueError(f"Param {param} does not exist")
        return _parse_int_prefix(self.args_for_run[param])

    def get_float_param(self, param):
        if param not in self.args_for_run:
            raise ValueError(f"Param {param} does not exist")
        return _parse_float_prefix(self.args_for_run[param])

    def get_string_param(self, param):
        if param not in self.args_for_run:
            raise ValueError(f"Param {param} does not exist")
        return str(self.args_for_run[param])

    def get_vector_param(self, key):
        return self.get_string_param(key).split(":")


class CountConfig:
    def __init__(self, argv):
        parser = argparse.ArgumentParser(description="Count experiment parameter combinations")
        parser.add_argument("--config", required=True)
        args = parser.parse_args(argv)
        j = load_config(args.config)
        print(_prod(get_combinations(j)))
