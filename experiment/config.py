import csv
import json
import os
import re
import sys

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
    return int(_INT_PREFIX.match(value.strip()).group(0))


def _parse_float_prefix(value):
    # std::stof semantics: parse the leading float prefix
    return float(_FLOAT_PREFIX.match(value.strip()).group(0))


class ExperimentJSON:
    def __init__(self, argv):
        self.args = parse_params(argv)
        if "config" not in self.args:
            print("Config file not provided; quiting")
            sys.exit(0)
        file_name = self.args["config"][0]
        if "run" not in self.args:
            print("Run ID not provided")
            sys.exit(1)
        run = int(self.args["run"][0])

        with open(file_name, encoding="utf-8") as f:
            j = json.load(f)
        total_combinations = _prod(get_combinations(j))
        print("TOTAL COMBINATIONS =", total_combinations)

        self.args_for_run = get_args_for_run(j, run)
        self.args_for_run["run"] = str(run)
        for key in sorted(self.args_for_run):
            print(key, self.args_for_run[key])

        self.run = run
        self.database_name = self.args_for_run["name"]
        self._write_runs_row()

    def _write_runs_row(self):
        out_dir = os.path.join("results", self.database_name)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "runs.csv")
        keys = sorted(self.args_for_run)
        write_header = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(keys)
            writer.writerow([self.args_for_run[k] for k in keys])

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
        return self.args_for_run[param]

    def get_vector_param(self, key):
        return self.get_string_param(key).split(":")


class CountConfig:
    def __init__(self, argv):
        self.args = parse_params(argv)
        if "config" not in self.args:
            print("Config file not provided; quiting")
            sys.exit(0)
        with open(self.args["config"][0], encoding="utf-8") as f:
            j = json.load(f)
        print(_prod(get_combinations(j)))
