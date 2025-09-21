import itertools
from dataclasses import dataclass
from copy import deepcopy
from typing import Tuple


@dataclass(kw_only=True)
class ConfigCombinations:
    configs: list[dict]
    num_iterations: list[int]
    reduced_population_sizes: list[int]



def _dict_product(d: dict) -> list[dict]:
    if not isinstance(d, dict):
        return [d]

    keys = list(d.keys())
    values_lists = []

    for k in keys:
        v = d[k]
        if isinstance(v, dict):
            values_lists.append(_dict_product(v))
        elif isinstance(v, list):
            values_lists.append(v)
        else:
            values_lists.append([v])

    combinations = []
    for prod in itertools.product(*values_lists):
        combo = {}
        for key, val in zip(keys, prod):
            combo[key] = val
        combinations.append(combo)

    return combinations


def construct_configs(base: dict, combinations: dict) -> ConfigCombinations:
    required_fields = ['configs', 'num_iterations', 'reduced_population_sizes']
    if not all([k in combinations for k in required_fields]):
        raise ValueError(f'All keys of {required_fields} must be in optional config.')
    all_configs = []

    merged_opts = []
    for config in combinations['configs']:
        expanded_options = _dict_product(config)
        for option in expanded_options:
            new_config = deepcopy(base)
            def merge(d1, d2):
                for k, v in d2.items():
                    if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                        merge(d1[k], v)
                    else:
                        d1[k] = v
                        merged_opts.append(f'{k}={v}')
            merge(new_config, option)
            all_configs.append(new_config)

    return ConfigCombinations(
        configs=all_configs,
        num_iterations=combinations['num_iterations'],
        reduced_population_sizes=combinations['reduced_population_sizes']
    )
