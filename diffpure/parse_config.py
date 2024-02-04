import os, yaml
import argparse

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


# def parse_args_and_config_cifar(args):
#     with open(os.path.join("diffpure/configs", args.score_sde_config), 'r') as f:
#         config = yaml.safe_load(f)
#     config = dict2namespace(config)
#     return config