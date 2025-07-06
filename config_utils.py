import yaml
import argparse
import os
from typing import Any, Dict, List, Union

# Copy-paste from dnnlib.util to avoid numpy dependency in dnnlib.util
class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

def parse_type(type_str: str) -> Any:
    """Convert string representation of type to actual type."""
    if type_str == 'str':
        return str
    elif type_str == 'int':
        return int
    elif type_str == 'float':
        return float
    elif type_str == 'bool':
        return bool
    elif type_str.startswith('List['):
        inner_type = parse_type(type_str[5:-1])
        return lambda x: inner_type(x)
    else:
        raise ValueError(f"Unknown type: {type_str}")

def validate_and_convert(config: Dict[str, Any], schema: Dict[str, str]) -> EasyDict:
    """
    Validate and convert config values based on the provided schema.
    """
    validated_config = {}
    for key, value in config.items():
        if key in schema:
            expected_type = parse_type(schema[key])
            if value!=None:
                try:
                    if expected_type == bool:
                        # Special handling for boolean values
                        if isinstance(value, str):
                            value = value.lower() in ('true', 'yes', '1', 'on')
                        else:
                            value = bool(value)
                    elif schema[key].startswith('List['):
                        # Special handling for lists
                        if isinstance(value, str):
                            value = [expected_type(v.strip()) for v in value.split(',')]
                        elif isinstance(value, list):
                            value = [expected_type(v) for v in value]
                    else:
                        value = expected_type(value)
                    validated_config[key] = value
                except ValueError:
                    raise ValueError(f"Invalid type for {key}. Expected {schema[key]}, got {type(value).__name__}")
            else:
                validated_config[key] = None
        else:
            # Keep unspecified keys as is
            validated_config[key] = value
    return EasyDict(validated_config)

def load_config(args=None):
    """
    Load and merge configuration from multiple sources:
    1. Default configuration from 'config/config.yaml'
    2. Provided arguments (usually from command-line)
    
    Args:
        args: An argparse.Namespace object containing the parsed arguments.
              If None, will parse arguments from command line.
    
    Returns:
        EasyDict: A dictionary-like object containing the merged configuration
    """
    # Load default config and schema
    with open('config/config.yaml', 'r') as f:
        yaml_data = yaml.safe_load(f)
        schema = yaml_data['schema']
        config = yaml_data['config']
    
    # If args were not provided, parse them from command line
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--outdir', type=str, required=True)
        args, unknown = parser.parse_known_args()
        config['outdir'] = args.outdir
    else:
        # If args were provided, we assume they're already parsed
        unknown = []
        for arg in args:
            if arg != 'outdir':
                unknown.append(f'--{arg}={args[arg]}')    
    # Update config with provided arguments
    for arg in unknown:
        if arg.startswith('--'):
            param, value = arg.strip('--').split('=')
            config[param] = value

    # Also set dataset_path based on 'dataset' and 'data_subset' arguments if data_subset is not set
    if 'dataset_path' not in config:
        config['dataset_path'] = os.path.join("data", f"{config['dataset']}_{config['data_subset']}")

    # Validate and convert config values
    return validate_and_convert(config, schema)

def get_job_name_params(cfg):
    """
    Get the parameters to include in the job name
    """
    key_shorthands = {
            'operator_name': 'op',
            'noise_sigma': 'ns',
            'solver': 'slv',
            'num_steps': 'stp',
            'S_churn': 'ch',
            'total_images': 'img',
            'cond_scaling': 'cs',
            'image_base_covariance': 'ibc',
            'pca_component_count': 'pca',
            'denoiser_mean_error_threshold': 'det',
            'use_analytical_score_time_update': 'atu',
            'project_to_diagonal': 'ptd',
            'space_step_update_threshold': 'sut',
            'space_step_update_lower_threshold': 'slt',
            'pigdm_posthoc_scaling': 'pps',
            'clip_x0_mean': 'cx0m',
            'conditioning_mechanism': 'cm',
            'denoiser_mean_error_threshold': 'det',
            'use_rtol_func': 'urf',
            'solver_type': 'st'
    }
    # No need to add the measurement operators etc., since they are included in the experiment name
    keys_to_include = []
    if cfg['conditioning_mechanism'] == 'dps':
        keys_to_include.extend(['operator_name', 'noise_sigma', 'solver', 'num_steps', 'S_churn', 'total_images', 'cond_scaling'])
        return {k: v for k, v in cfg.items() if k != 'outdir' and k in keys_to_include}
    elif cfg['conditioning_mechanism'] == 'pigdm' or cfg['conditioning_mechanism'] == 'tmpd' or cfg['conditioning_mechanism'] == 'peng_convert' or cfg['conditioning_mechanism'] == 'peng_analytic' or cfg['conditioning_mechanism'] == 'ddnm' or cfg['conditioning_mechanism'] == 'diffpir':
        keys_to_include.extend(['operator_name', 'noise_sigma', 'solver', 'num_steps', 'S_churn', 'total_images', 'cond_scaling', 'pigdm_posthoc_scaling', 'clip_x0_mean'])
        # Transform the dictionary to use shorthands if the shorthands are defined
        original_dict = {k: v for k, v in cfg.items() if k != 'outdir' and k in keys_to_include}
        return {key_shorthands.get(k, k): v for k, v in original_dict.items()}
    elif cfg['conditioning_mechanism'] == 'online_covariance':
        keys_to_include.extend(['operator_name', 'noise_sigma', 'solver', 'num_steps', 'S_churn', 'total_images', 'cond_scaling', 'image_base_covariance', 'pca_component_count', 'denoiser_mean_error_threshold', 'use_analytical_score_time_update', 'project_to_diagonal', 'space_step_update_threshold', 'space_step_update_lower_threshold', 'pigdm_posthoc_scaling', 'clip_x0_mean', 'max_rtol', 'use_analytic_var_at_end', 'denoiser_mean_error_threshold', 'use_rtol_func', 'solver_type'])
        # Get the original dictionary
        original_dict = {k: v for k, v in cfg.items() if k != 'outdir' and k in keys_to_include}
        # Transform the dictionary to use shorthands
        return {key_shorthands.get(k, k): v for k, v in original_dict.items()}
    else:
        raise ValueError(f"Unknown conditioning mechanism: {cfg['conditioning_mechanism']}")
