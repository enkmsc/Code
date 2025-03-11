import warnings
warnings.filterwarnings("ignore")

from mace.cli.run_train import main as mace_run_train_main
import sys
import logging
import os
import yaml

def train_mace(base_path, config_file_name):
    """
    base_path: The shared root directory (e.g., '/Users/emilydai/Downloads/700K')
    config_file_name: The YAML config file name (e.g., 'config-train.yml')
    """
    # 1. Load the original YAML config (relative to base_path or absolute)
    config_file_path = os.path.join(base_path, config_file_name)
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Convert YAML fields to absolute paths
    if 'train_file' in config:
        config['train_file'] = os.path.join(base_path, config['train_file'])
    if 'valid_file' in config:
        config['valid_file'] = os.path.join(base_path, config['valid_file'])
    if 'test_file' in config:
        config['test_file'] = os.path.join(base_path, config['test_file'])
    if 'model_dir' in config:
        config['model_dir'] = os.path.join(base_path, config['model_dir'])
        # If needed, copy the same path to log_dir / checkpoints_dir / results_dir
        config['log_dir'] = config['model_dir']
        config['checkpoints_dir'] = config['model_dir']
        config['results_dir'] = config['model_dir']
    if 'foundation_model' in config:
        config['foundation_model'] = os.path.join(base_path, config['foundation_model'])

    # 3. Write out an updated config file for MACE (with absolute paths)
    updated_config_path = os.path.join(base_path, "config-train-updated.yml")
    with open(updated_config_path, 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # 4. Call the MACE CLI with the new config
    logging.getLogger().handlers.clear()
    sys.argv = ["program", "--config", updated_config_path]
    mace_run_train_main()

if __name__ == "__main__":
    # The one place to change your base path
    base_path = "/Users/emilydai/Downloads/LACTP"

    # The original YAML file
    config_file_name = "config-train.yml"

    train_mace(base_path, config_file_name)
