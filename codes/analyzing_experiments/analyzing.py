import os

import joblib
import numpy as np

from utils import goto_root_dir
import pathlib
from pathlib import Path
from path_settings import *
import pandas as pd
from agents import Agent
from datasets import Dataset
from contextlib import contextmanager
import sys
from utils.logger import PrinterLogger

@contextmanager
def set_posix_windows():
    """Temporarily change the posix path to windows path for loading the models."""
    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup


def pd_full_print_context():
    """Print the full dataframe in the console."""
    return pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr', False, 'display.max_colwidth', None)


def get_config_from_path(model_path):
    """Get the config from the model path."""
    model_path = Path(model_path)
    if MODEL_SAVE_PATH.name not in model_path.parts:
        model_path = MODEL_SAVE_PATH / model_path
    with set_posix_windows():
        config = joblib.load(model_path / 'config.pkl')
        config['model_path'] = Path(str(config['model_path']))
    return config


def get_model_from_config(config, best_device=False):
    """Get the model from the config."""
    if best_device and 'rnn_type' in config:
        config['device'] = 'cuda'
    ag = Agent(config['agent_type'], config=config)
    ag.load(config['model_path'], strict=False) # for the dummy variable
    return ag


def transform_model_format(ob, source='', target=''):
    """
    Transform the model format from one format to another.
    row->path; row->config
    path<->config
    config->agent
    Args:
        ob: The model object.
        source: 'row', 'path', 'config'
        target: 'path', 'config', 'agent'

    Returns:
        The transformed model object.
    """
    assert source in ['row', 'path', 'config']
    assert target in ['path', 'config', 'agent']
    if source == 'row':
        ob = ob['config']
        source = 'config'
    # now source is either 'path' or 'config'
    if target == 'path':
        if source == 'path':
            return Path(ob)
        else: # source == 'config'
            return Path(ob['model_path'])
    # now target is either 'config' or 'agent'
    if source == 'path':
        ob = get_config_from_path(ob)
    # now source is 'config'
    if target == 'config':
        return ob
    else: # target is 'agent'
        return get_model_from_config(ob)


def get_model_path_in_exp(exp_folder, name_filter=''):
    """Obtain the path of all models in the training experiment folder.

    Args:
        exp_folder (str): The name of the experiment folder.
        name_filter (str, optional): The filter to select the model. Defaults to ''.

    Returns:
        list: The list of paths of the models.
    """
    path = MODEL_SAVE_PATH / Path(exp_folder)
    model_paths = []
    for p in path.rglob("*"): # recursively search all subfolders
        if p.name == 'config.pkl':
            model_paths.append(p.parent)
    if len(model_paths)==0:
        print('No models found at ',path)
        raise ValueError()
    return model_paths


def get_model_test_scores(row, behav_dt):
    """Get the model test scores from the row of a dataframe. (Currently not used)

    Args:
        row (pd.Series): The row of the dataframe.
        behav_dt (Dataset): The loaded dataset.

    Returns:
        scores: The scores of the model on the test set.
    """
    config = transform_model_format(row, source='row', target='config')
    ag = transform_model_format(config, source='config', target='agent')
    # behav_dt = Dataset(config['dataset'], behav_data_spec=config['behav_data_spec'])
    behav_dt = behav_dt.behav_to(config)
    data = behav_dt.get_behav_data(config['test_index'], config)
    test_model_pass = ag.forward(data)
    return test_model_pass['output'].detach().cpu().numpy()


def insert_model_test_scores_in_df(df):
    """Insert the model test scores into the dataframe. (Currently not used)

    Args:
        df (pd.DataFrame): The dataframe containing the model paths.

    Returns:
        pd.DataFrame: The dataframe with the test scores.
    """
    config = df.iloc[0]['config']
    behav_dt = Dataset(config['dataset'], behav_data_spec=config['behav_data_spec'])
    df['test_scores'] = df.apply(lambda x: get_model_test_scores(x, behav_dt), axis=1)
    return df


def combine_test_scores(df, group_by_keys):
    """Combine the test scores from the inner folds. (Currently not used)
    TODO
    """
    df = df.groupby(group_by_keys, as_index=False).agg({'test_scores': lambda x: list(x.mean())})
    df['test_scores'] = df['test_scores'].apply(np.array)




if __name__ == '__main__':
    pass
    # find_best_models_for_exp(exp_folder, 'PRLCog', additional_rnn_keys={'model_identifier_keys': ['input_dim']})
    # model_paths = get_model_path_in_exp(exp_folder)
    # [print(i, m) for i,m in enumerate(model_paths)]
