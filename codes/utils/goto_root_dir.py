"""This file should be imported at the beginning of every script run from console.

It will check the path and change the working directory to the root of the project."""
from pathlib import Path
import os

def run():
    path = Path.cwd()
    # while len(path.name) and path.name != 'cognitive_dynamics':
    # while len(path.name) and path.name != 'RT_RNN':
    while len(path.name) and path.name != 'Codes':
        path = path.parent

    if len(path.name):
        os.chdir(path)
    else:
        raise ValueError('Cannot find the root directory of the project.')

run()