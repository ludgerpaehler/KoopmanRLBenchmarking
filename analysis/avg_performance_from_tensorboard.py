import numpy as np
import pandas as pd
import os
from distutils.util import strtobool

from analysis.utils import create_folder
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./runs',
        help="Root folder in which to find the folders of the Tensorboard files. Defaults to `/runs`")
    parser.add_argument('--episodic-returns-parquet', type=lambda x: bool(strtobool(x)),
        default=False, nargs="?", const=False,
        help="Return a csv of the episodic returns of the current directory. Uses the folder names as headers.")
    parser.add_argument('--mean-of-means', type=lambda x: bool(strtobool(x)),
        default=False, nargs="?", const=False,
        help="Whether to only compute the mean of all mean episodic returns")
    return parser.parse_args()


def collect_episodic_returns(
    tensorboard_file_directory: str,
    tensorboard_file_name: str
):
    """
    Collects episodic returns and corresponding environment steps from a TensorBoard file.

    Parameters:
    - tensorboard_file_directory (str): The directory containing the TensorBoard file.
    - tensorboard_file_name (str): The name of the TensorBoard file.

    Returns:
    Tuple[List[float], List[int]]: A tuple containing two lists -
    1. episodic_returns (List[float]): List of episodic return values.
    2. steps (List[int]): List of corresponding step values.

    Example:
    >>> directory = '/path/to/tensorboard/files'
    >>> file_name = 'experiment_log'
    >>> returns, steps = collect_episodic_returns(directory, file_name)
    """

    summary_iterator = EventAccumulator(os.path.join(tensorboard_file_directory, tensorboard_file_name)).Reload()

    scalar_name = 'charts/episodic_return'

    steps = [e.step for e in summary_iterator.Scalars(scalar_name)]
    episodic_returns = [e.value for e in summary_iterator.Scalars(scalar_name)]

    return episodic_returns, steps

if __name__ == '__main__':

    args = parse_args()

    path = args.path

    # Get the subfolders, and their respective names
    paths_of_subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    names_of_subfolders = [f.name for f in os.scandir(path) if f.is_dir()]

    file_names = os.listdir(path)

    """
    1. Compute the mean of all mean episodic returns, if given the input argument `mean-of-means`
    """

    if args.mean_of_means:

        # Initialize empty tensor
        means_of_episodic_returns = []

        for i, file_name in enumerate(file_names):
            
            # Collect the individual episodic returns
            episodic_returns, _ = collect_episodic_returns(path, file_name)

            # Compute the mean of the episodic returns
            single_mean = np.mean(episodic_returns)

            # Append the mean to the tensor
            means_of_episodic_returns.append(single_mean)
        
        # Compute the mean of the mean episodic returns
        mean_of_means = np.mean(means_of_episodic_returns)
        
        # Print out the mean of the mean episodic returns across all contained episodic returns
        print("The mean of all mean episodic returns is:", mean_of_means)
    
    elif args.episodic_returns_parquet:

        # Initialize empty pandas dataframe
        df = pd.DataFrame()

        for i, folder_name in enumerate(paths_of_subfolders):

            # Walk the directory
            for _root, _ , _files in os.walk(folder_name):

                # Read in the steps, and episodic returns
                episodic_returns, steps = collect_episodic_returns(_root, _files[0])

                # Insert into the dataframe
                df.insert(i, names_of_subfolders[i], episodic_returns, allow_duplicates=False)

        # Set the index of the dataframe to be the extracted steps
        df.index = steps

        # Store the dataframe
        df.to_parquet(os.path.join(path, 'episodic_returns.parquet.gz'))
    
    else:
        # For each tensorboard file in the pre-defined list above,
        for i, file_name in enumerate(file_names):
            # Collect the episodic returns and environment steps
            # and save the data into numpy files for use in other places
            episodic_returns, steps = collect_episodic_returns(path, file_name)
            create_folder(f'./analysis/{file_name}')
            np.save(f"./analysis/{file_name}/episodic_returns.npy", episodic_returns)
            np.save(f"./analysis/{file_name}/steps.npy", steps)

            # Print information about the data
            print(f"{file_name}'s episodic returns length: {len(episodic_returns)}")
            print(f"{file_name}'s steps length: {len(steps)}")
            print(f"{file_name}'s steps per episode: {steps[0]+1}")

            # Compute mean episodic return
            print(f"{file_name}'s mean episodic return: {np.mean(episodic_returns)}\n")