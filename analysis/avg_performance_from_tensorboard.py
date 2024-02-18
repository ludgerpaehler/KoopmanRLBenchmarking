import numpy as np
import os

from analysis.utils import create_folder
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


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

    path = "./runs"
    file_names = [
        "LinearSystem-v0__koopman_83_eval__1__1708261673",
        "LinearSystem-v0__koopman_103_eval__1__1708261676",
        "LinearSystem-v0__koopman_123_eval__1__1708261679",
        "LinearSystem-v0__koopman_143_eval__1__1708261682",
        "LinearSystem-v0__koopman_163_eval__1__1708261685"
    ]

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