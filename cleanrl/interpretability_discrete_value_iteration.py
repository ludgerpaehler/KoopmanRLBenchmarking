import argparse
import gym
import numpy as np
import os
import random
import time
import torch
torch.set_default_dtype(torch.float64)

from custom_envs import *
from distutils.util import strtobool
from koopman_tensor.torch_tensor import KoopmanTensor
from koopman_tensor.utils import load_tensor
from cleanrl.discrete_value_iteration import DiscreteKoopmanValueIterationPolicy
from torch.utils.tensorboard import SummaryWriter

delta = torch.finfo(torch.float64).eps # 2.220446049250313e-16


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment (default: 1)")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False` (default: True)")
    
    # Koopman-specific arguments
    parser.add_argument("--env-id", type=str, default="FluidFlow-v0",
        help="the id of the environment (default: FluidFlow-v0)")
    parser.add_argument("--total-timesteps", type=int, default=10_000,
        help="total timesteps of the experiments (default: 10_000)")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma (default: 0.99)")
    parser.add_argument("--batch-size", type=int, default=2**14,
        help="the batch size of sample from the reply memory (default: 2^14 = 16,384)")
    parser.add_argument("--lr", type=float, default=1e-3,
        help="the learning rate of the Q network network optimizer (default: 0.001)")
    parser.add_argument("--alpha", type=float, default=1.0,
        help="entropy regularization coefficient (default: 1.0)")
    parser.add_argument("--num-actions", type=int, default=101,
        help="number of actions that the policy can pick from (default: 101)")
    parser.add_argument("--num-training-epochs", type=int, default=150,
        help="number of epochs that the model should be trained over (default: 150)")
    parser.add_argument("--batch-scale", type=int, default=1,
        help="increase batch size by this multiple for computing bellman error (default: 1)")
    parser.add_argument("--model-name", type=str, default=None,
        help="Name the saved model is to be stored into. Default path: saved_models/env_id/model_name")
    
    # Interpretability-specific argument(s)
    parser.add_argument("--value-fn-weights", type=list,
        default=[[-300.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [200.0], [0.0], [150.0]],
        help="Array of array of the weights for the individual terms defining the value function. Presumes an order 2 monomial space.")

    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Run on the CPU by default
    device = torch.device("cpu")

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, False, run_name)])

    koopman_tensor = load_tensor(args.env_id, "path_based_tensor")

    try:
        dt = envs.envs[0].dt
    except:
        dt = None

    # Construct set of all possible actions
    all_actions = torch.from_numpy(np.linspace(
        start=envs.single_action_space.low,
        stop=envs.single_action_space.high,
        num=args.num_actions
    )).T

    value_iteration_policy = DiscreteKoopmanValueIterationPolicy(
        env_id=args.env_id,
        gamma=args.gamma,
        alpha=args.alpha,
        dynamics_model=koopman_tensor,
        all_actions=all_actions,
        cost=envs.envs[0].vectorized_cost_fn,
        save_data_path="./saved_models",
        use_ols=True,
        learning_rate=args.lr,
        dt=dt,
        seed=args.seed,
        load_model=True,
        initial_value_function_weights=torch.tensor(args.value_fn_weights),
        args=args
    )

    envs.single_observation_space.dtype = np.float64
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        actions = value_iteration_policy.get_action(torch.Tensor(obs).to(device))
        actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # Write data
        if global_step % 100 == 0:
            sps = int(global_step / (time.time() - start_time))
            print("Steps per second (SPS):", sps)
            writer.add_scalar("charts/SPS", sps, global_step)

    envs.close()
    writer.close()