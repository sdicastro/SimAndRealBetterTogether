import os
import numpy as np
from arguments import get_args
from mpi4py import MPI
import random
import torch
from rl_modules.ddpg_agent import ddpg_agent

"""
The DDPG and HER code are based on a repository "https://github.com/TianhongDai/hindsight-experience-replay"
We added the sim2real aspects of the code.

Train the agent, the MPI part code is copy from openai 
baselines(https://github.com/openai/baselines/blob/master/baselines/her)
"""


def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0], 'max_timesteps': 50}
    return params


def launch(args):
    # create the environment
    if args.env_name == 'FetchPush':
        from environments.FetchPush import FetchPushEnv, real_xml_file, sim_xml_file
        env_real = FetchPushEnv(reward_type='sparse', model_xml_path=real_xml_file)
        env_sim = FetchPushEnv(reward_type='sparse', model_xml_path=sim_xml_file)
        envs = (env_real, env_sim)
    else:
        print("Invalid env_name. Implement a Gym environment with separate real and sim xml files")

    # set random seeds
    seed = np.random.randint(0, 1000)
    env_real.seed(seed + MPI.COMM_WORLD.Get_rank())
    env_sim.seed(seed + MPI.COMM_WORLD.Get_rank())
    random.seed(seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(seed + MPI.COMM_WORLD.Get_rank())

    # get the environment parameters
    env_params = get_env_params(env_real)
    env_params['batch_size'] = args.batch_size
    # create the ddpg agent to interact with the environment 
    ddpg_trainer = ddpg_agent(args, envs, env_params)
    ddpg_trainer.learn()


if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    launch(args)
