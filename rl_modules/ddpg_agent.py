import torch
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.replay_buffer import replay_buffer
from rl_modules.models import actor, critic
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler
import gym
import cv2

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root

"""
DDPG with HER (MPI-version)
The DDPG and HER code are based on a repository "https://github.com/TianhongDai/hindsight-experience-replay"
We added the sim2real aspects of the code.
"""


def get_time_str():
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


class ddpg_agent:
    def __init__(self, args, envs, env_params):
        self.args = args
        self.envs = envs
        self.env_params = env_params
        # create the network
        self.actor_network = actor(env_params)
        self.critic_network = critic(env_params)
        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)

        # build up the target network
        self.actor_target_network = actor(env_params)
        self.critic_target_network = critic(env_params)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k,
                                      self.envs[0].compute_reward)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions,
                                    len(envs))
        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)

        num_sim_envs = len(envs) - 1
        # fraction_interaction: [real, sim] - probability of interacting with real/sim environment.
        real_p_probability = 0.1
        uniform_sim_p_prob = (1. - real_p_probability) / num_sim_envs
        self.fraction_interaction = [real_p_probability] + [uniform_sim_p_prob for _ in range(num_sim_envs)]

        # optimize_strategy: [real, sim] - probability of optimizing the model with real/sim environment samples.
        real_beta_probability = 0.5
        uniform_sim_beta_prob = (1. - real_beta_probability) / num_sim_envs
        self.optimize_strategy = [real_beta_probability] + [uniform_sim_beta_prob for _ in range(num_sim_envs)]
        # strategy for collecting real samples
        self.strategy = "sim_dependent"  # "mixed", "sim_dependent", "sim_first"
        self.total_experiment_samples = self.args.n_epochs * self.args.n_cycles * self.args.num_rollouts_per_mpi \
                                        * MPI.COMM_WORLD.Get_size()
        self.fixed_real_samples = self.total_experiment_samples * self.fraction_interaction[0]
        self.only_real_flag = False
        self.render_during_training = False

        # storing data
        env_name = envs[0].unwrapped.spec.id if isinstance(envs[0], gym.wrappers.time_limit.TimeLimit) else envs[0].name
        print("env_name", env_name)
        alg = args.alg
        tst_name = f"tst_s2r_{env_name}_{alg}_{get_time_str()}_strategy-{self.strategy}" \
                   f"_interact-{self.fraction_interaction[0]}_opt-{self.optimize_strategy[0]}"
        model_path = ROOT_DIR + f"/data/{alg}/{env_name}/" + tst_name
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.tb_writer = SummaryWriter(log_dir=model_path)
        # create the dict for store the model
        self.model_path = model_path
        # path for loading existing models:
        self.model_path_to_load = model_path

        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)

    def learn(self):
        """
        train the network
        """
        real_counter = 0  # counts number of collected real episodes
        sim_counter = 0  # counts number of collected sim episodes
        optimize_with_real_counter = 0  # counts number of used real episodes in optimization step
        optimize_with_sim_counter = 0  # counts number of used sim episodes in optimization step
        episode_counter = 0
        cycle_counter = 0
        update_counter = 0
        success_rate_1 = 0.
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
        hitting_success = -1 * np.ones((10,))  # records the first time the success rate passed 10%, 20%, .... 100%

        test_numbers = [1000 * (k + 1) for k in range(
            self.total_experiment_samples // 1000)]  # these numbers will be used to check what is the success rate
        succeess_rate_for_real_samples = -1 * np.ones((len(test_numbers),))
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            for cycle in range(self.args.n_cycles):
                cycle_counter += 1
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                # choosing the environment to collect samples:
                if self.strategy == 'mixed':
                    idx_env = np.random.choice(len(self.fraction_interaction), p=self.fraction_interaction)
                elif self.strategy == 'sim_first':
                    if success_rate_1 >= .7 or self.only_real_flag:
                        self.only_real_flag = True
                        idx_env = 0
                    else:
                        idx_env = 0
                        # Choose one of the sim environments
                        while idx_env == 0:
                            idx_env = np.random.choice(len(self.fraction_interaction), p=self.fraction_interaction)
                elif self.strategy == "sim_dependent":
                    if success_rate_1 >= .7:
                        idx_env = 0
                        self.strategy = "mixed"
                    else:
                        idx_env = 0
                        while idx_env == 0:
                            idx_env = np.random.choice(len(self.fraction_interaction), p=self.fraction_interaction)
                else:
                    print("you need to choose a valid strategy")

                for _ in range(self.args.num_rollouts_per_mpi):
                    episode_counter += 1
                    if idx_env == 0:
                        real_counter += 1
                    else:
                        sim_counter += 1
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    # reset the environment
                    observation = self.envs[idx_env].reset()
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        if MPI.COMM_WORLD.Get_rank() == 0 and self.render_during_training:
                            im = self.envs[idx_env].render(mode='rgb_array')
                            im_rgb = cv2.cvtColor(im[:, :, :3].astype('uint8'), cv2.COLOR_BGR2RGB)
                            cv2.imshow(f"Rendering env {idx_env}", im_rgb)
                            cv2.waitKey(1)
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs, g)
                            pi = self.actor_network(input_tensor)
                            action = self._select_actions(pi)
                        # feed the actions into the environment
                        observation_new, _, _, info = self.envs[idx_env].step(action)
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())
                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                # store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions], num_buf=idx_env)
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
                #  choose environment to optimize with the environment samples:
                optim_idx_env = np.random.choice(len(self.optimize_strategy), p=self.optimize_strategy)
                if self.buffer.current_size[optim_idx_env] > 0:
                    pass
                elif self.strategy == "sim_first":
                    optim_idx_env = idx_env
                else:
                    optim_idx_env = idx_env
                if optim_idx_env == 0:
                    optimize_with_real_counter += 1
                else:
                    optimize_with_sim_counter += 1
                for _ in range(self.args.n_batches):
                    update_counter += 1
                    self._update_network(num_buf=optim_idx_env)

                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)
            # start to do the evaluation
            total_real_count_so_far = MPI.COMM_WORLD.allreduce(real_counter, op=MPI.SUM)
            total_sim_count_so_far = MPI.COMM_WORLD.allreduce(sim_counter, op=MPI.SUM)
            success_rate_0 = self._eval_agent(index_env=0, render=False)
            index_env = np.random.randint(1, len(self.envs))
            success_rate_1 = self._eval_agent(index_env=index_env, render=False)
            if MPI.COMM_WORLD.Get_rank() == 0:
                self.tb_writer.add_scalar(f"success_rate: 0 (X-axis: epoch)", success_rate_0, epoch)
                self.tb_writer.add_scalar(f"success_rate: 1 (X-axis: epoch)", success_rate_1, epoch)
                self.tb_writer.add_scalar(f"success_rate: 0 (X-axis: real episodes)", success_rate_0,
                                          total_real_count_so_far)
                self.tb_writer.add_scalar(f"success_rate: 0 (X-axis: sim episodes)", success_rate_0,
                                          total_sim_count_so_far)
                self.tb_writer.add_scalar("total real episode count", total_real_count_so_far, epoch)
                for (i, qnt) in enumerate(quantiles):
                    if hitting_success[i] == -1:
                        if success_rate_0 >= qnt:
                            hitting_success[i] = total_real_count_so_far
                            self.tb_writer.add_scalar(f"quantile success rate 0", qnt, total_real_count_so_far)
                            self.tb_writer.add_scalar(f"quantile success rate 0 (sim count)", qnt, total_sim_count_so_far)
                            self.tb_writer.add_scalar(f"quantile success rate 0 (X-axis: sim episodes, Y-axis: real episodes)", total_real_count_so_far,
                                                      total_sim_count_so_far)
                for i, n in enumerate(test_numbers):
                    if succeess_rate_for_real_samples[i] == -1:
                        if total_real_count_so_far >= n:
                            succeess_rate_for_real_samples[i] = success_rate_0
                            self.tb_writer.add_scalar(f"success rate 0 by number of real samples", success_rate_0, n)
                print('[{}] epoch is: {}, eval success rate for real is: {:.3f} and for sim is {:.3f}. '
                      'total number of real samples so far: {}'.format(datetime.now(), epoch, success_rate_0,
                                                                       success_rate_1, total_real_count_so_far))
                torch.save({"o_norm_mean": self.o_norm.mean,
                            "o_norm_std": self.o_norm.std,
                            "g_norm_mean": self.g_norm.mean,
                            "g_norm_std": self.g_norm.std,
                            "actor_network": self.actor_network.state_dict(),
                            "critic_network": self.critic_network.state_dict()},
                           self.model_path + '/model.pt')

                print("hitting_success", hitting_success, "total_real_count_so_far", total_real_count_so_far)

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs

    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'],
                                           size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs,
                       'ag': mb_ag,
                       'g': mb_g,
                       'actions': mb_actions,
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self, num_buf):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size, num_buf=num_buf)
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
        # calculate the target Q value function
        with torch.no_grad():
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # the actor loss
        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()

    # do the evaluation
    def _eval_agent(self, index_env=0, render=False):
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.envs[index_env].reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                if MPI.COMM_WORLD.Get_rank() == 0 and render:
                    im = self.envs[index_env].render(mode='rgb_array')
                    im_rgb = cv2.cvtColor(im[:, :, :3].astype('uint8'), cv2.COLOR_BGR2RGB)
                    cv2.imshow(f"Rendering env {index_env}", im_rgb)
                    cv2.waitKey(2)
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = self.envs[index_env].step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()

    def test_agent(self):
        print("model is loaded from: ", self.model_path_to_load)
        model = torch.load(self.model_path_to_load + '/model.pt')
        self.actor_network.load_state_dict(model["actor_network"])
        self.critic_network.load_state_dict(model["critic_network"])
        self.o_norm.mean = model["o_norm_mean"]
        self.o_norm.std = model["o_norm_std"]
        self.g_norm.mean = model["g_norm_mean"]
        self.g_norm.std = model["g_norm_std"]

        success_rate0 = self._eval_agent(index_env=0, render=True)
        success_rate1 = self._eval_agent(index_env=1, render=True)

        print("success_rate0", success_rate0)
        print("success_rate1", success_rate1)
