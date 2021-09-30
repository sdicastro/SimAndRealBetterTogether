import threading
import numpy as np

"""
the replay buffer here is basically from the openai baselines code
"""


class replay_buffer:
    def __init__(self, env_params, buffer_size, sample_func, num_envs):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T
        # memory management
        # create a list of the buffer to store info. convention: [real_buffer, sim1_buffer, sim2_buffers...]
        # Hence, all memory parameters are also lists that store information for each buffer separately.
        self.num_envs = num_envs  # number of envs to create different buffers for each env.
        self.current_size = [0 for _ in range(self.num_envs)]
        self.n_transitions_stored = [0 for _ in range(self.num_envs)]
        if sample_func is None:
            self.sample_func = self.pure_sample_func
        else:
            self.sample_func = sample_func
        if isinstance(self.env_params['obs'], tuple):
            size = self.env_params['obs']
            empty_array = np.empty([self.size, self.T + 1, size[0], size[1], size[2]], dtype=np.uint8)
        else:
            empty_array = np.empty([self.size, self.T + 1, self.env_params['obs']])
        # list of buffers
        self.buffers = [{'obs': empty_array,
                        'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                         'g': np.empty([self.size, self.T, self.env_params['goal']]),
                         'actions': np.empty([self.size, self.T, self.env_params['action']]),
                         'r': np.empty([self.size, self.T, 1])
                         } for _ in range(self.num_envs)]
        # thread lock
        self.lock = threading.Lock()

        # last index stored:
        self.last_idx_stored = [0 for _ in range(self.num_envs)]
    
    # store the episode
    def store_episode(self, episode_batch, num_buf):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(num_buf=num_buf, inc=batch_size)
            self.last_idx_stored[num_buf] = idxs.copy()
            # store the informations
            self.buffers[num_buf]['obs'][idxs] = mb_obs
            self.buffers[num_buf]['ag'][idxs] = mb_ag
            self.buffers[num_buf]['g'][idxs] = mb_g
            self.buffers[num_buf]['actions'][idxs] = mb_actions
            self.n_transitions_stored[num_buf] += self.T * batch_size
    
    # sample the data from the replay buffer. num_buf indicates from which environment to get the samples.
    def sample(self, batch_size, num_buf):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers[num_buf].keys():
                temp_buffers[key] = self.buffers[num_buf][key][:self.current_size[num_buf]]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, num_buf, inc=None):
        inc = inc or 1
        if self.current_size[num_buf]+inc <= self.size:
            idx = np.arange(self.current_size[num_buf], self.current_size[num_buf]+inc)
        elif self.current_size[num_buf] < self.size:
            overflow = inc - (self.size - self.current_size[num_buf])
            idx_a = np.arange(self.current_size[num_buf], self.size)
            idx_b = np.random.randint(0, self.current_size[num_buf], overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size[num_buf] = min(self.size, self.current_size[num_buf] + inc)
        if inc == 1:
            idx = idx[0]
        return idx

    def pure_sample_func(self, episode_batch, batch_size):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions
