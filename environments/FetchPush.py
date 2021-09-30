import os
from gym import utils
from gym.envs.robotics import fetch_env
import numpy as np


# Definition of the action in Fetch:
# The first 3 dimensions of the action space are an offset in Cartesian space from the current end effector position.
# They specify the desired relative gripper position at the next timestep.
# The 4th dimension is the state of the parallel gripper in joint space. It specifies the desired distance between
# the 2 fingers which are position controlled.


# Ensure we get the path separator correct on windows
real_xml_file = os.path.join(os.path.dirname(__file__), "assets", 'fetch', 'push_real.xml')
sim_xml_file = os.path.join(os.path.dirname(__file__), "assets", 'fetch', 'push_sim.xml')


class FetchPushEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', model_xml_path=real_xml_file):
        self.name = f"FetchPush{reward_type.capitalize()}"
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, model_xml_path, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal and distance between gripper and object
        assert achieved_goal.shape == goal.shape

        d_object_goal = np.linalg.norm(achieved_goal - goal, axis=-1)
        if self.reward_type == 'sparse':
            return -(d_object_goal > self.distance_threshold).astype(np.float32)
        else:
            # We get grip_pos and object_pos from info if this key exists.
            # else we get the default value from the simulator
            grip_pos = info.get("grip_pos", self.sim.data.get_site_xpos('robot0:grip'))
            grip_pos = grip_pos.copy()
            object_pos = info.get("object_pos", achieved_goal)
            assert grip_pos.shape == object_pos.shape
            d_grip_object = np.linalg.norm(grip_pos - object_pos, axis=-1)
            return -d_object_goal - (0.1 * d_grip_object)
