import numpy as np
from collections import deque
import mujoco
from scipy.spatial.transform import Rotation as R
from typing import List


class HistoryBuffer:
    def __init__(self, obs_history_len: int, num_obs_terms: int = 8):
        self.obs_history_len = obs_history_len
        self.num_obs_terms = num_obs_terms
        self.buffers = [deque(maxlen=obs_history_len) for _ in range(num_obs_terms)]

    def update_history(self, current_terms: List[np.ndarray]):
        # append the new data to the end of the each term buffer
        for i, term_data in enumerate(current_terms):
            self.buffers[i].append(term_data.flatten())

    def get_stacked_obs(self) -> np.ndarray:
        term_histories = []
        for buff in self.buffers:
            # handle the case where the buffer is not full
            while len(buff) < self.obs_history_len:
                buff.append(buff[-1])

            # concatenate
            term_histories.append(np.concatenate(list(buff)))

        # concatenate the term histories
        final_obs = np.concatenate(term_histories)
        return final_obs.astype(np.float32)


def get_mujoco_data(
    model: mujoco.MjModel, data: mujoco.MjData, joint_names: List[str]
) -> np.ndarray:
    quat_raw = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat_raw)

    # projected gravity in the base frame
    proj_gravity = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)

    # linear velocity in the base frame
    lin_vel = r.apply(data.qvel[:3], inverse=True).astype(np.double)

    # angular velocity
    ang_vel = data.sensor("angular-velocity").data.copy()

    # get joint positions and joint velocities
    joints_pos = []
    joints_vel = []

    for name in joint_names:
        addr_pos = model.joint(name).qposadr
        addr_vel = model.joint(name).dofadr
        joints_pos.append(data.qpos[addr_pos])
        joints_vel.append(data.qvel[addr_vel])

    joints_pos = np.array(joints_pos)
    joints_vel = np.array(joints_vel)

    return (
        proj_gravity.flatten(),
        lin_vel.flatten(),
        ang_vel.flatten(),
        joints_pos.flatten(),
        joints_vel.flatten(),
    )
