from gym.envs.registration import register

register(
    id='MujocoQuadForce-v0',
    entry_point='mujocoquad_gym.envs:MujocoQuadForceEnv',
)
