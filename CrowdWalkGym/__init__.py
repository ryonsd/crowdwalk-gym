from gym.envs.registration import register

register(
    id="two-routes-v0",
    entry_point="CrowdWalkGym.envs.two_routes:TwoRoutesEnv"
)

register(
    id="moji-v0",
    entry_point="CrowdWalkGym.envs.moji:MojiEnv"
)
