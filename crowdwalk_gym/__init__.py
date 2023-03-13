from gym.envs.registration import register

register(
    id="two-routes-v0",
    entry_point="crowdwalk_gym.envs.two_routes:TwoRoutesEnv"
)

register(
    id="moji-v0",
    entry_point="crowdwalk_gym.envs.moji:MojiEnv"
)

register(
    id="moji-v1",
    entry_point="crowdwalk_gym.envs.moji:MojiSmallEnv"
)
