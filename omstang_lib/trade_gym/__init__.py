from gym.envs.registration import register

register(
    id='trading-v0',
    entry_point='omstang_lib.trade_gym.trade_gym:MarketEnv',
    timestep_limit=1000,
)
