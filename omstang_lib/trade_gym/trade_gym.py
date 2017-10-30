import os
import glob

from random import random
import numpy as np
import pandas as pd
import math

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from collections import Counter

from ..market_data import MarketData
from ..market_sim import TradingSim


class MarketEnv(gym.Env):
    """This gym implements a simple trading environment for reinforcement learning.
    The gym provides daily observations based on real market data pulled
    from Quandl on, by default, the SPY etf. An episode is defined as 252
    contiguous days sampled from the overall dataset. Each day is one
    'step' within the gym and for each step, the algo has a choice:
    SHORT (0)
    FLAT (1)
    LONG (2)
    If you trade, you will be charged, by default, 10 BPS of the size of
    your trade. Thus, going from short to long costs twice as much as
    going from short to/from flat. Not trading also has a default cost of
    1 BPS per step. Nobody said it would be easy!
    At the beginning of your episode, you are allocated 1 unit of
    cash. This is your starting Net Asset Value (NAV). If your NAV drops
    to 0, your episode is over and you lose. If your NAV hits 2.0, then
    you win.
    The trading env will track a buy-and-hold strategy which will act as
    the benchmark for the game.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, datadir="", filename="", days=252, scale=True):
        self.actions = [
            "BUY",
            "WAIT",
            "SELL",
        ]

        self.src = MarketData(days=days, datadir=datadir, filename=filename, scale=scale)
        self.sim = TradingSim(steps=days, trading_cost_bps=1e-3, time_cost_bps=1e-4)

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(self.src.min_values, self.src.max_values)
        self.reset()

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        observation, done = self.src._step()

        yret = observation[0]

        reward, info = self.sim._step(action, yret)

        # info = { 'pnl': daypnl, 'nav':self.nav, 'costs':costs }

        return observation, reward, done, info

    def _reset(self):
        self.src.reset()
        self.sim.reset()
        return self.src._step()[0]

    def _render(self, mode='human', close=False):
        # ... TODO
        pass

    # some convenience functions:

    def run_strategy(self, strategy, return_df=True):
        """run provided strategy, returns dataframe with all steps"""
        observation = self.reset()
        done = False
        while not done:
            action = strategy(observation, self)  # call strategy
            observation, reward, done, info = self.step(action)

        return self.sim.to_df() if return_df else None

    def run_strategys(self, strategy, episodes=1, write_log=True, return_df=True):
        """ run provided strategy the specified # of times, possibly
            writing a log and possibly returning a dataframe summarizing activity.

            Note that writing the log is expensive and returning the df is moreso.
            For training purposes, you might not want to set both.
        """
        alldf = None

        for i in range(episodes):
            df = self.run_strategy(strategy, return_df=return_df)
            if return_df:
                alldf = df if alldf is None else pd.concat([alldf, df], axis=0)

        return alldf

    def info(self):
        self.src.env_info()
