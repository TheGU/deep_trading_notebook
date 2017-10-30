import numpy as np
import pandas as pd


def _sharpe(returns, freq=252):
    """
    Given a set of returns, calculates naive (rfr=0) sharpe
    """
    return (np.sqrt(freq) * np.mean(returns))/np.std(returns)


def _prices2returns(prices):
    px = pd.DataFrame(prices)
    nl = px.shift().fillna(0)
    R = ((px - nl)/nl).fillna(0).replace([np.inf, -np.inf], np.nan).dropna()
    R = np.append( R[0].values, 0)
    return R


class TradingSim(object):
    """ Implements core trading simulator for single-instrument univ """

    def __init__(self, steps, trading_cost_bps=1e-3, time_cost_bps=1e-4):
        # invariant for object life
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps
        self.steps = steps

        # change every step
        self.step = 0
        self.actions = np.zeros(self.steps)
        self.navs = np.ones(self.steps)
        self.mkt_nav = np.ones(self.steps)
        self.strategy_retrns = np.ones(self.steps)
        self.position = np.zeros(self.steps)
        self.costs = np.zeros(self.steps)
        self.trades = np.zeros(self.steps)
        self.mkt_retrns = np.zeros(self.steps)

    def reset(self):
        self.step = 0
        self.actions.fill(0)
        self.navs.fill(1)
        self.mkt_nav.fill(1)
        self.strategy_retrns.fill(0)
        self.position.fill(0)
        self.costs.fill(0)
        self.trades.fill(0)
        self.mkt_retrns.fill(0)

    def _step(self, action, retrn):
        """
        Given an action and return for prior period, calculates costs, navs,
        etc and returns the reward and a  summary of the day's activity.
        """

        base_position = 0.0 if self.step == 0 else self.position[self.step - 1]
        agent_nav = 1.0 if self.step == 0 else self.navs[self.step - 1]
        mkt_nav = 1.0 if self.step == 0 else self.mkt_nav[self.step - 1]

        self.mkt_retrns[self.step] = retrn
        self.actions[self.step] = action

        self.position[self.step] = action - 1
        self.trades[self.step] = self.position[self.step] - base_position

        trade_costs_pct = abs(self.trades[self.step]) * self.trading_cost_bps
        self.costs[self.step] = trade_costs_pct + self.time_cost_bps
        reward = ((base_position * retrn) - self.costs[self.step])
        self.strategy_retrns[self.step] = reward

        if self.step != 0:
            self.navs[self.step] = agent_nav * (1 + self.strategy_retrns[self.step - 1])
            self.mkt_nav[self.step] = mkt_nav * (1 + self.mkt_retrns[self.step - 1])

        info = {
            'reward': reward,
            'nav': self.navs[self.step],
            'base': self.mkt_nav[self.step],
            'position': self.position[self.step],
            'costs': self.costs[self.step]}

        self.step += 1
        return reward, info

    def to_df(self):
        """returns internal state in new dataframe """
        cols = ['action', 'navs', 'mkt_nav', 'mkt_return', 'sim_return',
                'position', 'costs', 'trade']
        rets = _prices2returns(self.navs)
        # pdb.set_trace()
        df = pd.DataFrame({'action': self.actions,  # today's action (from agent)
                           'navs': self.navs,  # BOD Net Asset Value (NAV)
                           'mkt_nav': self.mkt_nav,
                           'mkt_return': self.mkt_retrns,
                           'sim_return': self.strategy_retrns,
                           'position': self.position,  # EOD position
                           'costs': self.costs,  # eod costs
                           'trade': self.trades},  # eod trade
                          columns=cols)
        return df
