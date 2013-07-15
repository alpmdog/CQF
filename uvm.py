import pandas
from pandas import Series
from numpy import interp
#import numpy
#from numpy import identity, diagonal
#from math import sqrt, exp, ceil
#import random
import logging
from stats import *
#import optparse
from scipy.optimize import fsolve, minimize, fmin
from collections import *

logging.basicConfig(format='[%(levelname)s]%(asctime)s:%(message)s',
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)

modelparams = namedtuple('modelparams', ['r', 'spot', 'lowvol', 'highvol'])
security = namedtuple('security', ['option_type', 'strike',
                                  'maturity', 'quantity'])


class Security(object):

    def __init__(self, option_type, strike, maturity, quantity):
        self.option_type = option_type
        self.strike = strike
        self.maturity = maturity
        self.quantity = quantity


class Portfolio(object):

    def __init__(self):
        self._longest_maturity = None
        self._highest_strike = None
        self.securities = []

    def add(self, security):
        self.securities.append(security)
        if self.longest_maturity is None:
            self.longest_maturity = security.maturity
        elif security.maturity > self.longest_maturity:
            self.longest_maturity = security.maturity
        if self.highest_strike is None:
            self.highest_strike = security.strike
        elif security.strike > self.highest_strike:
            self.highest_strike = security.strike

    def __iter__(self):
        for sec in self.securities:
            yield sec

    def __getitem__(self, x):
        return self.securities[x]

    @property
    def highest_strike(self):
        return self._highest_strike

    @highest_strike.setter
    def highest_strike(self, value):
        self._highest_strike = value

    @property
    def longest_maturity(self):
        return self._longest_maturity

    @longest_maturity.setter
    def longest_maturity(self, value):
        self._longest_maturity = value

    def size(self):
        return len(self.securities)


def heaviside(s, k):
    if s > k:
        return 1.0
    return 0.0


def payoff(security, strike):
    if security.option_type == 'call':
        return max(strike - security.strike, 0) * security.quantity
    elif security.option_type == 'put':
        return max(security.strike - strike, 0) * security.quantity
    elif security.option_type == 'digital call':
        return heaviside(strike, security.strike) * security.quantity
    elif security.option_type == 'digital put':
        return heaviside(-strike, -security.strike) * security.quantity


def get_value_bylevel(results, level):
    prices = results.index.values.tolist()
    values = results.iloc[:, 0].values.tolist()
    return interp(level, prices, values)


class BSPricer(object):
    def __init__(self, modelparams):
        self.params = modelparams

    def mid_vol(self):
        return (self.params.highvol + self.params.lowvol) / 2.

    def hedging_cost(self, portfolio, spread):
        cost = 0
        for idx, sec in enumerate(portfolio):
            if sec.option_type in ['call', 'put']:
                price = abs(self.price_vanilla(sec))
                bid = price - spread / 2.
                ask = price + spread / 2.
                if sec.quantity > 0:
                    cost += ask * sec.quantity
                else:
                    cost += bid * sec.quantity
        return cost

    def price_vanilla(self, security):
        t = security.maturity
        s = float(self.params.spot)
        r = float(self.params.r)
        k = float(security.strike)
        sigma = self.mid_vol()
        quantity = security.quantity
        position = 1
        if quantity < 0:
            position = -1
        d1 = (log(s / k) + (r * 0.5 * sigma * sigma) * t) / (sigma * sqrt(t))
        d2 = d1 - sigma * sqrt(t)
        return (s * normcdf(d1) - exp(-r * t) * k * normcdf(d2)) * position


class ExplicitFDMSolver(object):

    def __init__(self, NAS, modelparams, worst=True):
        self.NAS = NAS
        self.params = modelparams
        self.S = []
        self.payoff = [0] * NAS
        self.prev_state = [0] * NAS
        self.current_state = [0] * (NAS)
        self.worst_case = worst

    def get_results(self):
        option_value = Series(self.current_state, index=self.S)
        payoff = Series(self.payoff, index=self.S)
        return pandas.concat([option_value, payoff], axis=1)

    def delta(self, prev, idx, ds):
        return (prev[idx + 1] - prev[idx - 1]) / (2 * ds)

    def gamma(self, prev, idx, ds):
        return (prev[idx + 1] - (2 * prev[idx])
                + prev[idx - 1]) / (ds * ds)

    def theta(self, prev, idx, delta, gamma, vol, strike):
        r = self.params.r
        return ((r * prev[idx]) -
                (0.5 * vol * vol * strike * strike * gamma)
                - (r * delta * strike))

    def calc_option_values(self, NTS, ds, dt, payoffs):
        prev = payoffs
        current = [0] * len(prev)
        for k in xrange(0, NTS):
            for j in xrange(1, self.NAS - 1):
                delta = self.delta(prev, j, ds)
                gamma = self.gamma(prev, j, ds)
                vol = 0.
                if self.worst_case:
                    if gamma > 0:
                        vol = self.params.lowvol
                    else:
                        vol = self.params.highvol
                else:
                    if gamma > 0:
                        vol = self.params.highvol
                    else:
                        vol = self.params.lowvol
                theta = self.theta(prev, j, delta, gamma, vol, self.S[j])
                current[j] = prev[j] - theta * dt
            current[0] = (prev[0] * (1 - self.params.r * dt))
            last = self.NAS - 1
            current[last] = (2 * current[last - 1] - current[last - 2])
            prev = current
        return current

    def solve(self, portfolio, solve_separately=False):
        dt = 0.9 / (self.NAS * self.NAS *
                    self.params.highvol * self.params.highvol)
        maturity = portfolio.longest_maturity
        strike = portfolio.highest_strike
        NTS = int(maturity / dt) + 1
        dt = float(maturity) / NTS
        ds = 2 * float(strike) / self.NAS

        for i in xrange(0, self.NAS):
            self.S.append(i * ds)

        if solve_separately:
            for idx, sec in enumerate(portfolio.securities):
                for n in xrange(0, self.NAS):
                    self.payoff[n] = payoff(sec, self.S[n])
                sec_results = self.calc_option_values(NTS, ds, dt, self.payoff)
                self.current_state = [ x + y for x,y in zip(sec_results,self.current_state) ]
        else:
            for idx, sec in enumerate(portfolio.securities):
                for n in xrange(0, self.NAS):
                    #self.prev_state[n] += payoff(sec, self.S[n])
                    self.payoff[n] += payoff(sec, self.S[n])
            self.current_state = self.calc_option_values(NTS, ds,
                                                        dt, self.payoff)


def f_to_optimize(x, portfolio, model):
    portfolio[0].quantity = 1
    portfolio[1].quantity = x[0]
    portfolio[2].quantity = x[1]
    fdmSolver = ExplicitFDMSolver(100, model)
    fdmSolver.solve(portfolio)
    results = fdmSolver.get_results()
    bsPricer = BSPricer(model)
    hedging_cost = bsPricer.hedging_cost(portfolio, 0.1)
    bid = get_value_bylevel(results, 100.) - hedging_cost

    portfolio[0].quantity = -1
    portfolio[1].quantity = -x[0]
    portfolio[2].quantity = -x[1]
    fdmSolver = ExplicitFDMSolver(100, model)
    fdmSolver.solve(portfolio)
    results = fdmSolver.get_results()
    bsPricer = BSPricer(model)
    hedging_cost = bsPricer.hedging_cost(portfolio, 0.1)
    ask = hedging_cost - get_value_bylevel(results, 100.)
    value = ask - bid
    print x[0], x[1], bid, ask, value
    return value


def optimize_portfolio(portfolio, model):
    x0 = [-0.05, 0.05]
    #xx = fsolve(f_to_optimize, x0, args=(portfolio, model))
    xx = fmin(f_to_optimize, x0, args=(portfolio, model))
    return xx
