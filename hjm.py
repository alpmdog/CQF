import pandas
from pandas import DataFrame, Series
import numpy
from numpy import identity, diagonal
from math import sqrt, exp, ceil
import random
import logging
import optparse
import os

logging.basicConfig(format='[%(levelname)s]%(asctime)s:%(message)s',
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)


def ZCRate(F, dt, start, end):
    try:
        start_idx = start / dt
        end_idx = (end / dt) + 1
        return F.iloc[start_idx:end_idx, 0].sum() * dt
    except:
        return -1


def libor_rate(F, dt, start, end, tenor):
    try:
        start_idx = start / dt
        end_idx = (end / dt) + 1
        return F.iloc[start_idx:end_idx, tenor].mean()
    except:
        return -1


def discount(F, dt, start, end):
    rate = ZCRate(F, dt, start, end)
    return exp(-rate)


class Product(object):

    def __init__(self):
        self.principal = 1

    def cashflow(self, time):
        return self.principal


class ZeroCouponBond(Product):

    def __init__(self, maturity):
        super(ZeroCouponBond, self).__init__()
        self.maturity = maturity

    def cashflow(self, time):
        if self.maturity == time:
            return self.principal
        else:
            return 0.0

    def get_pay_schedule(self):
        return [self.maturity]

    def __repr__(self):
        return "%sy ZCB" % self.maturity


class Cap(Product):

    def __init__(self, maturity, strike, freq):
        super(Cap, self).__init__()
        self.maturity = maturity
        self.strike = strike
        self.freq = freq
        self.caplets = []
        for i in xrange(1, int(ceil(self.maturity / self.freq)) + 1):
            self.caplets.append(i * self.freq)

    def get_pay_schedule(self):
        return self.caplets

    def cashflow(self, libor, time):
        if time in self.caplets:
            return max(libor - self.strike, 0) * self.principal * self.freq

    def __repr__(self):
        return "%sy %s%% Cap" % (self.maturity, self.strike * 100)


class Floor(Product):

    def __init__(self, maturity, strike, freq):
        super(Floor, self).__init__()
        self.maturity = maturity
        self.strike = strike
        self.freq = freq
        self.floorlets = []
        for i in xrange(1, int(ceil(self.maturity / self.freq)) + 1):
            self.floorlets.append(i * self.freq)

    def get_pay_schedule(self):
        return self.floorlets

    def cashflow(self, libor, time):
        if time in self.floorlets:
            return max(self.strike - libor, 0) * self.principal * self.freq

    def __repr__(self):
        return "%sy %s%% Floor" % (self.maturity, self.strike * 100)


def freq_column(F, freq):
    try:
        return F.columns.get_loc(freq)
    except:
        return -1


def Pricer(F, dt, product):
    pv = 0
    pay_schedule = product.get_pay_schedule()
    for idx, item in enumerate(pay_schedule):
        df = discount(F, dt, 0, item)
        start = 0
        if idx > 0:
            start = pay_schedule[idx - 1]
        if hasattr(product, 'freq'):
            freq_col = freq_column(F, product.freq)
            libor = libor_rate(F, dt, start, item, freq_col)
            pv += df * product.cashflow(libor, item)
        else:
            pv += df * product.cashflow(item)
    return pv


def jacobi(a, tol=1.0e-9):
    '''
        Jacobi method
        #http://w3mentor.com/learn/python/scientific-computation/
        #python-code-for-solving-eigenvalue-problem-by-jacobis-method/
    '''

    def maxElem(a):
        '''
            Find largest off-diag. element a[k,l]
        '''
        n = len(a)
        aMax = 0.0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if abs(a[i, j]) >= aMax:
                    aMax = abs(a[i, j])
                    k = i
                    l = j
        return aMax, k, l

    def rotate(a, p, k, l):
        '''
            Rotate to make a[k,l] = 0
        '''
        n = len(a)
        aDiff = a[l, l] - a[k, k]
        if abs(a[k, l]) < abs(aDiff) * 1.0e-36:
            t = a[k, l] / aDiff
        else:
            phi = aDiff / (2.0 * a[k, l])
            t = 1.0 / (abs(phi) + sqrt(phi ** 2 + 1.0))
            if phi < 0.0:
                t = -t
        c = 1.0 / sqrt(t ** 2 + 1.0)
        s = t * c
        tau = s / (1.0 + c)
        temp = a[k, l]
        a[k, l] = 0.0
        a[k, k] = a[k, k] - t * temp
        a[l, l] = a[l, l] + t * temp
        for i in range(k):      # Case of i < k
            temp = a[i, k]
            a[i, k] = temp - s * (a[i, l] + tau * temp)
            a[i, l] = a[i, l] + s * (temp - tau * a[i, l])
        for i in range(k + 1, l):  # Case of k < i < l
            temp = a[k, i]
            a[k, i] = temp - s * (a[i, l] + tau * a[k, i])
            a[i, l] = a[i, l] + s * (temp - tau * a[i, l])
        for i in range(l + 1, n):  # Case of i > l
            temp = a[k, i]
            a[k, i] = temp - s * (a[l, i] + tau * temp)
            a[l, i] = a[l, i] + s * (temp - tau * a[l, i])
        for i in range(n):      # Update transformation matrix
            temp = p[i, k]
            p[i, k] = temp - s * (p[i, l] + tau * p[i, k])
            p[i, l] = p[i, l] + s * (temp - tau * p[i, l])

    n = len(a)
    maxRot = 5 * (n ** 2)    # Set limit on number of rotations
    p = identity(n) * 1.0    # Initialize transformation matrix
    for i in range(maxRot):  # Jacobi rotation loop
        aMax, k, l = maxElem(a)
        if aMax < tol:
            return diagonal(a), p
        rotate(a, p, k, l)
    print 'Jacobi method did not converge'


def curve_fit(vols_to_fit, degree=3):
    y = vols_to_fit.values
    x = vols_to_fit.index.astype("float64").tolist()
    poly_params = numpy.polyfit(x, y, degree)
    poly = numpy.poly1d(poly_params)
    yPoly = poly(x)
    fitted_vol = Series(yPoly, index=x)
    return fitted_vol, poly


def trapezoid_int(f, tau):
    if tau == 0:
        return 0
    dTau = 0.01
    N = int(tau / dTau)
    dTau = tau / N
    m = 0.5 * f(0)
    for i in xrange(1, N):
        m += f(i * dTau)
    m = m + 0.5 * f(tau)
    m = m * dTau
    m = f(tau) * m
    return m


def discrete_drift(poly1, poly2, poly3, tau):
    mu = []
    for idx, item in enumerate(tau):
        m1 = trapezoid_int(poly1, item)
        m2 = trapezoid_int(poly2, item)
        m3 = trapezoid_int(poly3, item)
        mu.append(m1 + m2 + m3)
    m = Series(mu, index=tau)
    return m


def discrete_drift2(factors, tau):
    mu = []
    for idx, item in enumerate(tau):
        d = 0
        for key in factors:
            factor = factors[key]
            d += trapezoid_int(factor['poly'], item)
        mu.append(d)
    m = Series(mu, index=tau)
    return m


def generate_fwd_curves(initial_fwd_curve, dt, tenor, mu,
                        vol1fit, vol2fit, vol3fit):
    mu = mu * dt
    F = initial_fwd_curve.values
    mu_arry = mu.values
    tau = initial_fwd_curve.columns.astype('float64')
    dTau = numpy.diff(tau.values)
    dTau = numpy.append(dTau, dTau[-1])
    N = int(tenor / dt)
    for i in xrange(1, N + 1):
        X1 = random.normalvariate(0, 1)
        X2 = random.normalvariate(0, 1)
        X3 = random.normalvariate(0, 1)
        dF = numpy.diff(F[i - 1])
        dF = numpy.append(dF, dF[-1])
        row = F[i - 1] + mu_arry + \
              ((vol1fit * X1 + vol2fit * X2 + vol3fit * X3) * sqrt(dt)) \
              + ((dF / dTau) * dt)
        #Set 1e-7 as limit in the array in order
        #to make sure fwds are not negative
        numpy.putmask(row, row < 1e-7, 1e-7)
        F = numpy.vstack((F, row))
    F_df = DataFrame(F, columns=tau)
    return F_df


def generate_fwd_curves2(initial_fwd_curve, dt, tenor, mu, factors):
    mu = mu * dt
    F = initial_fwd_curve.values
    mu_arry = mu.values
    tau = initial_fwd_curve.columns.astype('float64')
    dTau = numpy.diff(tau.values)
    dTau = numpy.append(dTau, dTau[-1])
    N = int(tenor / dt)
    for i in xrange(1, N + 1):
        sigma = 0
        for key in factors:
            factor = factors[key]
            X = random.normalvariate(0, 1)
            sigma += factor['volfit'] * X
        dF = numpy.diff(F[i - 1])
        dF = numpy.append(dF, dF[-1])
        row = F[i - 1] + mu_arry + (sigma * sqrt(dt)) \
              + ((dF / dTau) * dt)
        #Set 1e-7 as limit in the array in order
        #to make sure fwds are not negative
        numpy.putmask(row, row < 1e-7, 1e-7)
        F = numpy.vstack((F, row))
    F_df = DataFrame(F, columns=tau)
    return F_df


def create_product(options):
    maturity = options.maturity
    if options.product == 'zc':
        return ZeroCouponBond(maturity)
    elif options.product == 'cap':
        strike = options.strike
        freq = options.freq
        return Cap(maturity, float(strike) / 100., freq)
    elif options.product == 'floor':
        strike = options.strike
        freq = options.freq
        return Cap(maturity, float(strike) / 100., freq)


def calculate_pca(forwards, no_factors=3):
    fwddiff = forwards.diff()
    fwddiff = fwddiff.dropna()
    covmat = fwddiff.cov()
    covmat = covmat * 252 / 10000
    eigenvecs, eigenmat = jacobi(covmat.values)
    eigvecs = Series(eigenvecs, index=covmat.columns)
    sorted_eigvecs = eigvecs.order(ascending=False)
    top3 = sorted_eigvecs[:no_factors].index
    eigenmat_df = DataFrame(eigenmat, index=covmat.columns,
                            columns=covmat.columns)
    filtered_eigenmat = eigenmat_df.filter(top3)
    return sorted_eigvecs, filtered_eigenmat


def create_factors(vols, no_factors=3):
    factors = {}
    for i in xrange(1, no_factors + 1):
        degree = 3
        if i == 1:
            degree = 1
        volfit, poly = curve_fit(vols.iloc[i - 1], degree)
        factors['PCA%s' % i] = {'volfit': volfit,
                                'poly': poly}
    return factors

if __name__ == "__main__":
    parser = optparse.OptionParser("usage: %prog [options] arg1 arg2")
    parser.add_option("-f", "--filename", dest="filename", type="string",
                       default="CBLC.csv", help="csv file for historical fwds")
    parser.add_option("-s", "--simno", dest="sim_no", type="int",
                       default=100, help="no of monte carlo simulations")
    parser.add_option("-p", "--product", dest="product", type="string",
                        help="product to price (cap/floor/zc)")
    parser.add_option("-k", "--strike", dest="strike", type="float",
                        default=1.0, help="Cap/Floor strike. Enter 5% as 5")
    parser.add_option("--freq", dest="freq", type="float",
                        default=0.25, help="Cap/Floor payment freq.Year Fract")
    parser.add_option("-m", "--maturity", dest="maturity", type="float",
                        default=1, help="Product maturity.Year Fract")
    parser.add_option('--cache_pca', dest="cache_pca", default=False,
                      help="Flag to skip pca calibration step",
                      action="store_true")
    (options, args) = parser.parse_args()
    filename = options.filename
    sim_no = options.sim_no
    if options.product is None:
        parser.error("Need to pass in a product")
    product = create_product(options)
    logger.info('filename: %s, no of sim: %s price %s', filename, sim_no,
                 product)
    forwards = pandas.read_csv(filename)
    forwards = forwards.set_index('Dates')
    forwards = forwards.dropna()
    if options.cache_pca and \
       os.path.exists('eigenmat.csv') and \
       os.path.exists('eigenvecs.csv'):
        logger.info("Using previously calculated PCAs")
        eigenmat = DataFrame.from_csv('eigenmat.csv')
        eigenvecs = Series.from_csv('eigenvecs.csv')
    else:
        logger.info('Calculating PCA')
        eigenvecs, eigenmat = calculate_pca(forwards)
        eigenmat.to_csv("eigenmat.csv")
        eigenvecs.to_csv("eigenvec.csv")
    tenors = eigenvecs.index.astype("float64").tolist()
    tenors.sort()
    sqrt_lambdas = eigenvecs[:3].apply(sqrt)
    eigenmat = eigenmat.transpose()
    vols = eigenmat.apply(lambda x: numpy.asarray(x)
                          * numpy.asarray(sqrt_lambdas))
    factors = create_factors(vols, 3)
    vol1fit, poly1 = curve_fit(vols.iloc[0], 1)
    vol2fit, poly2 = curve_fit(vols.iloc[1])
    vol3fit, poly3 = curve_fit(vols.iloc[2])
    mu = discrete_drift(poly1, poly2, poly3, tenors)
    mu2 = discrete_drift2(factors, tenors)
    initial_fwd_curve = forwards.iloc[-1:] / 100
    initial_fwd_curve = initial_fwd_curve.reset_index()
    initial_fwd_curve.pop('Dates')

    dt = 0.01
    tenor = product.maturity
    prices = []
    for i in xrange(0, sim_no):
        logger.info('Running simulation: %s', i)
        F = generate_fwd_curves(initial_fwd_curve, dt, tenor, mu,
                            vol1fit, vol2fit, vol3fit)
        #F2 = generate_fwd_curves2(initial_fwd_curve, dt, tenor, mu, factors)
        price = Pricer(F, dt, product)
        if price != -1:
            prices.append(price)
    price_series = Series(prices)
    logging.info("pv is %s", price_series.mean())
