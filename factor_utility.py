import numpy as np
import pandas as pd
from pyfinance.ols import OLS, RollingOLS, PandasRollingOLS
# from pandarallel import pandarallel
# pandarallel.initialize(progress_bar=True)
from scipy.stats import rankdata
import statsmodels.api as sm
import talib


def wa(df):
    weights = 0.9 * np.arange(len(df))
    return np.average(df, weights=weights)


def rolling_wma(df, window):
    return df.rolling(window).apply(wa)


def wma(df, window):
    """
    Weighted moving average
    :param df: dataframe
    :param window: int
    :return: dataframe of wma time seires
    """
    return df.groupby('Ticker').apply(rolling_wma, window=window)


def vwap_calc(df):
    df['Cum_Vol'] = df['Volume'].cumsum()
    df['Cum_Vol_Price'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum()
    df['VWAP'] = df['Cum_Vol_Price'] / df['Cum_Vol']
    return pd.DataFrame(df['VWAP'])


def vwap(df):
    """
    Calculation for VWAP
    :param df: dataframe
    :return: dataframe
    """

    return df.groupby('Ticker').apply(vwap_calc).squeeze()


def stock_return(df):
    """
    calcuate daily return
    :param df:
    :return:
    """
    return df.groupby('Ticker').pct_change().squeeze()


def LOG(df):
    return np.log(df)


def rolling_sum(df, window):
    return df.rolling(window).sum()


def ts_sum(df, window=10):
    """
    Wrapper function to estimate rolling sum.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """

    return df.groupby('Ticker').apply(lambda x: rolling_sum(x, window)).squeeze()


def rolling_mean(df, window):
    return df.rolling(window).mean()


def mean(df, window=10):
    """
    Wrapper function to estimate rolling mean.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.groupby('Ticker').apply(lambda x: rolling_mean(x, window)).squeeze()


def sma(df, n, m):
    """
    Wrapper function to estimate rolling SMA.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    # return df.groupby('Ticker').ewm(alpha=m/n).mean()
    return df.groupby('Ticker').apply(lambda x: x.ewm(alpha=m / n, ignore_na=True).mean())


def rolling_std(df, window):
    return df.rolling(window).std()


def ts_std(df, window=10):
    """
    Wrapper function to estimate rolling standard deviation.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.groupby('Ticker').apply(lambda x: rolling_std(x, window)).squeeze()


def rolling_corr(x, y, window):
    return pd.DataFrame(x.rolling(window).corr(y))


def correlation(x, y, window=10):
    """
    Wrapper function to estimate rolling corelations.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    df = pd.concat([x, y], axis=1)
    return df.groupby('Ticker').apply(lambda x: rolling_corr(x.iloc[:, 0], x.iloc[:, 1], window)).squeeze()


def rolling_cov(x, y, window):
    return pd.DataFrame(x.rolling(window).cov(y))


def covariance(x, y, window=10):
    """
    Wrapper function to estimate rolling covariance.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    df = pd.concat([x, y], axis=1)
    return df.groupby('Ticker').apply(lambda x: rolling_cov(x.iloc[:, 0], x.iloc[:, 1], window)).squeeze()


# def get_rank(df):
#     return rankdata(df)[-1]
#
#
# def rolling_rank(na, window):
#     """
#     Auxiliary function to be used in pd.rolling_apply
#     only used in ts_rank()
#     :param na: numpy array.
#     :return: The rank of the last value in the array.
#     """
#     return na.rolling(window).apply(get_rank)


def MAX(df, compare):
    """
    :param df: dataseries
    :param compare: series or float
    :return: series after comparing
    """

    return np.maximum(df, compare)


def MIN(df, compare):
    """
    :param df:
    :param value:
    :return:
    """
    return np.minimum(df, compare)


def rolling_count(condition, window):
    return condition.rolling(window, min_periods=0).count()

    # return len(True_Count)


def COUNT(condition, window=10):
    """
    :param df:
    :param windows:
    :return:
    """
    return condition.groupby('Ticker').apply(lambda df: rolling_count(df, window)).squeeze()


def rolling_rank(na, window):
    """
    Auxiliary function to be used in pd.rolling_apply
    only used in ts_rank()
    :param na: dataframe
    :return: The rank of the last value in the array.
    """
    return na.rolling(window).rank(pct=True)


def ts_rank(df, window=10):
    """
    Wrapper function to estimate rolling rank.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series rank over the past window days.
    """
    return df.groupby('Ticker').apply(lambda x: rolling_rank(x, window)).squeeze()


def rolling_prod(df, window):
    """
    Auxiliary function to be used in pd.rolling_apply
    used only in product()
    :param na: numpy array.
    :return: The product of the values in the array.
    """
    return df.rolling(window).apply(np.prod, raw=True)


def PROD(df, window=10):
    """
    Wrapper function to estimate rolling product.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series product over the past 'window' days.
    """
    return df.groupby('Ticker').apply(lambda x: rolling_prod(x, window)).squeeze()


def rolling_min(df, window):
    """
    estimate rolling min value
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: min value
    """
    return df.rolling(window).min()


def ts_min(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.groupby('Ticker').apply(lambda x: rolling_min(x, window)).squeeze()


def rolling_max(df, window):
    return df.rolling(window).max()


def ts_max(df, window=10):
    """
    Wrapper function to estimate rolling max.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series max over the past 'window' days.
    """
    return df.groupby('Ticker').apply(lambda x: rolling_max(x, window)).squeeze()


def delta(df, period=1):
    """
    Wrapper function to estimate difference.
    if period=1:  close[-1]-close[-2] if period=2 : close[-1]-close[-3]
    :param df: a pandas DataFrame.
    :param period: the difference grade.
    :return: a pandas DataFrame with today’s value minus the value 'period' days ago.
    """
    return df.groupby('Ticker').diff(period)


def delay(df, period=1):
    """
    Wrapper function to estimate lag n period's value.
    :param df: a pandas DataFrame.
    :param period: the lag grade.
    :return: a pandas DataFrame with lagged time series
    """
    return df.groupby('Ticker').shift(period).squeeze()


def rank(df):
    """
    Cross sectional rank
    :param df: a pandas DataFrame.
    :return: a pandas DataFrame with rank along columns.
    """
    return df.groupby('Date').rank(pct=True)
    # return df.rank(pct=True)


def scale(df, k=1):
    """
    Scaling time serie.
    sum(abs(x))=k
    :param df: a pandas DataFrame.
    :param k: scaling factor.
    :return: a pandas DataFrame rescaled df such that sum(abs(df)) = k
    """
    return df.mul(k).div(np.abs(df).sum())


def ABS(df):
    """
    Abs value of pandas dataframe
    :param df: a pandas DataFrame.
    :return: pandas Series
    """
    return df.abs().squeeze()


def SIGN(df):
    """
    Fill a DataFrame with "sign" numbers
    :param df:  a pandas DataFrame.
    :return: pandas Series
    """
    return np.sign(df).squeeze()


def rolling_argmax(df, window):
    return -df.rolling(window).apply(np.argmax) + window - 1


def ts_argmax(df, window=10):
    """
    Wrapper function to estimate which day ts_max(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.groupby('Ticker').apply(lambda x: rolling_argmax(x, window)).squeeze()


def rolling_argmin(df, window):
    return -df.rolling(window).apply(np.argmin) + window - 1


def ts_argmin(df, window=10):
    """
    Wrapper function to estimate which day ts_min(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.groupby('Ticker').apply(lambda x: rolling_argmin(x, window)).squeeze()


def rolling_decay_linear(df, period):
    """
    rolling lwma
    :param df:
    :param period:
    :return:
    """
    weights = np.arange(1, period + 1)
    wmas = df.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    return wmas


def decay_linear(df, period=10):
    """
    Linear weighted moving average implementation.
    df中从远及近分别乘以权重d，d-1，d-2，...，权重和为1
    例如：period=10时的权重列表
    [ 0.01818182,  0.03636364,  0.05454545,  0.07272727,  0.09090909,
        0.10909091,  0.12727273,  0.14545455,  0.16363636,  0.18181818]
    :param df: a pandas DataFrame.
    :param period: the LWMA period
    :return: a pandas DataFrame with the LWMA.
    """
    return df.groupby('Ticker').apply(lambda x: rolling_decay_linear(x, period)).squeeze()
    # return pd.DataFrame(na_lwma, index=df.index, columns=['CLOSE'])  # 本行有修订
    # return pd.DataFrame(na_lwma, index=df.index, columns=df.keys())  # 本行有修订


def reg(y, x):
    model = sm.OLS(y, x).fit()
    # print(model.coef_)
    return model.params


def rolling_reg(x, window):
    return x.rolling(window).apply(lambda x: reg(SEQUENCE(window), x)).squeeze()


def REGBETA(x, window):
    return x.groupby('Ticker').apply(lambda x: rolling_reg(x, window)).squeeze()


def RESI(df, window):
    """
    Beta Calculation
    :param df: a dataframe
    :return: dataframe containing
    """
    A = df.iloc[:, 0]
    B = df.iloc[:, 1]
    try:
        model = PandasRollingOLS(y=B, x=A, window=window)
        return model.se_alpha
    except:
        return pd.DataFrame(index=df.index)


def REGRESI(A, B, n):
    """
    rolling residuals calculation
    :param A: X
    :param B: Y
    :param n: rolling period
    :return: residuals time series
    """
    df = pd.concat([A, B], axis=1)
    return df.groupby('Ticker').apply(RESI, window=n).droplevel(0).sort_index().squeeze()


def SEQUENCE(n):
    return np.arange(1, n + 1)


def condition(condition, df, replace_by=0):
    try:
        alpha = pd.DataFrame(index=condition.index)
        alpha['alpha'] = replace_by
        alpha['alpha'][condition] = df[condition]
    except:
        alpha = pd.DataFrame(index=condition.index)
        alpha['alpha'] = replace_by
        alpha['alpha'][condition] = df
    return alpha.squeeze()


def SUMIF(df, n, condition):
    """
    sum if meet the condition
    :param df:
    :param n:
    :param condition:
    :return:
    """
    alpha = pd.DataFrame(index=condition.index)
    alpha['alpha'] = 0
    alpha['alpha'][condition] = df
    return ts_sum(alpha, n)


# technical indicator
def ER(df):
    bullpower = df.High - df.Close.ewm(span=20).mean()
    bearpower = df.Low - df.Close.ewm(span=20).mean()
    alpha = bullpower - bearpower
    return pd.DataFrame(alpha)


def DPO(df):
    ma = df.Close.rolling(20).mean()
    delay = ma.shift(11)
    alpha = df.Close - delay

    return pd.DataFrame(alpha)


def POS(df):
    price = (df.Close - df.Close.shift(100)) / df.Close.shift(100)
    rolling_min = price.rolling(100).min()
    rolling_max = price.rolling(100).max()
    alpha = (price - rolling_min) / (rolling_max - rolling_min)
    return pd.DataFrame(alpha)


def TII(df):
    N1 = 50
    M = 26
    N2 = 9
    close_ma = df.Close.rolling(N1).mean()
    dev = df.Close - close_ma
    devpos = pd.DataFrame(dev.where(dev > 0, 0), index=df.index)
    devneg = pd.DataFrame(-dev.where(dev < 0, 0), index=df.index)
    sumpos = devpos.rolling(M).sum()
    sumneg = devneg.rolling(M).sum()
    alpha = 100 * sumpos / (sumpos + sumneg)
    return pd.DataFrame(alpha)


def ADTM(df):
    n = 20
    dtm = pd.DataFrame(np.where(df.Open > df.Open.shift(1),
                                pd.concat([df.High - df.Open, df.Open - df.Open.shift(1)]).max(
                                    level=0), 0), index=df.index)
    dbm = pd.DataFrame(np.where(df.Open < df.Open.shift(1),
                                pd.concat([df.Open - df.Low, df.Open.shift(1) - df.Open]).max(
                                    level=0), 0), index=df.index)
    stm = dtm.rolling(n).sum()
    sbm = dbm.rolling(n).sum()
    adtm = (stm - sbm) / pd.concat([stm, sbm]).max(level=0)

    return pd.DataFrame(adtm)


def PO(df):
    ema_short = df.Close.ewm(span=9).mean()
    ema_long = df.Close.ewm(span=26).mean()
    po = (ema_short - ema_long) / ema_long * 100
    return pd.DataFrame(po)


def MADisplaced(df):
    n = 20
    m = 10
    ma_close = df.Close.rolling(n).mean()
    madisplaced = ma_close.shift(m)
    return pd.DataFrame(madisplaced)


def T3(df):
    n = 20
    va = 0.5
    temp1 = df.Close.ewm(span=n).mean()
    t1 = temp1 * (1 + va) - temp1.ewm(span=n).mean() * va
    temp2 = t1.ewm(span=n).mean()
    t2 = temp2 * (1 + va) - temp2.ewm(span=n).mean() * va
    temp3 = t2.ewm(span=n).mean()
    t3 = temp3 * (1 + va) - temp3.ewm(span=n).mean() * va

    return pd.DataFrame(t3)


def VMA(df):
    n = 20
    price = (df.High + df.Low + df.Open + df.Close) / 4
    vma = price.rolling(n).mean()
    return pd.DataFrame(vma)


def CR(df):
    n = 20
    typ = (df.High + df.Low + df.Close) / 3
    h = MAX(df.High - typ.shift(1), 0)
    l = MAX(typ.shift(1) - df.Low, 0)
    cr = h.rolling(n).sum() / l.rolling(n).sum() * 100
    return pd.DataFrame(cr)


def VIDYA(df):
    n = 10
    temp = (df.Close - df.Close.shift(n)).abs()
    vi = temp / temp.rolling(n).sum()
    vidya = vi * df.Close + (1 - vi) * df.Close.shift(1)

    return pd.DataFrame(vidya)
