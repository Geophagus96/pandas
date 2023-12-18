import time
import pandas as pd
from numpy import abs
from numpy import log
from numpy import sign
from STOCK_Factor_Research.Tenjin_gtja_research.factor_utility import *
import talib


class Tenjin_alpha(object):

    def __init__(self, price):
        self.price = price
        self.open = self.price['Open']
        self.close = self.price['Close']
        self.low = self.price['Low']
        self.high = self.price['High']
        self.volume = self.price['Volume']
        self.returns = stock_return(self.close)
        self.DBM = condition(self.open >= delay(self.open, 1), 0,
                             MAX((self.open - self.low), (self.open - delay(self.open, 1))))
        self.DTM = condition(self.open <= delay(self.open, 1), 0,
                             MAX((self.high - self.open), (self.open - delay(self.open, 1))))
        # self.vwap = vwap(self.price)

    # gtja recommendation 1:
    # def alpha1(self, window):
    #     # alpha = self.volume.rolling(window).corr(self.vwap)
    #     return -1 * correlation(self.volume,self.vwap, window)

    # gtja recommendation 2:
    def alpha_gtja_recommendation_2(self):
        return self.open / delay(self.close, 1)

    # gtja recommendation 3:
    def alpha_gtja_recommendation_3(self, window=20):
        # temp = self.volume.rolling(window + 1).mean()
        # alpha = -1 * self.volume / temp

        return -1 * self.volume / mean(self.volume, window)

    # gtja recommendation 4:
    def alpha_gtja_recommendation_4(self, window=20):
        return -1 * correlation(self.high / self.low, self.volume, window)

    ###########################
    # def Vwap(self):
    #     return pd.DataFrame(self.vwap)

    # def Skewness_20(self):
    #     ret = self.returns
    #     alpha = ret.rolling(20).skew()
    #     return pd.DataFrame(alpha)
    #
    # def Skewness_60(self):
    #     ret = self.returns
    #     alpha = ret.rolling(60).skew()
    #     return pd.DataFrame(alpha)
    #
    # def Kurtosis_20(self):
    #     ret = self.close.returns
    #     alpha = ret.rolling(20).kurt()
    #     return pd.DataFrame(alpha)
    #
    # def Kurtosis_60(self):
    #     ret = self.close.returns
    #     alpha = ret.rolling(60).kurt()
    #     return pd.DataFrame(alpha)

    ######price
    # - close/vwap
    # def alpha_ql_001(self):
    #     return -self.close / self.vwap

    # -delta(close/vwap,1)
    # def alpha_ql_002(self):
    #     temp = self.close / self.vwap
    #     return -1*delta(temp)

    # -delta(returns,1)
    def alpha_ql_003(self):
        ret = self.returns
        return -1 * delta(ret, 1)

    # -ts_rank(returns,10)
    def alpha_ql_004(self):
        ret = self.returns
        return -1 * ts_rank(ret, 10)

    # -delta(delta(close,1),1)
    def alpha_ql_005(self):
        return -1 * delta(delta(self.close))

    # - ts_std((high-low),10)
    def alpha_ql_006(self):
        temp = self.high - self.low
        return ts_std(temp, 10)

    # delta(ts_std((high-low),10),1)
    def alpha_ql_007(self):
        temp = self.high - self.low
        return delta(ts_std(temp, 10))

    # - ts_std(delta((high-low),1),10)
    def alpha_ql_008(self):
        temp = self.high - self.low
        return -1 * ts_std(delta(temp, 1), 10)

    ######volume
    # - volume/adv20
    def alpha_ql_009(self):
        return -1 * self.volume / mean(self.volume, 20)

    # - delta(volume/adv20,1)
    def alpha_ql_010(self):
        return -1 * delta(self.alpha_ql_009())

    # ts_max(volume, 10) / ts_min(volume, 10)
    def alpha_ql_011(self):
        return ts_max(self.volume, 10) / ts_min(self.volume, 10)

    # ts_rank(volume,10)
    def alpha_ql_012(self):
        return ts_rank(self.volume, 10)

    # delta(ts_rank(volume,10),1)
    def alpha_ql_013(self):
        return delta(ts_rank(self.volume, 10), 1)

    # ts_std(ts_rank(adv20,10),10)
    def alpha_ql_014(self):
        adv20 = mean(self.volume, 20)
        return ts_std(ts_rank(adv20, 10), 10)

    # ts_std(delta(ts_rank(volume,10),1),10)
    def alpha_ql_015(self):
        return ts_std(delta(ts_rank(self.volume, 10), 1), 10)

    # delta(ts_std(ts_rank(volume,10),10),1)
    def alpha_ql_016(self):
        return delta(ts_std(ts_rank(self.volume, 10), 10), 1)

    ######
    # volume / mean(volume, 5) * close / Ts-max(close, 5)
    def alpha_ql_017(self):
        return self.volume / mean(self.volume, 5) * self.close / ts_max(self.close, 5)

    # ( - delta(close, 1) + delta(delay(close, 1), 1)) / delay(close, 1)
    def alpha_ql_018(self):
        return (delta(self.close, 1) + delta(delay(self.close, 1), 1)) / delay(self.close, 1)

    # rank(1 / ts_std(high-low, 5) ) * rank(- delta(close, 1) ) * rank(delta(volume, 1) )
    def alpha_ql_019(self):
        return rank(1 / ts_std(self.high - self.low, 5)) * rank(-1 * delta(self.close, 1)) * rank(delta(self.volume, 1))

    #
    # def alpha_ql_020(self):
    #     alpha = pd.DataFrame(index=self.price.index)
    #     alpha['Alpha20'] = np.nan
    #
    #     for i in range(10, len(self.price)):
    #         high = self.high[i - 10:i]
    #         vwap = self.vwap[i - 10:i]
    #         close = self.close[i - 10:i]
    #
    #         c = (high - vwap).rank(pct=True)
    #         p1 = (close.rank(pct=True)).diff(1).diff(1).iloc[-1]
    #         p2 = (2 - c.iloc[0] / c.iloc[2] + 1) * c.iloc[2] / c.iloc[1]
    #         alpha['Alpha20'].iloc[i] = p1 * p2
    #
    #     return alpha
    #
    # def alpha_ql_021(self):
    #     alpha = pd.DataFrame(index=self.price.index)
    #     alpha['Alpha21'] = np.nan
    #
    #     for i in range(10, len(self.price)):
    #         close = self.close[i - 10:i]
    #
    #         c = close.rank(pct=True)
    #         a = c.iloc[0] - c.iloc[1]
    #         b = c.iloc[8] - c.iloc[9]
    #
    #         alpha['Alpha21'].iloc[i] = (a ** 2 - b ** 2) / 2 / (c.iloc[0] - c.iloc[9])
    #
    #     return alpha

    # Para1 = -delta(MA5 - MA10, 1) * abs(delta(close, 1)) * (high - low) / close
    # Para2 = (mean(vwap, 5) – close) * (volume - adv5) / adv5
    # Alpha3 = Para1 * (1 + Para2)

    # def alpha_ql_022(self):
    #     para1 = -delta(mean(self.close,5)-mean(self.close,10),1) * delta(self.close,1).abs() * (self.high-self.low)/self.close
    #     para2 = (mean(self.vwap,5) - self.close) * (self.volume - mean(self.volume, 5)) / (mean(self.volume, 5))
    #     return para1 * (1+para2)

    #############################################################################
    # -1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE OPEN) / OPEN)), 6))
    def alpha_gtja_001(self):
        # check
        return -1 * correlation(rank(delta(np.log(self.volume), 1)),
                                rank(((self.close - self.open) / self.open)), 6)

    # -1 * delta((((close-low)-(high-close))/((high-low)),1))
    def alpha_gtja_002(self):
        # check
        return -1 * delta(((self.close - self.low) - (self.high - self.close)) / (self.high - self.low), 1)

    ################################################################
    # SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
    def alpha_gtja_003(self):
        # check

        # delay1 = delay(self.close, 1)
        # condtion1 = (self.close == delay1)
        # condition2 = (self.close > delay1)
        # condition3 = (self.close < delay1)
        #
        # part2 = (self.close - np.minimum(delay1[condition2], self.low[condition2])).fillna(0)
        # part3 = (self.close - np.maximum(delay1[condition3], self.low[condition3])).fillna(0)
        # return mean(part2 + part3, 6)
        df = ts_sum(condition((self.close == delay(self.close, 1)), 0,
                              self.close - condition((self.close > delay(self.close, 1)),
                                                     MIN(self.low, delay(self.close, 1)),
                                                     MAX(self.high, delay(self.close, 1)))), 6)
        return df

    ########################################################################
    # IF (((SUM(CLOSE, 8) / 8) + STD(CLOSE, 8)) < (SUM(CLOSE, 2) / 2)), -1*1
    # ELIF (((SUM(CLOSE, 2) / 2) < ((SUM(CLOSE, 8) / 8) STD(CLOSE, 8))), 1
    # ELIF STD(CLOSE, 8))) ? 1 : (((1 < (VOLUME / MEAN(VOLUME,20) 20))) or ((VOLUME / MEAN(VOLUME,20) 20)) == 1)), 1
    # ELSE ,-1*1
    def alpha_gtja_004(self):
        condition1 = ((ts_sum(self.close, 8) / 8 + ts_std(self.close, 8)) < (ts_sum(self.close, 2) / 2))
        condition2 = ((ts_sum(self.close, 2) / 2) < (ts_sum(self.close, 8) / 8 + ts_std(self.close, 8)))
        condition3 = (1 <= self.volume / mean(self.volume, 20))
        return condition(condition1, -1, condition(condition2, 1, condition(condition3, 1, -1)))

    ################################################################
    # (-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))
    def alpha_gtja_005(self):
        # check
        return -1 * ts_rank(correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5), 3)

    ###############################################################
    # (RANK(SIGN(DELTA((((OPEN * 0.85) + (HIGH * 0.15))), 4)))* -1)
    def alpha_gtja_006(self):
        return -1 * rank(SIGN(delta(self.open * 0.85 + self.high * 0.15, 4)))

    ##################################################################
    # ((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP-CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))
    # def alpha_gtja_007(self):
    #     # check
    #     return rank(MAX(self.vwap - self.close,3)) + rank(MIN(self.vwap-self.close,3)) * rank(delta(self.volume,3))

    ##################################################################
    # RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)
    # def alpha_gtja_008(self):
    #     # check
    #     return rank(delta(((self.high + self.low)/2*0.2 + self.vwap * 0.8), 4) * -1)

    ##################################################################
    # SMA(((HIGH+LOW)/2-(DEALAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH - LOW)/VOLUME,7,2)
    def alpha_gtja_009(self):
        # check
        return sma(((self.high + self.low) / 2 - (delay(self.high) + delay(self.low)) / 2) * (
                (self.high - self.low) / self.volume), 7, 2)

    ##################################################################
    # (RANK(MAX(((RET < 0) ? STD(RET, 20) : CLOSE)^2),5))
    def alpha_gtja_010(self):
        return rank(MAX(condition((self.returns < 0), ts_std(self.returns, 20), self.close) ** 2, 5))

    ##################################################################
    # SUM(((CLOSE-LOW)-(HIGH - CLOSE))./(HIGH - LOW).*VOLUME,6)
    def alpha_gtja_011(self):
        # check
        return ts_sum(((self.close - self.low) - (self.high - self.close)) / ((self.high - self.low) * self.volume),
                      window=6)

    ##################################################################
    # (RANK((OPEN-(SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE VWAP)))))
    # def alpha_gtja_012(self):
    #     # check
    #     return rank(self.open - (ts_sum((self.vwap),10)) / 10) * (-1 * rank((self.close - self.vwap).abs()))

    ##################################################################
    # (((HIGH * LOW)^0.5)-VWAP)
    # def alpha_gtja_013(self):
    #     # check
    #     temp = self.high - self.low
    #     return temp.groupby('Ticker').apply(np.sqrt) - self.vwap

    ##################################################################
    # CLOSE - DELAY(CLOSE,5)
    def alpha_gtja_014(self):
        # check
        return self.close - delay(self.close, 5)

    ##################################################################
    # OPEN/DELAY(CLOSE,1) - 1
    def alpha_gtja_015(self):
        return self.open / delay(self.close, 1) - 1

    ##################################################################
    # -1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5))
    # def alpha_gtja_016(self):
    #     return -1 * ts_max(rank(correlation(rank(self.volume), rank(self.vwap),5)),5)

    ##################################################################
    # RANK((VWAP - MAX(VWAP, 15)))^DELTA(CLOSE, 5)
    # def alpha_gtja_017(self):
    #     return rank(self.vwap - MAX(self.vwap,15)) ** delta(self.close,5)

    ##################################################################
    # CLOSE/DELAY(CLOSE,5)
    def alpha_gtja_018(self):
        # check
        return self.close / delay(self.close, 5)

    ##################################################################
    # (CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE -DELAY(CLOSE,5))/CLOSE))
    def alpha_gtja_019(self):
        # alpha = pd.DataFrame(index=self.price.index)
        # alpha['alpha'] = np.nan
        # delay5 = delay(self.close, 5)
        # condition1 = (self.close < delay5)
        # condition2 = (self.close > delay5)
        #
        # alpha['alpha'].loc[condition1] = (self.close.loc[condition1] - delay5.loc[condition1]) / delay5.loc[condition1]
        # alpha['alpha'].loc[condition2] = (self.close.loc[condition2] - delay5.loc[condition2]) / self.close.loc[
        #     condition2]
        # df = condition((self.close - delay(self.close, 5), (self.close - delay(self.close, 5)) / delay(self.close, 5)),
        #                condition((self.close == delay(self.close, 5)), 0,
        #                          (self.close - delay(self.close, 5)) / self.close))
        return condition(self.close < delay(self.close, 5), (self.close - delay(self.close, 5)) / delay(self.close, 5),
                         condition(self.close == delay(self.close, 5), 0,
                                   (self.close - delay(self.close, 5)) / self.close))

    ##################################################################
    # (CLOSE - DELAY(CLOSE,6))/DELAY(CLOSE,6)*100
    def alpha_gtja_020(self):
        # check
        return (self.close - delay(self.close, 6)) / delay(self.close, 6) * 100

    ##################################################################
    # REGBETA(MEAN(CLOSE, 6), SEQUENCE(6))
    def alpha_gtja_021(self):
        # check
        A = mean(self.close, 6)
        alpha = REGBETA(A, 6)

        return alpha

    ##################################################################
    # SMA(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6) - DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)
    def alpha_gtja_022(self):
        # check
        return sma((self.close - mean(self.close, 6)) / mean(self.close, 6) - delay(
            (self.close - mean(self.close, 6)) / mean(self.close, 6), 6), 12, 1)

    ##################################################################
    # SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE:20),0),20,1)/(SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)+SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1))*100
    # def alpha_023(self):
    #     #
    #     condition1 = (self.close > self.close.shift())
    #     temp1 = pd.rolling_std(self.close, 20)[condition1]
    #     temp1 = temp1.fillna(0)
    #     temp2 = pd.rolling_std(self.close, 20)[~condition1]
    #     temp2 = temp2.fillna(0)
    #     part1 = pd.ewma(temp1, alpha=1.0 / 20)
    #     part2 = pd.ewma(temp2, alpha=1.0 / 20)
    #     result = part1 * 100 / (part1 + part2)
    #     alpha = result.iloc[-1, :]
    #     return alpha.dropna()

    ##################################################################
    # SMA(CLOSE - DELAY(CLOSE,5), 5,1)
    def alpha_gtja_024(self):
        # check
        return sma(self.close - delay(self.close, 5), 5, 1)

    ##################################################################
    # def alpha_025(self):
    #     n = 9
    #     part1 = (self.close.diff(7)).rank(axis=1, pct=True)
    #     part1 = part1.iloc[-1, :]
    #     temp = self.volume / pd.rolling_mean(self.volume, 20)
    #     temp1 = temp.iloc[-9:, :]
    #     seq = [2 * i / (n * (n + 1)) for i in range(1, n + 1)]
    #     weight = np.array(seq)
    #
    #     temp1 = temp1.apply(lambda x: x * weight)
    #     ret = self.close.pct_change()
    #     rank_sum_ret = (ret.sum()).rank(pct=True)
    #     part2 = 1 - temp1.sum()
    #     part3 = 1 + rank_sum_ret
    #     alpha = -part1 * part2 * part3
    #     return alpha.dropna()

    ##################################################################
    # ((((SUM(CLOSE, 7) / 7) - CLOSE)) + ((CORR(VWAP, DELAY(CLOSE, 5), 230))))
    # def alpha_gtja_026(self):
    #     return (ts_sum(self.close,7) / 7 - self.close) + correlation(self.vwap,delay(self.close,5),230)

    ##################################################################
    # WMA((CLOSE-DELAY(CLOSE,3))/DELAY( CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)
    def alpha_gtja_027(self):
        delay3 = delay(self.close, 3)
        delay6 = delay(self.close, 6)
        alpha = wma((self.close - delay3) / delay3 * 100 + (self.close - delay6) / delay6 * 100, 12)
        return alpha

    ##################################################################
    # 3*SMA((CLOSE - TSMIN (LOW, TSMAX (HIGH, TSMIN (LOW,9))* - 2*SMA(SMA((CLOSE-TSMIN (LOW,9))/MAX(HIGH,9)-TSMAX (LOW,9))*100,3,1),3,1)
    def alpha_gtja_028(self):
        # alpha = 3*sma(((self.close-ts_min(self.low,9))/ (ts_max(self.high,9) - ts_min(self.low,9)) * 100),3,1) - 2*sma(sma(((self.close-ts_min(self.low,9)) / (ts_max(self.high,9)-ts_max(self.low,9)) * 100),3,1),3,1)
        min_low9 = ts_min(self.low, 9)
        max_high9 = ts_max(self.high, 9)
        max_low9 = ts_max(self.low, 9)
        alpha = 3 * sma((self.close - min_low9) / (max_high9 - min_low9) * 100, 3, 1) - 2 * sma(
            sma((self.close - min_low9) / (max_high9 - max_low9) * 100, 3, 1), 3, 1)
        return alpha

    ##################################################################
    # (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME
    def alpha_gtja_029(self):
        delay6 = delay(self.close, 6)
        alpha = (self.close - delay6) / delay6 * self.volume
        return alpha

    ##################################################################
    # def alpha_030(self):
    #     return 0

    ##################################################################
    # (CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100
    def alpha_gtja_031(self):
        # check
        mean12 = mean(self.close, 12)
        alpha = (self.close - mean12) / mean12 * 100
        return alpha

    ##################################################################
    # -1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))
    def alpha_gtja_032(self):
        alpha = ts_sum(rank(correlation(rank(self.high), rank(self.volume), 3)), 3)
        return alpha

    # ################################################################# (((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5),
    # 5)) * RANK(((SUM(RET, 240) - SUM(RET, 20)) / 220))) *TSRANK(VOLUME, 5))
    def alpha_gtja_033(self):
        ret = self.returns
        return ((-1 * ts_min(self.low, 5) + delay(ts_min(self.low, 5), 5)) * rank(
            (ts_sum(ret, 240) - ts_sum(ret, 20)) / 220)) * ts_rank(self.volume, 5)

    ##################################################################
    def alpha_gtja_034(self):
        # check
        return mean(self.close, 12) / self.close

    ##################################################################
    # (MIN(RANK(DECAYLINEAR(DELTA(OPEN, 1), 15)),RANK(DECAYLINEAR(CORR((VOLUME), ((OPEN * 0.65) + (OPEN * 0.35)), 17), 7))) * 1)
    # def alpha_035(self):
    #     decay_linear(delta(self.open, 1), 15)
    #
    #     n = 15
    #     m = 7
    #     temp1 = self.open.diff()
    #     temp1 = temp1.iloc[-n:, :]
    #     seq1 = [2 * i / (n * (n + 1)) for i in range(1, n + 1)]
    #     seq2 = [2 * i / (m * (m + 1)) for i in range(1, m + 1)]
    #     weight1 = np.array(seq1)
    #     weight2 = np.array(seq2)
    #     part1 = temp1.apply(lambda x: x * weight1)
    #     part1 = part1.rank(axis=1, pct=True)
    #
    #     temp2 = 0.65 * self.open + 0.35 * self.open
    #     temp2 = pd.rolling_corr(temp2, self.volume, 17)
    #     temp2 = temp2.iloc[-m:, :]
    #     part2 = temp2.apply(lambda x: x * weight2)
    #     alpha = MIN(part1.iloc[-1, :], -part2.iloc[-1, :])
    #     alpha = alpha[(alpha < np.inf) & (alpha > -np.inf)]
    #     alpha = alpha.dropna()
    #     return alpha

    ##################################################################
    # def alpha_036(self):
    #     # check
    #     alpha = pd.DataFrame(index=self.price.index)
    #     alpha['alpha_036'] = np.nan
    #     for i in range(6, len(self.price)):
    #         df = self.price.iloc[:i]
    #         temp1 = df.Volume.rank(pct=True)
    #         temp2 = self.vwap.iloc[:i].rank(pct=True)
    #         p1 = temp1.rolling(6).corr(temp2)
    #         re = p1.rolling(2).sum()
    #         alpha['alpha_036'].iloc[i] = re.iloc[-1]
    #
    #     #         temp1=self.volume.rank(axis=1,pct=True)
    #     #         temp2=self.avg_price.rank(axis=1,pct=True)
    #     #         part1=pd.rolling_corr(temp1,temp2,6)
    #     #         result=pd.rolling_sum(part1,2)
    #     #         result=result.rank(axis=1,pct=True)
    #     #         alpha=result.iloc[-1,:]
    #     return alpha

    ##################################################################
    # (-1 * RANK(((SUM(OPEN, 5) * SUM(RET, 5)) DELAY((SUM(OPEN, 5) * SUM(RET, 5)), 10))))
    def alpha_gtja_037(self):
        # check
        ret = self.returns
        return -1 * rank(
            ts_sum(self.open, 5) * ts_sum(ret, 5) - delay(ts_sum(self.open, 5) * ts_sum(ret, 5), 10))

    ##################################################################
    # (((SUM(HIGH, 20) / 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0)
    # (((SUM(HIGH, 20) / 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0)
    def alpha_gtja_038(self):
        # check
        # print(ts_sum(self.high, 20)/20)
        return condition(ts_sum(self.high, 20) / 20 < self.high, -1 * delta(self.high, 2), 0)

    ##################################################################
    # def alpha_039(self):
    #     n = 8
    #     m = 12
    #     temp1 = self.close.diff(2)
    #     temp1 = temp1.iloc[-n:, :]
    #     seq1 = [2 * i / (n * (n + 1)) for i in range(1, n + 1)]
    #     seq2 = [2 * i / (m * (m + 1)) for i in range(1, m + 1)]
    #
    #     weight1 = np.array(seq1)
    #     weight2 = np.array(seq2)
    #     part1 = temp1.apply(lambda x: x * weight1)
    #     part1 = part1.rank(axis=1, pct=True)
    #
    #     temp2 = 0.3 * self.avg_price + 0.7 * self.open
    #     volume_180 = pd.rolling_mean(self.volume, 180)
    #     sum_vol = pd.rolling_sum(volume_180, 37)
    #     temp3 = pd.rolling_corr(temp2, sum_vol, 14)
    #     temp3 = temp3.iloc[-m:, :]
    #     part2 = -temp3.apply(lambda x: x * weight2)
    #     part2.rank(axis=1, pct=True)
    #     result = part1.iloc[-1, :] - part2.iloc[-1, :]
    #     alpha = result
    #     alpha = alpha[(alpha < np.inf) & (alpha > -np.inf)]
    #     alpha = alpha.dropna()
    #     return alpha

    ##################################################################
    # need more time
    def alpha_gtja_040(self):
        return ts_sum(condition(self.close > delay(self.close, 1), self.volume, 0), 26) / ts_sum(
            condition(self.close <= delay(self.close, 1), self.volume, 0), 26) * 100

    ##################################################################
    # def alpha_041(self):
    #     # check
    #     alpha = pd.DataFrame(index=self.price.index)
    #     alpha['alpha_041'] = np.nan
    #     delay = self.vwap.shift(3)
    #     rollmax = delay.diff().rolling(5).max()
    #     for i in range(6, len(self.price)):
    #         re = rollmax.iloc[:i].rank(pct=True)
    #         alpha['alpha_041'].iloc[i] = re.iloc[-1]
    #     #         delta_avg=self.avg_price.diff(3)
    #     #         part=MAX(delta_avg,5)
    #     #         result=-part.rank(axis=1,pct=True)
    #     #         alpha=result.iloc[-1,:]
    #     #         alpha=alpha.dropna()
    #     return alpha

    ##################################################################
    def alpha_gtja_042(self):
        # check
        return -1 * rank(ts_std(self.high, 10)) * correlation(self.high, self.volume, 10)

    ##################################################################
    # Alpha_gtja_43
    def alpha_gtja_43(self):
        return ts_sum(condition(self.close > delay(self.close, 1), self.volume,
                                condition(self.close < delay(self.close, 1), -self.volume, 0)), 6)

    # alpha_gtja_44 (TSRANK(DECAYLINEAR(CORR(((LOW )), MEAN(VOLUME,10), 7), 6),4) + TSRANK(DECAYLINEAR(DELTA((VWAP), 3), 10), 15))
    # def alpha_gtja_44(self):
    #     df = (ts_rank(decay_linear(correlation(((self.low)), mean(self.volume, 10), 7), 6), 4) + ts_rank(
    #         decay_linear(delta((VWAP), 3), 10), 15))
    #     return df

    def alpha_gtja_045(self):
        pass

    # alpha_gtja_46 (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)
    def alpha_gtja_046(self):
        df = (mean(self.close, 3) + mean(self.close, 6) + mean(self.close, 12) + mean(self.close, 24)) / (
                4 * self.close)
        return df

    # alpha_gtja_47 SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,9,1)
    def alpha_gtja_047(self):
        df = sma((ts_max(self.high, 6) - self.close) / (ts_max(self.high, 6) - ts_min(self.low, 6)) * 100, 9, 1)
        return df

    # alpha_gtja_48 (-1*((RANK(((SIGN((CLOSE - DELAY(CLOSE, 1))) + SIGN((DELAY(CLOSE, 1) - DELAY(CLOSE, 2)))) + SIGN((DELAY(CLOSE, 2) - DELAY(CLOSE, 3)))))) * SUM(VOLUME, 5)) / SUM(VOLUME, 20))
    def alpha_gtja_048(self):
        df = (-1 * ((rank(((SIGN((self.close - delay(self.close, 1))) + SIGN(
            (delay(self.close, 1) - delay(self.close, 2)))) + SIGN(
            (delay(self.close, 2) - delay(self.close, 3)))))) * ts_sum(self.volume, 5)) / ts_sum(self.volume, 20))
        return df

    # alpha_gtja_49 SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(L OW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(L OW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HI GH,1)),ABS(LOW-DELAY(LOW,1)))),12))
    def alpha_gtja_049(self):
        return ts_sum(condition((self.high + self.low) >= (delay(self.high, 1) + delay(self.low, 1)), 0,
                                MAX(ABS(self.high - delay(self.high, 1)), ABS(self.low - delay(self.low, 1)))), 12) / (
                       ts_sum(condition((self.high + self.low) >= (delay(self.high, 1) + delay(self.low, 1)), 0,
                                        MAX(ABS(self.high - delay(self.high, 1)),
                                            ABS(self.low - delay(self.low, 1)))), 12) + ts_sum(
                   condition((self.high + self.low) <= (delay(self.high, 1) + delay(self.low, 1)), 0,
                             MAX(ABS(self.high - delay(self.high, 1)), ABS(self.low - delay(self.low, 1)))), 12))

    # alpha_gtja_50 SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(L OW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(L OW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HI GH,1)),ABS(LOW-DELAY(LOW,1)))),12))-SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HI GH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0: MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELA Y(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))
    def alpha_gtja_050(self):
        return ts_sum(condition((self.high + self.low) <= (delay(self.high, 1) + delay(self.low, 1)), 0,
                                MAX(ABS(self.high - delay(self.high, 1)), ABS(self.low - delay(self.low, 1)))), 12) / (
                       ts_sum(condition((self.high + self.low) <= (delay(self.high, 1) + delay(self.low, 1)), 0,
                                        MAX(ABS(self.high - delay(self.high, 1)),
                                            ABS(self.low - delay(self.low, 1)))), 12) + ts_sum(
                   condition((self.high + self.low) >= (delay(self.high, 1) + delay(self.low, 1)), 0,
                             MAX(ABS(self.high - delay(self.high, 1)), ABS(self.low - delay(self.low, 1)))),
                   12)) - ts_sum(condition((self.high + self.low) >= (delay(self.high, 1) + delay(self.low, 1)), 0,
                                           MAX(ABS(self.high - delay(self.high, 1)),
                                               ABS(self.low - delay(self.low, 1)))), 12) / (ts_sum(
            condition((self.high + self.low) >= (delay(self.high, 1) + delay(self.low, 1)), 0,
                      MAX(ABS(self.high - delay(self.high, 1)), ABS(self.low - delay(self.low, 1)))), 12) + ts_sum(
            condition((self.high + self.low) <= (delay(self.high, 1) + delay(self.low, 1)), 0,
                      MAX(ABS(self.high - delay(self.high, 1)), ABS(self.low - delay(self.low, 1)))), 12))

    # alpha_gtja_51  SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(L OW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(L OW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HI GH,1)),ABS(LOW-DELAY(LOW,1)))),12))
    def alpha_gtja_051(self):
        df = ts_sum(condition((self.high + self.low) <= (delay(self.high, 1) + delay(self.low, 1)), 0,
                              MAX(ABS(self.high - delay(self.high, 1)), ABS(self.low - delay(self.low, 1)))), 12) / (
                     ts_sum(condition((self.high + self.low) <= (delay(self.high, 1) + delay(self.low, 1)), 0,
                                      MAX(ABS(self.high - delay(self.high, 1)),
                                          ABS(self.low - delay(self.low, 1)))), 12) + ts_sum(
                 condition((self.high + self.low) >= (delay(self.high, 1) + delay(self.low, 1)), 0,
                           MAX(ABS(self.high - delay(self.high, 1)), ABS(self.low - delay(self.low, 1)))), 12))
        return df

    # alpha_gtja_52  SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-L),26)* 100
    def alpha_gtja_052(self):
        df = ts_sum(MAX(self.high - delay((self.high + self.low + self.close) / 3, 1), 0), 26) / ts_sum(
            MAX(delay((self.high + self.low + self.close) / 3, 1) - self.low, 0), 26) * 100
        return df

    # alpha_gtja_53 COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100
    def alpha_gtja_053(self):
        df = COUNT(self.close > delay(self.close, 1), 12) / 12 * 100
        return df

    # alpha_gtja_54 (-1 * RANK((STD(ABS(CLOSE - OPEN)) + (CLOSE - OPEN)) + CORR(CLOSE, OPEN,10)))
    def alpha_gtja_054(self):
        df = (-1 * rank(
            (ts_std(ABS(self.close - self.open), 10) + (self.close - self.open)) + correlation(self.close, self.open,
                                                                                               10)))

        return df

    # # alpha_gtja_56 (RANK((OPEN - TSMIN(OPEN, 12))) < RANK((RANK(CORR(SUM(((HIGH + LOW) / 2), 19), SUM(MEAN(VOLUME,40), 19), 13))^5)))
    # def alpha_gtja_56(self):
    #     df=(rank((self.open - ts_min(self.open, 12))) < rank((RANK(CORR(SUM(((HIGH + LOW) / 2), 19), SUM(MEAN(VOLUME,40), 19), 13))^5)))
    #     return df
    # alpha_gtja_57 SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)
    def alpha_gtja_057(self):
        df = sma((self.close - ts_min(self.low, 9)) / (ts_max(self.high, 9) - ts_min(self.low, 9)) * 100, 3, 1)
        return df

    # alpha_gtja_58 COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100
    def alpha_gtja_058(self):
        df = COUNT(self.close > delay(self.close, 1), 20) / 20 * 100
        return df

    # alpha_gtja_59 SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,D ELAY(CLOSE,1)))),20)
    def alpha_gtja_059(self):
        return ts_sum(condition(self.close == delay(self.close, 1), 0,
                                self.close - condition(self.close > delay(self.close, 1),
                                                       MIN(self.low, delay(self.close, 1)),
                                                       MAX(self.high, delay(self.close, 1)))), 20)

    # alpha_gtja_60 SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,20)
    def alpha_gtja_060(self):
        df = ts_sum(((self.close - self.low) - (self.high - self.close)) / (self.high - self.low) * self.volume, 20)
        return df

    # alpha_gtja_62 (-1 * CORR(HIGH, RANK(VOLUME), 5))
    def alpha_gtja_062(self):
        df = (-1 * correlation(self.high, rank(self.volume), 5))
        return df

    # alpha_gtja_63 SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100
    def alpha_gtja_063(self):
        df = sma(MAX(self.close - delay(self.close, 1), 0), 6, 1) / sma(ABS(self.close - delay(self.close, 1)), 6,
                                                                        1) * 100
        return df

    # alpha_gtja_64 SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100
    def alpha_gtja_064(self):
        df = sma(MAX(self.close - delay(self.close, 1), 0), 6, 1) / sma(ABS(self.close - delay(self.close, 1)), 6,
                                                                        1) * 100
        return df

    # alpha_gtja_65 MEAN(CLOSE,6)/CLOSE
    def alpha_gtja_065(self):
        df = mean(self.close, 6) / self.close
        return df

    # alpha_gtja_66 (CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100
    def alpha_gtja_066(self):
        df = (self.close - mean(self.close, 6)) / mean(self.close, 6) * 100
        return df

    # alpha_gtja_67 SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100
    def alpha_gtja_067(self):
        df = sma(MAX(self.close - delay(self.close, 1), 0), 24, 1) / sma(ABS(self.close - delay(self.close, 1)), 24,
                                                                         1) * 100
        return df

    # alpha_gtja_68 SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,15,2)
    def alpha_gtja_068(self):
        df = sma(((self.high + self.low) / 2 - (delay(self.high, 1) + delay(self.low, 1)) / 2) * (
                self.high - self.low) / self.volume, 15, 2)
        return df

    # alpha_gtja_69 (SUM(DTM,20)>SUM(DBM,20) ? (SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20) : (SUM(DTM,20)=SUM(DBM,20) ? 0:(SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)))
    def alpha_gtja_069(self):
        return condition(ts_sum(self.DTM, 20) > ts_sum(self.DBM, 20),
                         (ts_sum(self.DTM, 20) - ts_sum(self.DBM, 20)) / ts_sum(self.DTM, 20),
                         condition(ts_sum(self.DTM, 20) == ts_sum(self.DBM, 20), 0,
                                   (ts_sum(self.DTM, 20) - ts_sum(self.DBM, 20)) / ts_sum(self.DBM, 20)))

    # alpha_gtja_70 STD(AMOUNT,6)
    def alpha_gtja_070(self):
        pass
        # df=ts_std(self.close*self.volume,6)

    # alpha_gtja_71 (CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100
    def alpha_gtja_071(self):
        df = (self.close - mean(self.close, 24)) / mean(self.close, 24) * 100
        return df

    # alpha_gtja_72 SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)
    def alpha_gtja_072(self):
        df = sma((ts_max(self.high, 6) - self.close) / (ts_max(self.high, 6) - ts_min(self.low, 6)) * 100, 15, 1)
        return df

    # alpha_gtja_76 STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)
    def alpha_gtja_076(self):
        df = ts_std(ABS((self.close / delay(self.close, 1) - 1)) / self.volume, 20) / mean(
            ABS((self.close / delay(self.close, 1) - 1)) / self.volume, 20)
        return df

    # alpha_gtja_78 ((HIGH+LOW+CLOSE)/3-MEAN((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOS E)/3,12)),12))
    def alpha_gtja_078(self):
        df = ((self.high + self.low + self.close) / 3 - mean((self.high + self.low + self.close) / 3, 12)) / (
                0.015 * mean(ABS(self.close - mean((self.high + self.low + self.close) / 3, 12)), 12))
        return df

    # alpha_gtja_79 SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100
    def alpha_gtja_079(self):
        df = sma(MAX(self.close - delay(self.close, 1), 0), 12, 1) / sma(ABS(self.close - delay(self.close, 1)), 12,
                                                                         1) * 100
        return df

    # alpha_gtja_80 (VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100
    def alpha_gtja_080(self):
        df = (self.volume - delay(self.volume, 5)) / delay(self.volume, 5) * 100
        return df

    # alpha_gtja_81 SMA(VOLUME,21,2)
    def alpha_gtja_081(self):
        df = sma(self.volume, 21, 2)
        return df

    # alpha_gtja_82 SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1)
    def alpha_gtja_082(self):
        df = sma((ts_max(self.high, 6) - self.close) / (ts_max(self.high, 6) - ts_min(self.low, 6)) * 100, 20, 1)
        return df

    # alpha_gtja_83 (-1 * RANK(COVIANCE(RANK(HIGH), RANK(VOLUME), 5)))
    def alpha_gtja_083(self):
        df = (-1 * rank(covariance(rank(self.high), rank(self.volume), 5)))
        return df

    # alpha_gtja_84 SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),20)
    def alpha_gtja_084(self):
        return ts_sum(condition(self.close > delay(self.close, 1), self.volume,
                                condition(self.close < delay(self.close, 1), -self.volume, 0)), 20)

    # alpha_gtja_85 (TSRANK((VOLUME / MEAN(VOLUME,20)), 20) * TSRANK((-1 * DELTA(CLOSE, 7)), 8))
    def alpha_gtja_085(self):
        df = (ts_rank((self.volume / mean(self.volume, 20)), 20) * ts_rank((-1 * delta(self.close, 7)), 8))
        return df

    # alpha_gtja_86 ((0.25 < (((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10))) ? (-1 * 1) : (((((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10)) < 0) ? 1 : ((-1 * 1) * (CLOSE - DELAY(CLOSE, 1)))))
    def alpha_gtja_086(self):
        return condition(0.25 < (((delay(self.close, 20) - delay(self.close, 10)) / 10) - (
                (delay(self.close, 10) - self.close) / 10)), -1, condition(((((delay(self.close, 20) - delay(
            self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10)) < 0), 1, (-1 * (
                self.close - delay(self.close, 1)))))

    # alpha_gtja_88 (CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100
    def alpha_gtja_088(self):
        df = (self.close - delay(self.close, 20)) / delay(self.close, 20) * 100
        return df

    # alpha_gtja_89 2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))
    def alpha_gtja_089(self):
        df = 2 * (sma(self.close, 13, 2) - sma(self.close, 27, 2) - sma(sma(self.close, 13, 2) - sma(self.close, 27, 2),
                                                                        10, 2))
        return df

    # alpha_gtja_91 ((RANK((CLOSE - MAX(CLOSE, 5)))*RANK(CORR((MEAN(VOLUME,40)), LOW, 5))) * -1)
    def alpha_gtja_091(self):
        df = ((rank((self.close - MAX(self.close, 5))) * rank(correlation((mean(self.volume, 40)), self.low, 5))) * -1)
        return df

    # alpha_gtja_93 SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)
    def alpha_gtja_093(self):
        return ts_sum(condition(self.open >= delay(self.open, 1), 0,
                                MAX((self.open - self.low), (self.open - delay(self.open, 1)))), 20)

    # alpha_gtja_94 SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30)
    def alpha_gtja_094(self):
        return ts_sum(condition(self.close > delay(self.close, 1), self.volume,
                                condition(self.close < delay(self.close, 1), -self.volume, 0)), 30)

    # alpha_gtja_95 STD(AMOUNT,20)
    def alpha_gtja_095(self):
        # df=ts_std(AMOUNT,20)
        pass
        # return df

    # alpha_gtja_96 SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)
    def alpha_gtja_096(self):
        df = sma(sma((self.close - ts_min(self.low, 9)) / (ts_max(self.high, 9) - ts_min(self.low, 9)) * 100, 3, 1), 3,
                 1)
        return df

    # alpha_gtja_97 STD(VOLUME,10)
    def alpha_gtja_097(self):
        df = ts_std(self.volume, 10)
        return df

    # alpha_gtja_98 ((((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) < 0.05) || ((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) == 0.05)) ? (-1 * (CLOSE - TSMIN(CLOSE, 100))) : (-1 * DELTA(CLOSE, 3)))
    def alpha_gtja_098(self):
        return condition((((delta((ts_sum(self.close, 100) / 100), 100) / delay(self.close, 100)) < 0.05) | (
                (delta((ts_sum(self.close, 100) / 100), 100) / delay(self.close, 100)) == 0.05)),
                         (-1 * (self.close - ts_min(self.close, 100))), (-1 * delta(self.close, 3)))

    # alpha_gtja_99 (-1 * RANK(COVIANCE(RANK(CLOSE), RANK(VOLUME), 5)))
    def alpha_gtja_099(self):
        df = (-1 * rank(covariance(rank(self.close), rank(self.volume), 5)))
        return df

    # alpha_gtja_100 STD(VOLUME,20)
    def alpha_gtja_100(self):
        df = ts_std(self.volume, 20)
        return df

    # alpha_gtja_102 SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100
    def alpha_gtja_102(self):
        df = sma(MAX(self.volume - delay(self.volume, 1), 0), 6, 1) / sma(ABS(self.volume - delay(self.volume, 1)), 6,
                                                                          1) * 100
        return df

    # alpha_gtja_103 ((20-LOWDAY(LOW,20))/20)*100
    def alpha_gtja_103(self):
        df = ((20 - ts_argmin(self.low, 20)) / 20) * 100
        return df

    # alpha_gtja_104 (-1 * (DELTA(CORR(HIGH, VOLUME, 5), 5) * RANK(STD(CLOSE, 20))))
    def alpha_gtja_104(self):
        df = (-1 * (delta(correlation(self.high, self.volume, 5), 5) * rank(ts_std(self.close, 20))))
        return df

    # alpha_gtja_105 (-1 * CORR(RANK(OPEN), RANK(VOLUME), 10))
    def alpha_gtja_105(self):
        df = (-1 * correlation(rank(self.open), rank(self.volume), 10))
        return df

    # alpha_gtja_106 CLOSE-DELAY(CLOSE,20)
    def alpha_gtja_106(self):
        df = self.close - delay(self.close, 20)
        return df

    # alpha_gtja_107 (((-1 * RANK((OPEN - DELAY(HIGH, 1)))) * RANK((OPEN - DELAY(CLOSE, 1)))) * RANK((OPEN - DELAY(LOW, 1))))
    def alpha_gtja_107(self):
        df = (((-1 * rank((self.open - delay(self.high, 1)))) * rank((self.open - delay(self.close, 1)))) * rank(
            (self.open - delay(self.low, 1))))
        return df

    # alpha_gtja_109 SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)
    def alpha_gtja_109(self):
        df = sma(self.high - self.low, 10, 2) / sma(sma(self.high - self.low, 10, 2), 10, 2)
        return df

    # alpha_gtja_110 SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100
    def alpha_gtja_110(self):
        df = ts_sum(MAX(0, self.high - delay(self.close, 1)), 20) / ts_sum(MAX(0, delay(self.close, 1) - self.low),
                                                                           20) * 100
        return df

    # alpha_gtja_111 SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-L OW),4,2)
    def alpha_gtja_111(self):
        df = sma(self.volume * ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low), 11,
                 2) - sma(self.volume * ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low),
                          4, 2)
        return df

    # alpha_gtja_112  (SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)-SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOS E-DELAY(CLOSE,1)):0),12))/(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)+SUM((CLOSE-DE LAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))*100
    def alpha_gtja_112(self):
        return (ts_sum(condition((self.close - delay(self.close, 1)) > 0, self.close - delay(self.close, 1), 0),
                       12) - ts_sum(
            condition((self.close - delay(self.close, 1)) < 0, ABS(self.close - delay(self.close, 1)), 0), 12)) / (
                       ts_sum(
                           condition((self.close - delay(self.close, 1)) > 0, self.close - delay(self.close, 1), 0),
                           12) + ts_sum(
                   condition(self.close - delay(self.close, 1) < 0, ABS(self.close - delay(self.close, 1)), 0),
                   12)) * 100

    # alpha_gtja_113 (-1 * ((RANK((SUM(DELAY(CLOSE, 5), 20) / 20)) * CORR(CLOSE, VOLUME, 2)) * RANK(CORR(SUM(CLOSE, 5), SUM(CLOSE, 20), 2))))
    def alpha_gtja_113(self):
        df = (-1 * ((rank((ts_sum(delay(self.close, 5), 20) / 20)) * correlation(self.close, self.volume, 2)) * rank(
            correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2))))
        return df

    # alpha_gtja_115  (RANK(CORR(((HIGH * 0.9) + (CLOSE * 0.1)), MEAN(VOLUME,30), 10))^RANK(CORR(TSRANK(((HIGH + LOW) / 2), 4), TSRANK(VOLUME, 10), 7)))
    def alpha_gtja_115(self):
        return (rank(correlation(((self.high * 0.9) + (self.close * 0.1)), mean(self.volume, 30), 10)) ** rank(
            correlation(ts_rank(((self.high + self.low) / 2), 4), ts_rank(self.volume, 10), 7)))

    # alpha_gtja_116 REGBETA(CLOSE,SEQUENCE,20)
    def alpha_gtja_116(self):
        df = REGBETA(self.close, 20)
        return df

    # alpha_gtja_117 (TSRANK(VOLUME, 32) * (1 - TSRANK(((CLOSE + HIGH) - LOW), 16))) * (1 - TSRANK(RET, 32)))
    def alpha_gtja_117(self):
        df = (ts_rank(self.volume, 32) * (1 - ts_rank(((self.close + self.high) - self.low), 16))) * (
                1 - ts_rank(self.returns, 32))
        return df

    # alpha_gtja_118 SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100
    def alpha_gtja_118(self):
        df = ts_sum(self.high - self.open, 20) / ts_sum(self.open - self.low, 20) * 100
        return df

    # alpha_gtja_119 (RANK(DECAYLINEAR(CORR(VWAP, SUM(MEAN(VOLUME,5), 26), 5), 7)) - RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN), RANK(MEAN(VOLUME,15)), 21), 9), 7), 8)))
    def alpha_gtja_119(self):
        # df=(rank(decay_linear(correlation(VWAP, SUM(MEAN(VOLUME,5), 26), 5), 7)) - RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN), RANK(MEAN(VOLUME,15)), 21), 9), 7), 8)))
        pass
        # return df

    # alpha_gtja_122 (SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))/DELAY(SM A(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)
    def alpha_gtja_122(self):
        df = (sma(sma(sma(LOG(self.close), 13, 2), 13, 2), 13, 2) - delay(
            sma(sma(sma(LOG(self.close), 13, 2), 13, 2), 13, 2), 1)) / delay(
            sma(sma(sma(LOG(self.close), 13, 2), 13, 2), 13, 2), 1)
        return df

    # alpha_gtja_123 ((RANK(CORR(SUM(((HIGH + LOW) / 2), 20), SUM(MEAN(VOLUME,60), 20), 9)) < RANK(CORR(LOW, VOLUME, 6))) * -1)
    def alpha_gtja_123(self):
        df = ((rank(correlation(ts_sum(((self.high + self.low) / 2), 20), ts_sum(mean(self.volume, 60), 20), 9)) < rank(
            correlation(self.low, self.volume, 6))) * -1)
        return df

    # alpha_gtja_126 (CLOSE+HIGH+LOW)/3
    def alpha_gtja_126(self):
        df = (self.close + self.high + self.low) / 3
        return df

    # alpha_gtja_127 (MEAN((100*(CLOSE-MAX(CLOSE,12))/(MAX(CLOSE,12)))^2))^(1/2)
    def alpha_gtja_127(self):
        df = (mean((100 * (self.close - MAX(self.close, 12)) / (MAX(self.close, 12))) ** 2)) ** (1 / 2)
        return df

    # alpha_gtja_128  100-(100/(1+SUM(((HIGH+LOW+CLOSE)/3>DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUM E:0),14)/SUM(((HIGH+LOW+CLOSE)/3<DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0), 14)))
    def alpha_gtja_128(self):
        return 100 - (100 / (1 + ts_sum(
            condition((self.high + self.low + self.close) / 3 > delay((self.high + self.low + self.close) / 3, 1),
                      (self.high + self.low + self.close) / 3 * self.volume, 0), 14) / ts_sum(
            condition((self.high + self.low + self.close) / 3 < delay((self.high + self.low + self.close) / 3, 1),
                      (self.high + self.low + self.close) / 3 * self.volume, 0), 14)))

    # alpha_gtja_133 ((20-HIGHDAY(HIGH,20))/20)*100-((20-LOWDAY(LOW,20))/20)*100
    def alpha_gtja_133(self):
        df = ((20 - ts_argmax(self.high, 20)) / 20) * 100 - ((20 - ts_argmin(self.low, 20)) / 20) * 100
        return df

    # alpha_gtja_134 (CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME
    def alpha_gtja_134(self):
        df = (self.close - delay(self.close, 12)) / delay(self.close, 12) * self.volume
        return df

    # alpha_gtja_135 SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)
    def alpha_gtja_135(self):
        df = sma(delay(self.close / delay(self.close, 20), 1), 20, 1)
        return df

    # alpha_gtja_136 ((-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10))
    def alpha_gtja_136(self):
        df = ((-1 * rank(delta(self.returns, 3))) * correlation(self.open, self.volume, 10))
        return df

    # alpha_gtja_109 (-1 * CORR(OPEN, VOLUME, 10))
    def alpha_gtja_139(self):
        df = (-1 * correlation(self.open, self.volume, 10))
        return df

    # alpha_gtja_140 MIN(RANK(DECAYLINEAR(((RANK(OPEN) + RANK(LOW)) - (RANK(HIGH) + RANK(CLOSE))), 8)), TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE, 8), TSRANK(MEAN(VOLUME,60), 20), 8), 7), 3))
    def alpha_gtja_140(self):
        df = MIN(rank(decay_linear(((rank(self.open) + rank(self.low)) - (rank(self.high) + rank(self.close))), 8)),
                 ts_rank(decay_linear(correlation(ts_rank(self.close, 8), ts_rank(mean(self.volume, 60), 20), 8), 7),
                         3))
        return df

    # alpha_gtja_141 (RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME,15)), 9))* -1)
    def alpha_gtja_141(self):
        df = (rank(correlation(rank(self.high), rank(mean(self.volume, 15)), 9)) * -1)
        return df

    # alpha_gtja_142 (((-1 * RANK(TSRANK(CLOSE, 10))) * RANK(DELTA(DELTA(CLOSE, 1), 1))) * RANK(TSRANK((VOLUME /MEAN(VOLUME,20)), 5)))
    def alpha_gtja_142(self):
        df = (((-1 * rank(ts_rank(self.close, 10))) * rank(delta(delta(self.close, 1), 1))) * rank(
            ts_rank((self.volume / mean(self.volume, 20)), 5)))
        return df

    # alpha_gtja_145 (MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100
    def alpha_gtja_145(self):
        df = (mean(self.volume, 9) - mean(self.volume, 26)) / mean(self.volume, 12) * 100
        return df

    # alpha_gtja_146  REGBETA(MEAN(CLOSE,12),SEQUENCE(12))
    def alpha_gtja_146(self):
        return REGBETA(mean(self.close, 12), 12)

    # alpha_gtja_147  REGBETA(MEAN(CLOSE,12),SEQUENCE(12))
    def alpha_gtja_147(self):
        return REGBETA(mean(self.close, 12), 12)

    # alpha_gtja_148 ((RANK(CORR((OPEN), SUM(MEAN(VOLUME,60), 9), 6)) < RANK((OPEN - TSMIN(OPEN, 14)))) * -1)
    def alpha_gtja_148(self):
        df = ((rank(correlation((self.open), ts_sum(mean(self.volume, 60), 9), 6)) < rank(
            (self.open - ts_min(self.open, 14)))) * -1)
        return df

    ##################################################################
    # (CLOSE+HIGH+LOW)/3*VOLUME
    def alpha_gtja_150(self):
        return (self.close + self.high + self.low) / 3 * self.volume

    ##################################################################
    # SMA(CLOSE -DELAY(CLOSE,20),20,1)
    def alpha_gtja_151(self):
        return sma(self.close - delay(self.close, 20), 20, 1)

    ##################################################################
    # SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26),
    def alpha_gtja_152(self):
        return sma(mean(delay(sma(delay(self.close / delay(self.close, 9), 1), 9, 1), 1), 12) - mean(
            delay(sma(delay(self.close / delay(self.close, 9), 1), 9, 1), 1), 26), 9, 1)

    ##################################################################
    # (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4
    def alpha_gtja_153(self):
        return (mean(self.close, 3) + mean(self.close, 6) + mean(self.close, 12) + mean(self.close, 24)) / 4

    ##################################################################
    # (((VWAP - MIN(VWAP, 16))) < (CORR(VWAP, MEAN(VOLUME,180), 18)))
    # def alpha_gtja_154(self):

    ##################################################################
    # SMA(VOLUME,13,2) - SMA(VOLUME,27,2) - SMA(SMA(VOLUME,13,2) - SMA(VOLUME,27,2),10,2)
    def alpha_gtja_155(self):
        return sma(self.volume, 13, 2) - sma(self.volume, 27, 2) - sma(
            sma(self.volume, 13, 2) - sma(self.volume, 27, 2), 10, 2)

    ##################################################################
    # (MAX(RANK(DECAYLINEAR(DELTA(VWAP, 5), 3)), RANK(DECAYLINEAR(((DELTA(((OPEN * 0.15) + (LOW *0.85)), 2) / ((OPEN * 0.15) + (LOW * 0.85))) * -1), 3))) * -1)
    # def alpha_gtja_156(self):

    ##################################################################

    # (MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1 * RANK(DELTA((CLOSE - 1), 5))))), 2), 1)))), 1), 5) + TSRANK(DELAY((-1 * RET), 6), 5))
    def alpha_gtja_157(self):
        ret = self.returns
        return MIN(
            PROD(rank(rank(np.log(ts_sum(ts_min(rank(rank(-1 * rank(delta(self.close - 1, 5)))), 2), 5)))), 1),
            5) + ts_rank(delay(-1 * ret, 6), 5)

    ##################################################################
    # ((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE
    def alpha_gtja_158(self):
        return ((self.high - sma(self.close, 15, 2)) - (self.low - sma(self.close, 15, 2))) / self.close

    ##################################################################
    # ((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6) *12*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),12))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CL OSE,1)),12)*6*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),24))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,D ELAY(CLOSE,1)),24)*6*24)*100/(6*12+6*24+12*24)
    def alpha_gtja_159(self):
        return ((self.close - ts_sum(MIN(self.low, delay(self.close, 1)), 6)) / ts_sum(
            MAX(self.high, delay(self.close, 1)) - MIN(self.low, delay(self.close, 1)), 6) * 12 * 24 + (
                        self.close - ts_sum(MIN(self.low, delay(self.close, 1)), 12)) / ts_sum(
            MAX(self.high, delay(self.close, 1)) - MIN(self.low, delay(self.close, 1)), 12) * 6 * 24 + (
                        self.close - ts_sum(MIN(self.low, delay(self.close, 1)), 24)) / ts_sum(
            MAX(self.high, delay(self.close, 1)) - MIN(self.low, delay(self.close, 1)),
            24) * 6 * 24) * 100 / (6 * 12 + 6 * 24 + 12 * 24)

    ##################################################################
    # SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
    def alpha_gtja_160(self):
        return sma(condition(self.close <= delay(self.close, 1), ts_std(self.close, 20), 0), 20, 1)

    ##################################################################
    # MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)
    def alpha_gtja_161(self):
        return mean(MAX(MAX(self.high - self.low, (delay(self.close, 1) - self.high).abs()),
                        (delay(self.close, 1) - self.low).abs()), 12)

    ##################################################################
    # (SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))/(MAX(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12)-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12, 1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))
    def alpha_gtja_162(self):
        return (sma(MAX(self.close - delay(self.close, 1), 0), 12, 1) / sma(
            (self.close - delay(self.close, 1)).abs(), 12, 1) * 100 - MIN(
            sma(MAX(self.close - delay(self.close, 1), 0), 12, 1) / sma(
                (self.close - delay(self.close, 1)).abs(), 12, 1) * 100, 12)) / (MAX(
            sma(MAX(self.close - delay(self.close, 1), 0), 12, 1) / (
                    sma((self.close - delay(self.close, 1)).abs(), 12, 1) * 100), 12) - MIN(
            sma(MAX(self.close - delay(self.close, 1), 0), 12, 1) / (
                    sma((self.close - delay(self.close, 1)).abs(), 12, 1) * 100), 12))

    ##################################################################
    # RANK(((((-1 * RET) * MEAN(VOLUME,20)) * VWAP) * (HIGH - CLOSE)))
    # def alpha_gtja_163(self):

    ##################################################################
    # SMA((((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1)-MIN(((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1),12))/(HIGH-LOW)*100,13,2)
    def alpha_gtja_164(self):
        return sma((condition(self.close > delay(self.close, 1), 1 / (self.close - delay(self.close, 1)), 1) - MIN(
            condition(self.close > delay(self.close, 1), 1 / (self.close - delay(self.close, 1)), 1), 12)) / (
                           self.high - self.low) * 100, 13, 2)

    ##################################################################
    # MAX(SUMAC(CLOSE-MEAN(CLOSE,48)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,48)))/STD(CLOSE,48)
    # def alpha_gtja_165(self):

    ##################################################################
    # -20* ( 20-1 ) ^1.5*SUM(CLOSE/DELAY(CLOSE,1)-1-MEAN(CLOSE/DELAY(CLOSE,1)-1,20),20)/((20-1)*(20-2)(SUM((CLOSE/DELAY(CLOSE,1),20)^2,20))^1.5)
    def alpha_gtja_166(self):
        return -20 * (20 - 1) ** 1.5 * ts_sum(
            self.close / delay(self.close, 1) - 1 - mean(self.close / delay(self.close, 1) - 1, 20)) / (
                       (20 - 1) * (20 - 2) * (ts_sum((self.close / delay(self.close, 1)) ** 2, 20)) ** 1.5)

    ##################################################################
    # SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)
    def alpha_gtja_167(self):
        alpha = pd.DataFrame(index=self.price.index)
        alpha['alpha'] = np.nan
        condition = ((self.close - delay(self.close, 1)) > 0)
        alpha['alpha'][condition] = (self.close - delay(self.close, 1))[condition]
        return ts_sum(alpha['alpha'].fillna(0), 12)

    ##################################################################
    # (-1*VOLUME/MEAN(VOLUME,20))
    def alpha_gtja_168(self):
        return -1 * self.volume / mean(self.volume, 20)

    ##################################################################
    # SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)-MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1), 26),10,1)
    def alpha_gtja_169(self):
        return sma(mean(delay(sma(self.close - delay(self.close, 1), 9, 1), 1), 12) - mean(
            delay(sma(self.close - delay(self.close, 1), 9, 1), 1), 26), 10, 1)

    ##################################################################
    # ((((RANK((1 / CLOSE)) * VOLUME) / MEAN(VOLUME,20)) * ((HIGH * RANK((HIGH - CLOSE))) / (SUM(HIGH, 5) / 5))) - RANK((VWAP - DELAY(VWAP, 5))))
    # def alpha_gtja_170(self):
    #     rank(1/self.close) * self.volume / mean(self.volume,20) * (self.high * rank(self.high-self.close)) / (ts_sum(self.high,5)/5) - rank()

    ##################################################################
    # ((-1 * ((LOW - CLOSE) * (OPEN^5))) / ((CLOSE - HIGH) * (CLOSE^5)))
    def alpha_gtja_171(self):
        return (-1 * (self.low - self.close) * self.open ** 5) / (self.close - self.high * self.close ** 5)

    ##################################################################
    # MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)
    def alpha_gtja_172(self):
        LD = delay(self.low, 1) - self.low
        HD = self.high - delay(self.high, 1)
        TR = MAX(MAX(self.high - self.low, ABS(self.high - delay(self.close, 1))), ABS(self.low - delay(self.close, 1)))
        return mean(ABS(ts_sum(condition((LD > 0 and LD > HD), LD, 0), 14) * 100 / ts_sum(TR, 14) - ts_sum(
            condition((HD > 0 and HD > LD), HD, 0), 14) * 100 / ts_sum(TR, 14)) / ts_sum(
            condition((LD > 0 and LD > HD), LD, 0), 14) * 100 / ts_sum(TR, 14) + ts_sum(
            condition((HD > 0 and HD > LD), HD, 0), 14) * 100 / ts_sum(TR, 14) * 100, 6)

    ##################################################################
    # 3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)
    def alpha_gtja_173(self):
        return 3 * sma(self.close, 13, 2) - 2 * sma(sma(self.close, 13, 2), 13, 2) + sma(
            sma(sma(np.log(self.close), 13, 2), 13, 2), 13, 2)

    ##################################################################
    # SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
    def alpha_gtja_174(self):
        return sma(condition(self.close > delay(self.close, 1), ts_std(self.close, 20), 0), 20, 1)

    ##################################################################
    # MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)
    def alpha_gtja_175(self):
        return mean(MAX(MAX(self.high - self.low, (delay(self.close, 1) - self.high).abs()),
                        (delay(self.close - 1) - self.low).abs()), 6)

    ##################################################################
    # CORR(RANK(((CLOSE - TSMIN(LOW, 12)) / (TSMAX(HIGH, 12) - TSMIN(LOW,12)))), RANK(VOLUME), 6)
    def alpha_gtja_176(self):
        # x = (rank((self.close - ts_min(self.low, 12)) / (ts_max(self.high, 12) - ts_min(self.low, 12))))
        # print(x)
        # y = rank(self.volume)
        # print(y)
        # print(correlation(x,y,6))
        return correlation((rank((self.close - ts_min(self.low, 12)) / (ts_max(self.high, 12) - ts_min(self.low, 12)))),
                           rank(self.volume), 6)

    ##################################################################
    # ((20-HIGHDAY(HIGH,20))/20)*100
    def alpha_gtja_177(self):
        return (20 - ts_argmax(self.high, 20)) / 20 * 100

    ##################################################################
    # (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME
    def alpha_gtja_178(self):
        return (self.close - delay(self.close, 1)) / delay(self.close, 1) * self.volume

    ##################################################################
    # (RANK(CORR(VWAP, VOLUME, 4)) *RANK(CORR(RANK(LOW), RANK(MEAN(VOLUME,50)), 12)))
    # def alpha_gtja_179(self):

    ##################################################################
    # ((MEAN(VOLUME,20) < VOLUME) ? ((-1 * TSRANK(ABS(DELTA(CLOSE, 7)), 60)) * SIGN(DELTA(CLOSE, 7)) : (-1 * VOLUME)))
    def alpha_gtja_180(self):
        return condition(mean(self.volume, 20) < self.volume,
                         (-1 * ts_rank(delta(self.close, 7).abs(), 60) * SIGN(delta(self.close, 7))), -1 * self.volume)

    ##################################################################
    # SUM(((CLOSE/DELAY(CLOSE,1)-1)-MEAN((CLOSE/DELAY(CLOSE,1)-1),20))-(BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^2,20)/SUM((BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^3)
    def alpha_gtja_181(self):
        benchmark_close = self.price.Benchmark_Close
        alpha = ts_sum(((self.close / delay(self.close, 1) - 1) - mean(self.close / delay(self.close, 1) - 1, 20)) - (
                benchmark_close - mean(benchmark_close, 20)) ** 2, 20) / ts_sum(
            (benchmark_close - mean(benchmark_close, 20)) ** 3, 20)
        return alpha

    ##################################################################
    # COUNT((CLOSE>OPEN & BANCHMARKINDEXCLOSE>BANCHMARKINDEXOPEN)OR(CLOSE<OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN),20)/20
    def alpha_gtja_182(self):
        benchmark_close = self.price.Benchmark_Close
        benchmark_open = self.price.Benchmark_Open
        condition = (((self.close > self.open) and (benchmark_close > benchmark_open)) or (
                (self.close < self.open) and (benchmark_close < benchmark_open)))
        return COUNT(self.price[condition]) / 20

    ##################################################################
    # COUNT((CLOSE>OPEN & BANCHMARKINDEXCLOSE>BANCHMARKINDEXOPEN)OR(CLOSE<OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN),20)/20
    # def alpha_gtja_183(self):

    ##################################################################
    # (RANK(CORR(DELAY((OPEN - CLOSE), 1), CLOSE, 200)) + RANK((OPEN - CLOSE)))
    def alpha_gtja_184(self):
        return rank(correlation(delay(self.open - self.close, 1), self.close, 200)) + rank(
            self.open - self.close)

    ##################################################################
    # RANK((-1 * ((1 - (OPEN / CLOSE))^2)))
    def alpha_gtja_185(self):
        return rank(-1 * (1 - self.open / self.close) ** 2)

    ##################################################################
    # (MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)+DELAY(MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6),6))/2
    # def alpha_gtja_186(self):

    ##################################################################
    # SUM((OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20)
    def alpha_gtja_187(self):
        df = ts_sum(condition((self.open <= delta(self.open, 1)), 0,
                              MAX(self.high - self.open, self.open - delay(self.open, 1))), 20)
        return df

    ##################################################################
    # ((HIGH-LOW–SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100
    def alpha_gtja_188(self):
        return (self.high - self.low - sma(self.high - self.low, 11, 2)) / sma(self.high - self.low, 11, 2) * 100

    ##################################################################
    # MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)
    def alpha_gtja_189(self):
        return mean((self.close - mean(self.close, 6)).abs(), 6)

    ##################################################################
    # LOG((COUNT(CLOSE/DELAY(CLOSE)-1>((CLOSE/DELAY(CLOSE,19))^(1/20)-1),20)-1)*(SUMIF(((CLOSE/DELAY(C LOSE)-1-(CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)- 1))/((COUNT((CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1),20))*(SUMIF((CLOSE/DELAY(CLOS E)-1-((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1>(CLOSE/DELAY(CLOSE,19))^(1/20)-1))) )
    def alpha_gtja_190(self):
        return LOG((COUNT((self.close / delay(self.close) - 1) > ((self.close / delay(self.close, 19)) ** (1 / 20) - 1),
                          20) - 1) * (SUMIF(
            (self.close / delay(self.close) - 1 - (self.close / delay(self.close, 19)) ** (1 / 20) - 1) ** 2, 20,
            (self.close / delay(self.close) - 1) < ((self.close / delay(self.close, 19)) ** (1 / 20) - 1))) / ((COUNT(
            (self.close / delay(self.close) - 1) < ((self.close / delay(self.close, 19)) ** (1 / 20) - 1), 20)) * (
                                                                                                                   SUMIF(
                                                                                                                       (
                                                                                                                               self.close / delay(
                                                                                                                           self.close) - 1 - (
                                                                                                                                       (
                                                                                                                                               self.close / delay(
                                                                                                                                           self.close,
                                                                                                                                           19)) ** (
                                                                                                                                               1 / 20) - 1)) ** 2,
                                                                                                                       20,
                                                                                                                       (
                                                                                                                               self.close / delay(
                                                                                                                           self.close) - 1) > (
                                                                                                                               (
                                                                                                                                       self.close / delay(
                                                                                                                                   self.close,
                                                                                                                                   19)) ** (
                                                                                                                                       1 / 20) - 1)))))

    ##################################################################
    # ((CORR(MEAN(VOLUME,20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE)
    def alpha_gtja_191(self):
        return correlation(mean(self.volume, 20), self.low, 5) + (self.high + self.close) / 2 - self.close

    def calculate(self):
        # alpha_2 = self.alpha_gtja_recommendation_2()
        # alpha_3 = self.alpha_gtja_recommendation_3()
        # alpha_4 = self.alpha_gtja_recommendation_4()
        # alpha_ql_003 = self.alpha_ql_003()
        # alpha_ql_004 = self.alpha_ql_004()
        # alpha_ql_005 = self.alpha_ql_005()
        # alpha_ql_006 = self.alpha_ql_006()
        # alpha_ql_007 = self.alpha_ql_007()
        # alpha_ql_008 = self.alpha_ql_008()
        # alpha_ql_009 = self.alpha_ql_009()
        # alpha_ql_010 = self.alpha_ql_010()
        # alpha_ql_011 = self.alpha_ql_011()
        # alpha_ql_012 = self.alpha_ql_012()
        # alpha_ql_013 = self.alpha_ql_013()
        # alpha_ql_014 = self.alpha_ql_014()
        # alpha_ql_015 = self.alpha_ql_015()
        # alpha_ql_016 = self.alpha_ql_016()
        # alpha_ql_017 = self.alpha_ql_017()
        # alpha_ql_018 = self.alpha_ql_018()
        # alpha_ql_019 = self.alpha_ql_019()
        # alpha_gtja_001 = self.alpha_gtja_001()
        # alpha_gtja_002 = self.alpha_gtja_002()
        # alpha_gtja_003 = self.alpha_gtja_003()
        # alpha_gtja_004 = self.alpha_gtja_004()
        # alpha_gtja_005 = self.alpha_gtja_005()
        # alpha_gtja_006 = self.alpha_gtja_006()
        # alpha_gtja_009 = self.alpha_gtja_009()
        # alpha_gtja_010 = self.alpha_gtja_010()
        # alpha_gtja_011 = self.alpha_gtja_011()
        # alpha_gtja_014 = self.alpha_gtja_014()
        # alpha_gtja_015 = self.alpha_gtja_015()
        # alpha_gtja_018 = self.alpha_gtja_018()
        # alpha_gtja_019 = self.alpha_gtja_019()
        # alpha_gtja_020 = self.alpha_gtja_020()
        # alpha_gtja_021 = self.alpha_gtja_021()
        # alpha_gtja_022 = self.alpha_gtja_022()
        # alpha_gtja_024 = self.alpha_gtja_024()
        # alpha_gtja_027 = self.alpha_gtja_027()
        # alpha_gtja_028 = self.alpha_gtja_028()
        # alpha_gtja_029 = self.alpha_gtja_029()
        # alpha_gtja_031 = self.alpha_gtja_031()
        # alpha_gtja_032 = self.alpha_gtja_032()
        # alpha_gtja_033 = self.alpha_gtja_033()
        # alpha_gtja_034 = self.alpha_gtja_034()
        # alpha_gtja_037 = self.alpha_gtja_037()
        # alpha_gtja_038 = self.alpha_gtja_038()
        # alpha_gtja_040 = self.alpha_gtja_040()
        # alpha_gtja_042 = self.alpha_gtja_042()
        # alpha_gtja_046 = self.alpha_gtja_046()
        # alpha_gtja_047 = self.alpha_gtja_047()
        # alpha_gtja_048 = self.alpha_gtja_048()
        # alpha_gtja_049 = self.alpha_gtja_049()
        # alpha_gtja_050 = self.alpha_gtja_050()
        # alpha_gtja_051 = self.alpha_gtja_051()
        # alpha_gtja_052 = self.alpha_gtja_052()
        # alpha_gtja_053 = self.alpha_gtja_053()
        # alpha_gtja_054 = self.alpha_gtja_054()
        # alpha_gtja_057 = self.alpha_gtja_057()
        # alpha_gtja_058 = self.alpha_gtja_058()
        # alpha_gtja_059 = self.alpha_gtja_059()
        # alpha_gtja_060 = self.alpha_gtja_060()
        # alpha_gtja_062 = self.alpha_gtja_062()
        # alpha_gtja_063 = self.alpha_gtja_063()
        # alpha_gtja_064 = self.alpha_gtja_064()
        # alpha_gtja_065 = self.alpha_gtja_065()
        # alpha_gtja_066 = self.alpha_gtja_066()
        # alpha_gtja_067 = self.alpha_gtja_067()
        # alpha_gtja_068 = self.alpha_gtja_068()
        # alpha_gtja_069 = self.alpha_gtja_069()
        # alpha_gtja_071 = self.alpha_gtja_071()
        # alpha_gtja_072 = self.alpha_gtja_072()
        # alpha_gtja_076 = self.alpha_gtja_076()
        # alpha_gtja_078 = self.alpha_gtja_078()
        # alpha_gtja_079 = self.alpha_gtja_079()
        # alpha_gtja_080 = self.alpha_gtja_080()
        # alpha_gtja_081 = self.alpha_gtja_081()
        # alpha_gtja_082 = self.alpha_gtja_082()
        # alpha_gtja_083 = self.alpha_gtja_083()
        # alpha_gtja_084 = self.alpha_gtja_084()
        # alpha_gtja_085 = self.alpha_gtja_085()
        # alpha_gtja_086 = self.alpha_gtja_086()
        # alpha_gtja_088 = self.alpha_gtja_088()
        # alpha_gtja_089 = self.alpha_gtja_089()
        # alpha_gtja_091 = self.alpha_gtja_091()
        # alpha_gtja_093 = self.alpha_gtja_093()
        # alpha_gtja_094 = self.alpha_gtja_094()
        # alpha_gtja_096 = self.alpha_gtja_096()
        # alpha_gtja_097 = self.alpha_gtja_097()
        # alpha_gtja_098 = self.alpha_gtja_098()
        # alpha_gtja_099 = self.alpha_gtja_099()
        # alpha_gtja_100 = self.alpha_gtja_100()
        # alpha_gtja_102 = self.alpha_gtja_102()
        # alpha_gtja_103 = self.alpha_gtja_103()
        # alpha_gtja_104 = self.alpha_gtja_104()
        # alpha_gtja_105 = self.alpha_gtja_105()
        # alpha_gtja_106 = self.alpha_gtja_106()
        # alpha_gtja_107 = self.alpha_gtja_107()
        # alpha_gtja_109 = self.alpha_gtja_109()
        # alpha_gtja_110 = self.alpha_gtja_110()
        # alpha_gtja_111 = self.alpha_gtja_111()
        # alpha_gtja_112 = self.alpha_gtja_112()
        # alpha_gtja_113 = self.alpha_gtja_113()
        # alpha_gtja_115 = self.alpha_gtja_115()
        # alpha_gtja_116 = self.alpha_gtja_116()
        # alpha_gtja_117 = self.alpha_gtja_117()
        # alpha_gtja_118 = self.alpha_gtja_118()
        # alpha_gtja_122 = self.alpha_gtja_122()
        # alpha_gtja_123 = self.alpha_gtja_123()
        # alpha_gtja_126 = self.alpha_gtja_126()
        # alpha_gtja_127 = self.alpha_gtja_127()
        # alpha_gtja_128 = self.alpha_gtja_128()
        # alpha_gtja_133 = self.alpha_gtja_133()
        # alpha_gtja_134 = self.alpha_gtja_134()
        # alpha_gtja_135 = self.alpha_gtja_135()
        # alpha_gtja_136 = self.alpha_gtja_136()
        # alpha_gtja_139 = self.alpha_gtja_139()
        # alpha_gtja_140 = self.alpha_gtja_140()
        # alpha_gtja_141 = self.alpha_gtja_141()
        # alpha_gtja_142 = self.alpha_gtja_142()
        # alpha_gtja_145 = self.alpha_gtja_145()
        # alpha_gtja_146 = self.alpha_gtja_146()
        # alpha_gtja_147 = self.alpha_gtja_147()
        # alpha_gtja_148 = self.alpha_gtja_148()
        # alpha_gtja_150 = self.alpha_gtja_150()
        # alpha_gtja_151 = self.alpha_gtja_151()
        # alpha_gtja_152 = self.alpha_gtja_152()
        # alpha_gtja_153 = self.alpha_gtja_153()
        # alpha_gtja_155 = self.alpha_gtja_155()
        # alpha_gtja_157 = self.alpha_gtja_157()
        # alpha_gtja_158 = self.alpha_gtja_158()
        # alpha_gtja_159 = self.alpha_gtja_159()
        # alpha_gtja_160 = self.alpha_gtja_160()
        # alpha_gtja_161 = self.alpha_gtja_161()
        # alpha_gtja_162 = self.alpha_gtja_162()
        # alpha_gtja_164 = self.alpha_gtja_164()
        # alpha_gtja_166 = self.alpha_gtja_166()
        # alpha_gtja_167 = self.alpha_gtja_167()
        # alpha_gtja_168 = self.alpha_gtja_168()
        # alpha_gtja_169 = self.alpha_gtja_169()
        # alpha_gtja_171 = self.alpha_gtja_171()
        # alpha_gtja_173 = self.alpha_gtja_173()
        # alpha_gtja_174 = self.alpha_gtja_174()
        # alpha_gtja_175 = self.alpha_gtja_175()
        # alpha_gtja_176 = self.alpha_gtja_176()
        # alpha_gtja_177 = self.alpha_gtja_177()
        # alpha_gtja_178 = self.alpha_gtja_178()
        # alpha_gtja_180 = self.alpha_gtja_180()
        # # alpha_gtja_181 = self.alpha_gtja_181()
        # # alpha_gtja_182 = self.alpha_gtja_182()
        # alpha_gtja_184 = self.alpha_gtja_184()
        # alpha_gtja_185 = self.alpha_gtja_185()
        # alpha_gtja_187 = self.alpha_gtja_187()
        # alpha_gtja_188 = self.alpha_gtja_188()
        # alpha_gtja_189 = self.alpha_gtja_189()
        # alpha_gtja_190 = self.alpha_gtja_190()
        # alpha_gtja_191 = self.alpha_gtja_191()
        pn = pd.DataFrame(
            {#'alpha_2': self.alpha_gtja_recommendation_2()
                # , 'alpha_3': self.alpha_gtja_recommendation_3()
                # , 'alpha_4': self.alpha_gtja_recommendation_4()
                # , 'alpha_ql_003': self.alpha_ql_003()
                # , 'alpha_ql_004': self.alpha_ql_004()
                # , 'alpha_ql_005': self.alpha_ql_005()
                # , 'alpha_ql_006': self.alpha_ql_006()
                # , 'alpha_ql_007': self.alpha_ql_007()
                # , 'alpha_ql_008': self.alpha_ql_008()
                # , 'alpha_ql_009': self.alpha_ql_009()
                # , 'alpha_ql_010': self.alpha_ql_010()
                # , 'alpha_ql_011': self.alpha_ql_011()
                # , 'alpha_ql_012': self.alpha_ql_012()
                # , 'alpha_ql_013': self.alpha_ql_013()
                # , 'alpha_ql_014': self.alpha_ql_014()
                # , 'alpha_ql_015': self.alpha_ql_015()
                # , 'alpha_ql_016': self.alpha_ql_016()
                # , 'alpha_ql_017': self.alpha_ql_017()
                # , 'alpha_ql_018': self.alpha_ql_018()
                # , 'alpha_ql_019': self.alpha_ql_019()
                # , 'alpha_gtja_001': self.alpha_gtja_001()
                # , 'alpha_gtja_002': self.alpha_gtja_002()
                # , 'alpha_gtja_003': self.alpha_gtja_003()
                # , 'alpha_gtja_004': self.alpha_gtja_004()
                # , 'alpha_gtja_005': self.alpha_gtja_005()
                # , 'alpha_gtja_006': self.alpha_gtja_006()
                # , 'alpha_gtja_009': self.alpha_gtja_009()
                # , 'alpha_gtja_010': self.alpha_gtja_010()
                # , 'alpha_gtja_011': self.alpha_gtja_011()
                # , 'alpha_gtja_014': self.alpha_gtja_014()
                # , 'alpha_gtja_015': self.alpha_gtja_015()
                # , 'alpha_gtja_018': self.alpha_gtja_018()
                # , 'alpha_gtja_019': self.alpha_gtja_019()
                # , 'alpha_gtja_020': self.alpha_gtja_020()
                # , 'alpha_gtja_021': self.alpha_gtja_021()
                # , 'alpha_gtja_022': self.alpha_gtja_022()
                # , 'alpha_gtja_024': self.alpha_gtja_024()
                # , 'alpha_gtja_027': self.alpha_gtja_027()
                # , 'alpha_gtja_028': self.alpha_gtja_028()
                # , 'alpha_gtja_029': self.alpha_gtja_029()
                # , 'alpha_gtja_031': self.alpha_gtja_031()
                # , 'alpha_gtja_032': self.alpha_gtja_032()
                # , 'alpha_gtja_033': self.alpha_gtja_033()
                # , 'alpha_gtja_034': self.alpha_gtja_034()
                # , 'alpha_gtja_037': self.alpha_gtja_037()
                # , 'alpha_gtja_038': self.alpha_gtja_038()
                # , 'alpha_gtja_040': self.alpha_gtja_040()
                # , 'alpha_gtja_042': self.alpha_gtja_042()
                # , 'alpha_gtja_046': self.alpha_gtja_046()
                # , 'alpha_gtja_047': self.alpha_gtja_047()
                # , 'alpha_gtja_048': self.alpha_gtja_048()
                # , 'alpha_gtja_049': self.alpha_gtja_049()
                # , 'alpha_gtja_050': self.alpha_gtja_050()
                # , 'alpha_gtja_051': self.alpha_gtja_051()
                # , 'alpha_gtja_052': self.alpha_gtja_052()
                # , 'alpha_gtja_053': self.alpha_gtja_053()
                # , 'alpha_gtja_054': self.alpha_gtja_054()
                # , 'alpha_gtja_057': self.alpha_gtja_057()
                # , 'alpha_gtja_058': self.alpha_gtja_058()
                # , 'alpha_gtja_059': self.alpha_gtja_059()
                # , 'alpha_gtja_060': self.alpha_gtja_060()
                # , 'alpha_gtja_062': self.alpha_gtja_062()
                # , 'alpha_gtja_063': self.alpha_gtja_063()
                # , 'alpha_gtja_064': self.alpha_gtja_064()
                # , 'alpha_gtja_065': self.alpha_gtja_065()
                # , 'alpha_gtja_066': self.alpha_gtja_066()
                # , 'alpha_gtja_067': self.alpha_gtja_067()
                # , 'alpha_gtja_068': self.alpha_gtja_068()
                # , 'alpha_gtja_069': self.alpha_gtja_069()
                # , 'alpha_gtja_071': self.alpha_gtja_071()
                # , 'alpha_gtja_072': self.alpha_gtja_072()
                # , 'alpha_gtja_076': self.alpha_gtja_076()
                # , 'alpha_gtja_078': self.alpha_gtja_078()
                # , 'alpha_gtja_079': self.alpha_gtja_079()
                # , 'alpha_gtja_080': self.alpha_gtja_080()
                # , 'alpha_gtja_081': self.alpha_gtja_081()
                # , 'alpha_gtja_082': self.alpha_gtja_082()
                # , 'alpha_gtja_083': self.alpha_gtja_083()
                # , 'alpha_gtja_084': self.alpha_gtja_084()
                # , 'alpha_gtja_085': self.alpha_gtja_085()
                # , 'alpha_gtja_086': self.alpha_gtja_086()
                # , 'alpha_gtja_088': self.alpha_gtja_088()
                # , 'alpha_gtja_089': self.alpha_gtja_089()
                # , 'alpha_gtja_091': self.alpha_gtja_091()
                # , 'alpha_gtja_093': self.alpha_gtja_093()
                # , 'alpha_gtja_094': self.alpha_gtja_094()
                # , 'alpha_gtja_096': self.alpha_gtja_096()
                # , 'alpha_gtja_097': self.alpha_gtja_097()
                # , 'alpha_gtja_098': self.alpha_gtja_098()
                # , 'alpha_gtja_099': self.alpha_gtja_099()
                # , 'alpha_gtja_100': self.alpha_gtja_100()
                # , 'alpha_gtja_102': self.alpha_gtja_102()
                # , 'alpha_gtja_103': self.alpha_gtja_103()
                # , 'alpha_gtja_104': self.alpha_gtja_104()
                # , 'alpha_gtja_105': self.alpha_gtja_105()
                # , 'alpha_gtja_106': self.alpha_gtja_106()
                # , 'alpha_gtja_107': self.alpha_gtja_107()
                # , 'alpha_gtja_109': self.alpha_gtja_109()
             # separate
                 'alpha_gtja_110': self.alpha_gtja_110()
                , 'alpha_gtja_111': self.alpha_gtja_111()
                , 'alpha_gtja_112': self.alpha_gtja_112()
                , 'alpha_gtja_113': self.alpha_gtja_113()
                , 'alpha_gtja_115': self.alpha_gtja_115()
                , 'alpha_gtja_116': self.alpha_gtja_116()
                , 'alpha_gtja_117': self.alpha_gtja_117()
                , 'alpha_gtja_118': self.alpha_gtja_118()
                , 'alpha_gtja_122': self.alpha_gtja_122()
                , 'alpha_gtja_123': self.alpha_gtja_123()
                , 'alpha_gtja_126': self.alpha_gtja_126()
                , 'alpha_gtja_127': self.alpha_gtja_127()
                , 'alpha_gtja_128': self.alpha_gtja_128()
                , 'alpha_gtja_133': self.alpha_gtja_133()
                , 'alpha_gtja_134': self.alpha_gtja_134()
                , 'alpha_gtja_135': self.alpha_gtja_135()
                , 'alpha_gtja_136': self.alpha_gtja_136()
                , 'alpha_gtja_139': self.alpha_gtja_139()
                , 'alpha_gtja_140': self.alpha_gtja_140()
                , 'alpha_gtja_141': self.alpha_gtja_141()
                , 'alpha_gtja_142': self.alpha_gtja_142()
                , 'alpha_gtja_145': self.alpha_gtja_145()
                , 'alpha_gtja_146': self.alpha_gtja_146()
                , 'alpha_gtja_147': self.alpha_gtja_147()
                , 'alpha_gtja_148': self.alpha_gtja_148()
                , 'alpha_gtja_150': self.alpha_gtja_150()
                , 'alpha_gtja_151': self.alpha_gtja_151()
                , 'alpha_gtja_152': self.alpha_gtja_152()
                , 'alpha_gtja_153': self.alpha_gtja_153()
                , 'alpha_gtja_155': self.alpha_gtja_155()
                , 'alpha_gtja_157': self.alpha_gtja_157()
                , 'alpha_gtja_158': self.alpha_gtja_158()
                , 'alpha_gtja_159': self.alpha_gtja_159()
                , 'alpha_gtja_160': self.alpha_gtja_160()
                , 'alpha_gtja_161': self.alpha_gtja_161()
                , 'alpha_gtja_162': self.alpha_gtja_162()
                , 'alpha_gtja_164': self.alpha_gtja_164()
                , 'alpha_gtja_166': self.alpha_gtja_166()
                , 'alpha_gtja_167': self.alpha_gtja_167()
                , 'alpha_gtja_168': self.alpha_gtja_168()
                , 'alpha_gtja_169': self.alpha_gtja_169()
                , 'alpha_gtja_171': self.alpha_gtja_171()
                , 'alpha_gtja_173': self.alpha_gtja_173()
                , 'alpha_gtja_174': self.alpha_gtja_174()
                , 'alpha_gtja_175': self.alpha_gtja_175()
                , 'alpha_gtja_176': self.alpha_gtja_176()
                , 'alpha_gtja_177': self.alpha_gtja_177()
                , 'alpha_gtja_178': self.alpha_gtja_178()
                , 'alpha_gtja_180': self.alpha_gtja_180()
#                , '# alpha_gtja_181': self.alpha_gtja_181()
#                , '# alpha_gtja_182': self.alpha_gtja_182()
                , 'alpha_gtja_184': self.alpha_gtja_184()
                , 'alpha_gtja_185': self.alpha_gtja_185()
                , 'alpha_gtja_187': self.alpha_gtja_187()
                , 'alpha_gtja_188': self.alpha_gtja_188()
                , 'alpha_gtja_189': self.alpha_gtja_189()
                , 'alpha_gtja_190': self.alpha_gtja_190()
                , 'alpha_gtja_191': self.alpha_gtja_191()
                , 'ER': self.price.groupby('Ticker').apply(ER).squeeze()
                , 'DPO': self.price.groupby('Ticker').apply(DPO).squeeze()
                , 'POS': self.price.groupby('Ticker').apply(POS).squeeze()
                , 'TII': self.price.groupby('Ticker').apply(TII).squeeze()
             # , 'ADTM': self.price.groupby('Ticker').apply(ADTM).squeeze()
                , 'PO': self.price.groupby('Ticker').apply(PO).squeeze()
                , 'MADisplaced': self.price.groupby('Ticker').apply(MADisplaced).squeeze()
                , 'T3': self.price.groupby('Ticker').apply(T3).squeeze()
                , 'VMA': self.price.groupby('Ticker').apply(VMA).squeeze()
                , 'CR': self.price.groupby('Ticker').apply(CR).squeeze()
                , 'VIDYA': self.price.groupby('Ticker').apply(VIDYA).squeeze()
             })
        return pn


if __name__ == '__main__':
    stock_data = pd.read_csv('BARRA_USE4_new.csv')
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    # stock_data = stock_data[stock_data['Ticker'].isin(['TSLA', 'AAPL', 'GOOGL'])]
    stock_data = stock_data.set_index(['Date', 'Ticker'])
    # print(stock_data)
    # stock_panel = alpha_data.get_basic_data("2010-01-01", "2018-12-30", stock_list=security)

    start_time = time.time()
    tenjinAlpha = Tenjin_alpha(stock_data)
    pn = tenjinAlpha.calculate()
    print(pn.tail(30))
    print('spend time %s' % (time.time() - start_time))
    pn.to_csv('alphas_p2.csv')
