from __future__ import annotations
from models.WavePattern import WavePattern
from models.WaveRules import Impulse, LeadingDiagonal, Correction
from models.WaveAnalyzer import WaveAnalyzer
from models.WaveOptions import WaveOptionsGenerator5
from models.helpers import plot_pattern, plot_pattern_m
import pandas as pd
import numpy as np
import math
from binance.client import Client
# matplotlib.use('TkAgg')
import datetime as dt
from tapy import Indicators
import time
import multiprocessing
from functools import partial
from ta import add_all_ta_features
from ta.utils import dropna
import random
from tqdm import tqdm
from scipy.signal import argrelextrema

client = Client("knwlJpZVdWLy20iRnHInxyRi2SXHknbj0tKTO9vJqi7NOno30fDO2y2zYNPyvYZq", "ZKlLtBwjRVI2QfQTNyH0vnchxRuTTsHDLZkcbA3JK9Z1ieYvbJeZgVSi8oyA17rE")
client = Client()
from backtesting import Strategy, Backtest


def SMA(values, n):
    """
    Return simple moving average of `values`, at
    each step taking into account `n` previous values.
    """
    return pd.Series(values).rolling(n).mean()

class Elliot5S(Strategy):

    def init(self):
        # Compute moving averages the strategy demands
        self.ma10 = self.I(SMA, self.data.Close, 10)
        self.ma20 = self.I(SMA, self.data.Close, 20)
        self.ma100 = self.I(SMA, self.data.Close, 100)

    def next(self):
        price = self.data.Close[-1]
        # If we don't already have a position, and
        # if all conditions are satisfied, enter long.
        if (not self.position and
                self.ma20[-1] > self.ma100[-1] and
                price > self.ma10[-1]):

            # Buy at market price on next open, but do
            # set 8% fixed stop loss.
            self.buy(sl=.92 * price)

        # If the price closes 2% or more below 10-day MA
        # close the position, if any.
        elif price < .98 * self.ma10[-1]:
            self.position.close()



def resample(df, rate):
    rt = df.resample(rate, closed='right', label='right').agg({'Open': 'first',
                                                                'High': 'max',
                                                                'Low': 'min',
                                                                'Close': 'last'}).dropna()
    return rt

def get_data(symbol, lookback):
    df = get_historical_ohlc_data(symbol, past_days=past_days, interval=interval_str)
    all_data = df.set_index('Date')
    all_data.index = pd.to_datetime(all_data.index)
    return all_data

def get_stock_data(stocklist, lookback):
    stock_data = {}
    for stock in tqdm(stocklist, desc='Getting stock data'):
        try:
            stock_data[stock] = get_data(stock, lookback)
        except Exception as e:
            print('Exception {} {}'.format(stock, e))
    return stock_data

def get_max_min(prices, smoothing, window_range):
    smooth_prices = prices['Close'].rolling(window=smoothing).mean().dropna()
    local_max = argrelextrema(smooth_prices.values, np.greater)[0]
    local_min = argrelextrema(smooth_prices.values, np.less)[0]
    price_local_max_dt = []
    for i in local_max:
        if (i > window_range) and (i < len(prices) - window_range):
            price_local_max_dt.append(prices.iloc[i - window_range:i + window_range]['Close'].idxmax())
    price_local_min_dt = []
    for i in local_min:
        if (i > window_range) and (i < len(prices) - window_range):
            price_local_min_dt.append(prices.iloc[i - window_range:i + window_range]['Close'].idxmin())
    maxima = pd.DataFrame(prices.loc[price_local_max_dt])
    minima = pd.DataFrame(prices.loc[price_local_min_dt])
    max_min = pd.concat([maxima, minima]).sort_index()
    max_min.index.name = 'Date'
    max_min = max_min.reset_index()
    max_min = max_min[~max_min.Date.duplicated()]
    p = prices.reset_index()
    max_min['day_num'] = p[p['Date'].isin(max_min.Date)].index.values
    max_min = max_min.set_index('day_num')['Close']
    return max_min


def get_maxs_mins(prices, smoothing, window_range):
    # prices['ix'] = np.arange(len(prices))
    smooth_prices = prices['Close'].rolling(window=smoothing).mean().dropna()
    local_max = argrelextrema(smooth_prices.values, np.greater)[0]
    local_min = argrelextrema(smooth_prices.values, np.less)[0]
    maxs = []
    m = []
    for i in local_max:
        if (i > window_range) and (i < len(prices) - window_range):
            hh = prices.iloc[i - window_range:i + window_range]['High'].argmax()
            m.append(i + hh)


    mins = []
    l = []
    for i in local_min:
        if (i > window_range) and (i < len(prices) - window_range):
            ll = prices.iloc[i - window_range:i + window_range]['Low'].argmin()
            l.append(i + ll)
    maxs = list(dict.fromkeys(m))
    mins = list(dict.fromkeys(l))
    return maxs, mins


def get_historical_ohlc_data(symbol, past_days=None, interval=None):
    """Returns historcal klines from past for given symbol and interval
    past_days: how many days back one wants to download the data"""

    if not interval:
        interval = '1h'  # default interval 1 hour
    if not past_days:
        past_days = 30  # default past days 30.

    start_str = str((pd.to_datetime('today') - pd.Timedelta(str(past_days) + ' days')).date())

    D = pd.DataFrame(client.get_historical_klines(symbol=symbol, start_str=start_str, interval=interval))
    D.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades',
                 'taker_base_vol', 'taker_quote_vol', 'is_best_match']
    D['open_date_time'] = [dt.datetime.fromtimestamp(x / 1000) for x in D.open_time]
    D['symbol'] = symbol
    D = D[['symbol', 'open_date_time', 'open', 'high', 'low', 'close', 'volume', 'num_trades', 'taker_base_vol',
           'taker_quote_vol']]
    D.rename(columns={
                        "open_date_time": "Date",
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume",
                      }, inplace=True)
    new_names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    # D['Open'] = D.iloc['Open'].astype(float)
    D['Open'] = D['Open'].astype(float)
    D['High'] = D['High'].astype(float)
    D['Low'] = D['Low'].astype(float)
    D['Close'] = D['Close'].astype(float)
    D['Volume'] = D['Volume'].astype(float)
    D = D[new_names]
    return D

# 1day:1440m
past_days = 60
interval = 5  # 1, 5, 15
timeunit = 'm'
interval_str = str(interval) + timeunit
stick_cnt_per_day = past_days * 1440 / interval if timeunit == 'm' else 0
divide = 5
period_unit = math.floor(stick_cnt_per_day/divide)
print('past_days:%s interval_str:%s period_unit:%s (total_unit:%s)' % (past_days, interval_str, period_unit, stick_cnt_per_day))



exchange_info = client.get_exchange_info()
symbols = []
for s in exchange_info['symbols']:
    symbol = s['symbol']
    symbols.append(symbol)
# print(symbols)

find_USDT = 'USDT'
index_USDTs = [i for i in range(len(symbols)) if find_USDT in symbols[i]]

symbols_usdt = []
for s in index_USDTs:
    symbols_usdt.append(symbols[s])
# print(symbols_usdt)
# print(len(symbols_usdt))
# symbols = [
#             'CHRUSDT',
#             # 'BTCUSDT',
#             # 'ETHUSDT',
#             # 'ALICEUSDT',
#             # 'GTCUSDT',
#             # 'TLMUSDT',
#             # 'EGLDUSDT',
#             # 'FTMUSDT',
#             # 'AXSUSDT',
#             # 'NUUSDT',
#             # 'LITUSDT',
#            ]

symbols_usdt = random.sample(symbols_usdt, len(symbols_usdt))

symbols = symbols_usdt[:200]
print(symbols)


def loopoptions(symbol, wa, new_option_impulse, idx_start, df, rules_to_check, wavepatterns_up):
    waves_up = wa.find_impulsive_wave(idx_start=idx_start, wave_config=new_option_impulse.values)
    if waves_up:
        wavepattern_up = WavePattern(waves_up, verbose=True)
        for rule in rules_to_check:
            if wavepattern_up.check_rule(rule):
                if wavepattern_up in wavepatterns_up:
                    continue
                else:
                    wavepatterns_up.add(wavepattern_up)
                    print(f'{rule.name} found: {new_option_impulse.values}')
                    plot_pattern(df=df, wave_pattern=wavepattern_up,
                                 title=str(symbol + '_' + interval_str + ':' + str(new_option_impulse)))

def fractals_low_loop(df):
    i = Indicators(df)
    i.fractals()
    df = i.df
    df = df[~(df[['fractals_low']] == 0).all(axis=1)]
    df = df.dropna()
    df = df.drop(['fractals_high', 'fractals_low'], axis=1)
    return df

def fractals_high_loop(df):
    i = Indicators(df)
    i.fractals()
    df = i.df
    df = df[~(df[['fractals_high']] == 0).all(axis=1)]
    df = df.dropna()
    df = df.drop(['fractals_high', 'fractals_low'], axis=1)
    return df

def fractals_only_hi_lo(df):
    i = Indicators(df)
    i.fractals()
    df = i.df
    df = df[~(df[['fractals_high', 'fractals_low']] == 0).all(axis=1)]
    df = df.dropna()
    df = df.drop(['fractals_high', 'fractals_low'], axis=1)
    return df

def backtest_trade_n1(df, wavepattern_up):
    w = wavepattern_up
    high = w.high
    low = w.low
    idx_end = w.idx_end
    close_start = df.iloc[idx_end]['Close']

    profit_target = w.waves['wave1'].high
    loss_target = high

    d = df[w.idx_end + 1:]

    c_l = d.Close.tolist()
    h_l = d.High.tolist()
    l_l = d.Low.tolist()

    pnl = 0.0
    fee = 0.0
    for i, c in enumerate(c_l):
        if h_l[i] >= loss_target:
            pnl = -((loss_target - close_start)/close_start)*100
            return [pnl, fee + 0.08]
        elif l_l[i] <= profit_target:
            pnl = ((close_start - profit_target)/close_start)*100
            return [pnl, fee + 0.06]
    return [pnl, fee]


def backtest_trade_n2(df, wavepattern_up):
    w = wavepattern_up
    high = w.high
    low = w.low
    idx_end = w.idx_end
    close_start = df.iloc[idx_end]['Close']

    diff_all = close_start - low
    pf_target = w.waves['wave1'].high
    loss_target = high

    d = df[w.idx_end+1:]

    close_ing_list = d.Close.tolist()
    pnl = 0.0
    fee = 0.00
    for c in close_ing_list:
        if c >= loss_target:
            pnl = -((c - close_start)/close_start)*100
            return [pnl, fee + 0.08]
        elif c <= pf_target:
            pnl = ((close_start - c)/close_start)*100
            return [pnl, fee + 0.08]
    return [pnl, fee]

def backtest_trade(df, wavepattern_up):
    w = wavepattern_up
    high = w.high
    low = w.low
    idx_end = w.idx_end
    close_start = df.iloc[idx_end]['Close']

    diff_all = close_start - low
    pf_target = w.waves['wave1'].high
    loss_target = high

    d = df[w.idx_end:]

    close_ing_list = d.Close.tolist()
    pnl = 0.0
    # fee = 0.0008
    fee = 0.04
    for c in close_ing_list:
        if c >= loss_target:
            pnl = -((loss_target - close_start)/close_start)*100
            return pnl, fee
        elif c <= pf_target:
            pnl = ((close_start - pf_target)/close_start)*100
            return pnl, fee
    return pnl, fee


    # bt = Backtest(d, Elliot5S, cash=10_000, commission=.0004)
    # stats = bt.run()
    # print(stats)
    # # print(stats._strategy)
    # bt.plot(plot_volume=False, plot_pl=False)
    # equity_curve = stats[
    #     '_equity_curve']  # Contains equity/drawdown curves. DrawdownDuration is only defined at ends of DD periods.
    # print(equity_curve)
    #
    # trade_history = stats['_trades']  # Contains individual trade data
    # print(trade_history)

from concurrent import futures
pnl_total = []
fee_total = []
pnl_fee_total_n1 = []
pnl_fee_total_n2 = []
count_total = []
def loopsymbol(symbol):
    print('\nsymbols: %s' % (symbols))
    print('symbol %s start' % (symbol))

    try:
        df = get_historical_ohlc_data(symbol, past_days=past_days, interval=interval_str)
        df_all = df
        df_lows = fractals_low_loop(df_all)
        # df_lows = fractals_low_loop(df_lows)
        df_lows = fractals_low_loop(df_lows)
        df_lows_plot = df_lows[['Date', 'Low']]

        df_hi_low = fractals_only_hi_lo(df_all)

        wa = WaveAnalyzer(df=df_all, verbose=True)
        up_to_count = 3
        # wave_options_impulse = WaveOptionsGenerator5(up_to=15)  # generates WaveOptions up to [15, 15, 15, 15, 15]
        wave_options_impulse = WaveOptionsGenerator5(up_to=up_to_count)  # generates WaveOptions up to [15, 15, 15, 15, 15]

        impulse = Impulse('impulse')
        rules_to_check = [impulse]
        print(f"will run up to {wave_options_impulse.number / 1e6}M combinations.")
        wavepatterns_up = set()

        mins = df_lows.index.tolist()
        # print('\n mins:%s' % (mins))

        filter_next_low_index = 0
        filter_next_wave2_low_indexs = []

        # plot_pattern_m(df=df_all, wave_pattern=None, df_plot=df_lows_plot, title=str(
        # symbol + '_' + interval_str))
        wavepattern_up_l = []
        wave_option_plot_l = []
        for i in mins:
            # print(f'Start at idx: {symbol} {i}')
            flg = True if i >= filter_next_low_index else False
            if flg:
                for new_option_impulse in wave_options_impulse.options_sorted:
                    waves_up = wa.find_impulsive_wave(idx_start=i, wave_config=new_option_impulse.values)
                    if waves_up:
                        wavepattern_up = WavePattern(waves_up, verbose=True)
                        for rule in rules_to_check:
                            # filter_next_wave2_low_indexs.append(wavepattern_up.waves['wave2'].idx_end)
                            # filter_next_low_index = wavepattern_up.waves['wave2'].idx_end
                            # print(f'filter_next_low_index: {filter_next_low_index}')
                            if wavepattern_up.check_rule(rule):
                                if wavepattern_up in wavepatterns_up:
                                    continue
                                else:
                                    wavepatterns_up.add(wavepattern_up)
                                    print(f'{rule.name} found: {new_option_impulse.values}')
                                    # plot_pattern(df=df, wave_pattern=wavepattern_up, title=str(
                                    # symbol + '_' + interval_str + '_' + str(i) + ':' + rule.name + ' ' + str(
                                    #     new_option_impulse.values)))




                                    # plot_pattern_m(df=df_all, wave_pattern=[wavepattern_up], df_plot=df_lows_plot, title=str(
                                    #     symbol + '_II_' + interval_str + '_' + str(i) + ':' + rule.name + ' ' + str(
                                    #         new_option_impulse.values)))



                                    wavepattern_up_l.append(wavepattern_up)
                                    wave_option_plot_l.append([
                                        [str(wavepattern_up.waves['wave5'].dates[1])],
                                        [wavepattern_up.waves['wave5'].high],
                                        [str(new_option_impulse.values)]
                                    ])
                                    # filter_next_wave2_low_indexs.append(wavepattern_up.waves['wave2'].idx_end)
                                    # filter_next_low_index = wavepattern_up.waves['wave2'].idx_end
                                    pnl, fee = backtest_trade(df_all, wavepattern_up)
                                    pnl_total.append(pnl)
                                    fee_total.append(fee)
                                    
                                    pnl_fee_n1 = backtest_trade_n1(df_all, wavepattern_up)
                                    pnl_fee_total_n1.append(pnl_fee_n1)
                                    
                                    pnl_fee_n2 = backtest_trade_n2(df_all, wavepattern_up)
                                    pnl_fee_total_n2.append(pnl_fee_n2)
                                    
                                    # msgi = ('%s (%s)impulsive wave [pnl:%s][pnl_mean:%s][pnl_total:%s]' %
                                    #        (symbol, len(wavepattern_up_l), pnl, np.mean(np.array(pnl_total)),
                                    #         str(pnl_total)))
                                    # print(msgi)
                                    count_total.append(1)
                                    # print('==============pure pnl===S===========')
                                    # print(sum(pnl_total))
                                    # print(sum(fee_total))
                                    # print(sum(pnl_total) - sum(fee_total))
                                    # print(sum(count_total))
                                    # print('====================================')
                # print(f'End at idx: {symbol} {i}')
            else:
                # print(f'Skip at idx: {symbol} {i}')
                pass
            # print('===============================')
            # print('impulsive wave : %s _ index: %s' % (symbol, i))
            # print('===============================')
        print('===============================')
        msg = ('%s (%s)impulsive wave [pnl_mean:%s][pnl_total:%s]' %
              (symbol, len(wavepattern_up_l), np.mean(np.array(pnl_total)), str(pnl_total)))
        print(msg)
        print('==============pure pnl=======H=======')
        print(sum(pnl_total))
        print(sum(fee_total))
        print(sum(pnl_total) - sum(fee_total))
        print(sum(count_total))
        print('===============================')

        # msgt = ('[pnl_mean:%s]' %
        #       (np.mean(np.array(pnl_total))))
        # if wavepattern_up_l:
        #     t = str(symbol + '_ALL_' + interval_str + '_' + str(i) + ':' + rule.name + ' ' + str(new_option_impulse.values) + ' '+msgt)
        #     plot_pattern_m(df=df_all, wave_pattern=wavepattern_up_l,
        #                    df_plot=df_lows_plot,
        #                    wave_options=wave_option_plot_l, title=t)
        #     print(t)
    except Exception as e:
        print('\nsymbols: %s' % (symbols))
        print('loopsymbol Exception : %s ' % e)
        pass

def single(symbols):
    for symbol in symbols:
        loopsymbol(symbol)


if __name__ == '__main__':
    # single(symbols)

    start = time.perf_counter()
    cpucount = multiprocessing.cpu_count() * 2
    # cpucount = multiprocessing.cpu_count()
    # cpucount = 1
    print('cpucount:%s' % cpucount)
    pool = multiprocessing.Pool(processes=cpucount)
    rt = pool.map(loopsymbol, symbols)
    pool.close()
    pool.join()
    print(f'Finished in {round(time.perf_counter() - start, 2)} second(s)')
    print('============End All==========')
