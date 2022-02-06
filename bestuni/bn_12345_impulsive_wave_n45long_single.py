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

import json
with open('config.json', 'r') as f:
    config = json.load(f)

futures = config['default']['futures']
rule = config['default']['rule']
leverage = config['default']['leverage']
seed = config['default']['seed']
fee = config['default']['fee']
fee_maker = config['default']['fee_maker']
fee_taker = config['default']['fee_taker']
fee_slippage = config['default']['fee_slippage']

basic_secret_key = config['basic']['secret_key']
basic_secret_value = config['basic']['secret_value']
futures_secret_key = config['futures']['secret_key']
futures_secret_value = config['futures']['secret_value']
period_days_ago = config['default']['period_days_ago']
period_days_ago_till = config['default']['period_days_ago_till']
period_interval = config['default']['period_interval']
timeframe = config['default']['timeframe']
up_to_count = config['default']['up_to_count']
long = config['default']['long']
h_fibo = config['default']['h_fibo']
l_fibo = config['default']['l_fibo']

print('-------------------------------')
print('futures:%s' % str(futures))
print('rule:%s' % str(rule))
print('leverage:%s' % str(leverage))
print('seed:%s' % str(seed))
print('fee:%s%%' % str(fee*100))
print('fee_maker:%s%%' % str(fee_maker*100))
print('fee_taker:%s%%' % str(fee_taker*100))
print('fee_slippage:%s%%' % str(round(fee_slippage*100, 4)))
fee_o = (fee_maker + fee_taker + fee_slippage) * leverage  # fee_maker buy:0.02%(0.0002) sell:0.02%(0.0002), sleepage:0.01%(0.0001)
print('fee_o:%s%%' % str(round(fee_o*100, 4)))

print('\n')
print('period_days_ago: %s' % period_days_ago)
print('period_days_ago_till: %s' % period_days_ago_till)
print('period_interval: %s' % period_interval)
print('timeframe: %s' % timeframe)

start_dt = str((pd.to_datetime('today') - pd.Timedelta(str(period_days_ago) + ' days')).date())
end_dt = str((pd.to_datetime('today') - pd.Timedelta(str(period_days_ago_till) + ' days')).date())
print('period: %s ~ %s' % (start_dt, end_dt))
print('\n')
print('up_to_count: %s' % up_to_count)
print('long: %s' % long)
print('h_fibo: %s' % h_fibo)
print('l_fibo: %s' % l_fibo)
print('-------------------------------')

pnl_total = []
fee_total = []
pnl_fee_total_n1 = []
pnl_fee_total_n2 = []
asset_total = [seed]
trade_count = []
trade_history = []
fee_history = []
pnl_history = []

client = None

if not futures:
    # basic
    client_basic = Client("basic_secret_key",
                    "basic_secret_value")
    client = client_basic
else:
    # futures
    client_futures = Client("futures_secret_key",
                    "futures_secret_value")
    client = client_futures


def get_historical_ohlc_data_start_end(symbol, start_str, end_str, past_days=None, interval=None, futures=False):
    try:
        """Returns historcal klines from past for given symbol and interval
        past_days: how many days back one wants to download the data"""
        if not futures:
            # basic
            client_basic = Client("basic_secret_key",
                                  "basic_secret_value")
            client = client_basic
        else:
            # futures
            client_futures = Client("futures_secret_key",
                                    "futures_secret_value")
            client = client_futures

        if not interval:
            interval = '1h'  # default interval 1 hour
        if not past_days:
            past_days = 30  # default past days 30.

        start_str = str((pd.to_datetime('today') - pd.Timedelta(str(start_str) + ' days')).date())
        if end_str:
            end_str = str((pd.to_datetime('today') - pd.Timedelta(str(end_str) + ' days')).date())
        else:
            end_str = None
        # print(start_str, end_str)
        D = None
        try:
            if futures:
                D = pd.DataFrame(
                    client.futures_historical_klines(symbol=symbol, start_str=start_str, end_str=end_str, interval=interval))
            else:
                D = pd.DataFrame(client.get_historical_klines(symbol=symbol, start_str=start_str, end_str=end_str, interval=interval))
            pass
        except Exception as e:
            print(e)
            return D, start_str, end_str

        if D is not None and D.empty:
            return D, start_str, end_str

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
        D['Open'] = D['Open'].astype(float)
        D['High'] = D['High'].astype(float)
        D['Low'] = D['Low'].astype(float)
        D['Close'] = D['Close'].astype(float)
        D['Volume'] = D['Volume'].astype(float)
        D = D[new_names]
    except Exception as e:
        print('e in get_historical_ohlc_data_start_end:%s' % e)
        pass
    return D, start_str, end_str

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


def backtest_trade45(df, wavepattern_up, symbol):
    w = wavepattern_up
    high = w.high
    low = w.low
    idx_end = w.idx_end
    idx_start = w.idx_start
    height = high - low
    w_end = w.dates[-1]
    w_start = w.dates[0]

    h_fibo_value = 0
    l_fibo_value = 0
    if h_fibo:
        h_fibo_value = height * 11.8 / 100
    if l_fibo:
        l_fibo_value = height * 11.8 / 100

    start = w.waves['wave4'].high
    if rule == '4low':
        start = w.waves['wave4'].low
        date = w.waves['wave4'].date_end
    elif rule == '1high':
        start = w.waves['wave1'].high
        date = w.waves['wave1'].date_start
    elif rule == '14half':
        start = (w.waves['wave1'].high + w.waves['wave4'].low)/2

    d = df[w.idx_end + 1:]
    dates = d.Date.tolist()
    closes = d.Close.tolist()
    lows = d.Low.tolist()
    highs = d.High.tolist()

    if futures:
        fee_o = (fee_maker + fee_taker + fee_slippage) * leverage  # fee_maker buy:0.02%(0.0002) sell:0.02%(0.0002), sleepage:0.01%(0.0001)
    else:
        fee_o = fee * leverage  # buy:0.1%(0.0001) sell:0.1%

    position = False
    entry_idx = None
    # long = True

    if trade_history:
        h_pre = trade_history[-1][-1]
        if h_pre.low == low and h_pre.high == high:
            return [0, 0, asset_total[-1]]

    if closes:
        for i, c in enumerate(closes):
            # entry
            if position is not True and lows[i] <= start:
                position = True
                entry_idx = i
                posi_start = dates[i]
            elif position is not True and lows[i] < low:
                return [0, 0, asset_total[-1]]
            elif position is not True and highs[i] > high:
                return [0, 0, asset_total[-1]]

            # PT about long or SL about short
            elif position and highs[i] >= high + h_fibo_value:
                pnl = +((high + h_fibo_value - start) / start) * (1 if long else -1) * leverage
                asset_after = asset_total[-1] * (1 + pnl - (fee_o))
                asset_total.append(asset_after)
                trade_count.append(+1)
                fee_history.append(asset_total[-1]*(fee_o))
                pnl_history.append(asset_total[-1]*pnl)
                posi_end = dates[i]
                # winrate = round((sum(list(map(lambda i: i if i > 0 else 0, trade_count)))/len(trade_count))*100, 2)
                winrate = round((sum(trade_count)/len(trade_count))*100, 2)
                h1 = [len(trade_count), winrate, symbol, asset_after, round(pnl, 2), sum(pnl_history), sum(fee_history)]
                h2 = [str(w_start), str(w_end), (str(date), start), (str(posi_start), start), (str(posi_end), high + h_fibo_value)]
                h = [h1, h2, wavepattern_up]
                trade_history.append(h)
                # print(str(h1) + ' + ' +str(h2))
                print(str(h1))
                return [asset_total[-1]*pnl, asset_total[-1]*(fee_o), asset_after]

            # SL about long or PT about short
            elif position and lows[i] <= low + l_fibo_value:
                pnl = ((start - (low - l_fibo_value)) / start) * (-1 if long else 1) * leverage
                asset_after = asset_total[-1] * (1 + pnl - (fee_o))
                asset_total.append(asset_after)
                trade_count.append(0)
                fee_history.append(asset_total[-1]*(fee_o))
                pnl_history.append(asset_total[-1]*pnl)
                posi_end = dates[i]
                # winrate = round((sum(list(map(lambda i: i if i > 0 else 0, trade_count)))/len(trade_count))*100, 2)
                winrate = round((sum(trade_count)/len(trade_count))*100, 2)
                h1 = [len(trade_count), winrate, symbol, asset_after, round(pnl, 2), sum(pnl_history), sum(fee_history)]
                h2 = [str(w_start), str(w_end), (str(date), start), (str(posi_start), start), (str(posi_end), low - l_fibo_value)]
                h = [h1, h2, wavepattern_up]
                trade_history.append(h)
                # print(str(h1) + ' + ' +str(h2))
                print(str(h1))
                return [asset_total[-1]*pnl, asset_total[-1]*(fee_o), asset_after]
    return [0, 0, asset_total[-1]]

def check_has_same_wavepattern(w_l, wavepattern_up):
    for i in w_l:
        eq_dates = np.array_equal(np.array(wavepattern_up.dates), np.array(i[-1].dates))
        eq_values = np.array_equal(np.array(wavepattern_up.values), np.array(i[-1].values))
        if eq_dates and eq_values:
            return True
    return False

def loopsymbol(symbol, i):
    ###################
    ## data settting ##
    ###################
    # 1day:1440m
    # period_days_ago = 30
    # period_interval = 1
    # for i in reversed(range(period_days_ago)):
    try:
        past_days = 1
        # timeframe = 1  # 1, 5, 15
        timeunit = 'm'
        bin_size = str(timeframe) + timeunit
        # stick_cnt_per_day = past_days * 1440 / timeframe if timeunit == 'm' else 0
        # divide = 4
        start_str = i
        end_str = start_str - period_interval
        if end_str < 0:
            end_str = None
        df, start_date, end_date = get_historical_ohlc_data_start_end(symbol, start_str=start_str,
                                                                          end_str=end_str, past_days=past_days,
                                                                          interval=bin_size, futures=futures)
        df_all = df
        df_lows = fractals_low_loop(df_all)
        df_lows = fractals_low_loop(df_lows)
        df_lows_plot = df_lows[['Date', 'Low']]
        df_hi_low = fractals_only_hi_lo(df_all)
        wa = WaveAnalyzer(df=df_all, verbose=True)
        wave_options_impulse = WaveOptionsGenerator5(up_to=up_to_count)  # generates WaveOptions up to [15, 15, 15, 15, 15]
        impulse = Impulse('impulse')
        rules_to_check = [impulse]
        wavepatterns_up = set()

        mins = df_lows.index.tolist()
        wavepattern_up_l = list()
        wave_option_plot_l = []
        pnl_small_l = []
        for i in mins:
            for new_option_impulse in wave_options_impulse.options_sorted:
                waves_up = wa.find_impulsive_wave(idx_start=i, wave_config=new_option_impulse.values)
                if waves_up:
                    wavepattern_up = WavePattern(waves_up, verbose=True)
                    for rule in rules_to_check:
                        if wavepattern_up.check_rule(rule):
                            if wavepattern_up in wavepatterns_up:
                                continue
                            else:
                                if not check_has_same_wavepattern(wavepattern_up_l, wavepattern_up):
                                    wavepatterns_up.add(wavepattern_up)
                                # print(f'{rule.name} found: {new_option_impulse.values}')
                                # plot_pattern(df=df, wave_pattern=wavepattern_up, title=str(
                                # symbol + '_' + interval_str + '_' + str(i) + ':' + rule.name + ' ' + str(
                                #     new_option_impulse.values)))
                                # plot_pattern_m(df=df_all, wave_pattern=[wavepattern_up], df_plot=df_lows_plot, title=str(
                                #     symbol + '_II_' + interval_str + '_' + str(i) + ':' + rule.name + ' ' + str(
                                #         new_option_impulse.values)))

                                    wavepattern_up_l.append([i, wavepattern_up.dates[0], id(wavepattern_up), wavepattern_up])
                                    wave_option_plot_l.append([
                                        [str(wavepattern_up.waves['wave5'].dates[1])],
                                        [wavepattern_up.waves['wave5'].high],
                                        [str(new_option_impulse.values)]
                                    ])

                                    pnl_fee_n2 = backtest_trade45(df_all, wavepattern_up, symbol)
                                    if pnl_fee_n2:
                                        # print(start_date, end_date)
                                        pnl_fee_total_n2.append(pnl_fee_n2)
                                        pnl_total.append(pnl_fee_n2[0])
                                        fee_total.append(pnl_fee_n2[1])
                                        pnl_small_l.append(pnl_fee_n2[0])

        # if wavepattern_up_l:
        #     t = interval_str + '_' + str(i) + ':' + rule.name + ' ' + str(new_option_impulse.values) + ' ' + str(trade_history[-1][0])
        #     plot_pattern_m(df=df_all, wave_pattern=wavepattern_up_l,
        #                    df_plot=df_lows_plot,
        #                    wave_options=wave_option_plot_l, title=t)
    except Exception as e:
        print('loopsymbol Exception : %s ' % e)
        pass


def single(symbols, i):
    for symbol in symbols:
        loopsymbol(symbol, i)


# if __name__ == '__main__':

###################
## symbol setting ##
###################
exchange_info = client.get_exchange_info()
symbols = []
for s in exchange_info['symbols']:
    symbol = s['symbol']
    symbols.append(symbol)

if futures:
    find_USDT = 'USDT'
    index_USDTs = [i for i in range(len(symbols)) if find_USDT in symbols[i]]
    symbols_usdt = []
    for s in index_USDTs:
        symbols_usdt.append(symbols[s])

    symbols = symbols_usdt

    #  134
    symbols_usdt_yes = [
                        # 'BTCUSDT', 'ETHUSDT', 'BNBUSDT',
                        'NEOUSDT', 'LTCUSDT', 'QTUMUSDT', 'ADAUSDT', 'XRPUSDT', 'EOSUSDT', 'IOTAUSDT', 'XLMUSDT', 'ONTUSDT', 'TRXUSDT', 'ETCUSDT', 'ICXUSDT', 'VETUSDT', 'LINKUSDT', 'WAVESUSDT', 'BTTUSDT', 'HOTUSDT', 'ZILUSDT', 'ZRXUSDT', 'BATUSDT', 'XMRUSDT', 'ZECUSDT', 'IOSTUSDT', 'CELRUSDT', 'DASHUSDT', 'OMGUSDT', 'THETAUSDT', 'ENJUSDT', 'MATICUSDT', 'ATOMUSDT', 'ONEUSDT', 'FTMUSDT', 'ALGOUSDT', 'DOGEUSDT', 'ANKRUSDT', 'MTLUSDT', 'TOMOUSDT', 'DENTUSDT', 'CVCUSDT', 'CHZUSDT', 'BANDUSDT', 'XTZUSDT', 'RENUSDT', 'RVNUSDT', 'HBARUSDT', 'NKNUSDT', 'KAVAUSDT', 'ARPAUSDT', 'IOTXUSDT', 'RLCUSDT', 'BCHUSDT', 'OGNUSDT', 'BTSUSDT', 'COTIUSDT', 'SOLUSDT', 'CTSIUSDT', 'CHRUSDT', 'LENDUSDT', 'STMXUSDT', 'KNCUSDT', 'LRCUSDT', 'COMPUSDT', 'SCUSDT', 'ZENUSDT', 'SNXUSDT', 'DGBUSDT', 'SXPUSDT', 'MKRUSDT', 'STORJUSDT', 'MANAUSDT', 'YFIUSDT', 'BALUSDT', 'BLZUSDT', 'SRMUSDT', 'ANTUSDT', 'CRVUSDT', 'SANDUSDT', 'OCEANUSDT', 'DOTUSDT', 'LUNAUSDT', 'RSRUSDT', 'TRBUSDT', 'BZRXUSDT', 'SUSHIUSDT', 'YFIIUSDT', 'KSMUSDT', 'EGLDUSDT', 'RUNEUSDT', 'BELUSDT', 'UNIUSDT', 'AVAXUSDT', 'HNTUSDT', 'FLMUSDT', 'ALPHAUSDT', 'AAVEUSDT', 'NEARUSDT', 'FILUSDT', 'AUDIOUSDT', 'CTKUSDT', 'AKROUSDT', 'AXSUSDT', 'UNFIUSDT', 'XEMUSDT', 'SKLUSDT', 'GRTUSDT', '1INCHUSDT', 'REEFUSDT', 'CELOUSDT', 'BTCSTUSDT', 'LITUSDT', 'SFPUSDT', 'DODOUSDT', 'ALICEUSDT', 'LINAUSDT', 'TLMUSDT', 'BAKEUSDT', 'ICPUSDT', 'ARUSDT', 'MASKUSDT', 'LPTUSDT', 'NUUSDT', 'ATAUSDT', 'GTCUSDT', 'KEEPUSDT', 'KLAYUSDT', 'C98USDT', 'RAYUSDT', 'DYDXUSDT', 'GALAUSDT', 'ENSUSDT', 'PEOPLEUSDT']
    #  256
    # symbols_usdt_no = ['BCCUSDT', 'TUSDUSDT', 'VENUSDT', 'NULSUSDT', 'PAXUSDT', 'BCHABCUSDT', 'BCHSVUSDT', 'USDCUSDT', 'USDSUSDT', 'ONGUSDT', 'FETUSDT', 'NANOUSDT', 'MITHUSDT', 'TFUELUSDT', 'USDSBUSDT', 'GTOUSDT', 'ERDUSDT', 'DUSKUSDT', 'WINUSDT', 'COSUSDT', 'NPXSUSDT', 'COCOSUSDT', 'PERLUSDT', 'MFTUSDT', 'KEYUSDT', 'STORMUSDT', 'DOCKUSDT', 'WANUSDT', 'FUNUSDT', 'BUSDUSDT', 'BEAMUSDT', 'HCUSDT', 'STXUSDT', 'MCOUSDT', 'CTXCUSDT', 'TROYUSDT', 'VITEUSDT', 'FTTUSDT', 'BUSDTRY', 'USDTTRY', 'USDTRUB', 'EURUSDT', 'DREPUSDT', 'BULLUSDT', 'BEARUSDT', 'ETHBULLUSDT', 'ETHBEARUSDT', 'TCTUSDT', 'WRXUSDT', 'LSKUSDT', 'BNTUSDT', 'LTOUSDT', 'EOSBULLUSDT', 'EOSBEARUSDT', 'XRPBULLUSDT', 'XRPBEARUSDT', 'STRATUSDT', 'AIONUSDT', 'MBLUSDT', 'BNBBULLUSDT', 'BNBBEARUSDT', 'STPTUSDT', 'USDTZAR', 'WTCUSDT', 'DATAUSDT', 'XZCUSDT', 'USDTIDRT', 'HIVEUSDT', 'BTCUPUSDT', 'BTCDOWNUSDT', 'GXSUSDT', 'ARDRUSDT', 'MDTUSDT', 'REPUSDT', 'PNTUSDT', 'USDTUAH', 'USDTBIDR', 'BKRWUSDT', 'ETHUPUSDT', 'ETHDOWNUSDT', 'ADAUPUSDT', 'ADADOWNUSDT', 'LINKUPUSDT', 'LINKDOWNUSDT', 'VTHOUSDT', 'GBPUSDT', 'DAIUSDT', 'DCRUSDT', 'BNBUPUSDT', 'BNBDOWNUSDT', 'XTZUPUSDT', 'XTZDOWNUSDT', 'USDTBKRW', 'AUDUSDT', 'IRISUSDT', 'KMDUSDT', 'USDTDAI', 'JSTUSDT', 'NMRUSDT', 'PAXGUSDT', 'WNXMUSDT', 'DIAUSDT', 'FIOUSDT', 'UMAUSDT', 'EOSUPUSDT', 'EOSDOWNUSDT', 'TRXUPUSDT', 'TRXDOWNUSDT', 'XRPUPUSDT', 'XRPDOWNUSDT', 'DOTUPUSDT', 'DOTDOWNUSDT', 'USDTNGN', 'WINGUSDT', 'LTCUPUSDT', 'LTCDOWNUSDT', 'NBSUSDT', 'OXTUSDT', 'SUNUSDT', 'UNIUPUSDT', 'UNIDOWNUSDT', 'ORNUSDT', 'UTKUSDT', 'XVSUSDT', 'USDTBRL', 'SXPUPUSDT', 'SXPDOWNUSDT', 'FILUPUSDT', 'FILDOWNUSDT', 'YFIUPUSDT', 'YFIDOWNUSDT', 'INJUSDT', 'BCHUPUSDT', 'BCHDOWNUSDT', 'HARDUSDT', 'DNTUSDT', 'STRAXUSDT', 'ROSEUSDT', 'AVAUSDT', 'AAVEUPUSDT', 'AAVEDOWNUSDT', 'SUSDUSDT', 'SUSHIUPUSDT', 'SUSHIDOWNUSDT', 'XLMUPUSDT', 'XLMDOWNUSDT', 'JUVUSDT', 'PSGUSDT', 'USDTBVND', 'OGUSDT', 'ATMUSDT', 'ASRUSDT', 'RIFUSDT', 'TRUUSDT', 'CKBUSDT', 'TWTUSDT', 'FIROUSDT', 'CAKEUSDT', 'ACMUSDT', 'BADGERUSDT', 'FISUSDT', 'OMUSDT', 'PONDUSDT', 'DEGOUSDT', 'PERPUSDT', 'RAMPUSDT', 'SUPERUSDT', 'CFXUSDT', 'EPSUSDT', 'AUTOUSDT', 'TKOUSDT', 'PUNDIXUSDT', '1INCHUPUSDT', '1INCHDOWNUSDT', 'BTGUSDT', 'MIRUSDT', 'BARUSDT', 'FORTHUSDT', 'BURGERUSDT', 'SLPUSDT', 'SHIBUSDT', 'USDTGYEN', 'POLSUSDT', 'MDXUSDT', 'XVGUSDT', 'TORNUSDT', 'ERNUSDT', 'PHAUSDT', 'BONDUSDT', 'MLNUSDT', 'DEXEUSDT', 'CLVUSDT', 'QNTUSDT', 'FLOWUSDT', 'TVKUSDT', 'MINAUSDT', 'FARMUSDT', 'ALPACAUSDT', 'QUICKUSDT', 'MBOXUSDT', 'FORUSDT', 'REQUSDT', 'GHSTUSDT', 'WAXPUSDT', 'TRIBEUSDT', 'GNOUSDT', 'XECUSDT', 'ELFUSDT', 'POLYUSDT', 'IDEXUSDT', 'VIDTUSDT', 'USDPUSDT', 'ILVUSDT', 'YGGUSDT', 'SYSUSDT', 'DFUSDT', 'FIDAUSDT', 'FRONTUSDT', 'CVPUSDT', 'AGLDUSDT', 'RADUSDT', 'BETAUSDT', 'RAREUSDT', 'LAZIOUSDT', 'CHESSUSDT', 'ADXUSDT', 'AUCTIONUSDT', 'DARUSDT', 'BNXUSDT', 'RGTUSDT', 'MOVRUSDT', 'CITYUSDT', 'KP3RUSDT', 'QIUSDT', 'PORTOUSDT', 'POWRUSDT', 'VGXUSDT', 'JASMYUSDT', 'AMPUSDT', 'PLAUSDT', 'PYRUSDT', 'RNDRUSDT', 'ALCXUSDT', 'SANTOSUSDT', 'MCUSDT', 'ANYUSDT', 'BICOUSDT', 'FLUXUSDT', 'FXSUSDT', 'VOXELUSDT', 'HIGHUSDT', 'CVXUSDT', 'OOKIUSDT', 'SPELLUSDT', 'USTUSDT', 'JOEUSDT']
    symbols = symbols_usdt_yes

# symbols = random.sample(symbols, len(symbols))
print(symbols[:20])
# symbols = ['AXSUSDT']

total_result_l = []
try:
    for i in range(period_days_ago, period_days_ago_till, -1 * period_interval):
        if i > 0:
            print(i, 'days ago start')
            symbols = random.sample(symbols, len(symbols))
            symbols = symbols[:20]
            single(symbols, i)

    # start = time.perf_counter()
    # cpucount = multiprocessing.cpu_count() * 1
    # # cpucount = multiprocessing.cpu_count()
    # # cpucount = 1
    # print('cpucount:%s' % cpucount)
    # pool = multiprocessing.Pool(processes=cpucount)
    # rt = pool.map(loopsymbol, [symbols_usdt])
    # # rt_l.append(rt)
    # print(str(rt))
    # pool.close()
    # pool.join()
    # print('======== i:%s --- %s agent done ...  == %s  ==> %s  ========' % (
    # str(i), symbol, str(start_date), str(end_date)))
    # print(f'Finished in {round(time.perf_counter() - start, 2)} second(s)')
except Exception as e:
    print(e)
    pass

print('============End All==========')
