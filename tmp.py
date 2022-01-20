from __future__ import annotations
from models.WavePattern import WavePattern
from models.WaveRules import Impulse, LeadingDiagonal, Correction
from models.WaveAnalyzer import WaveAnalyzer
from models.WaveOptions import WaveOptionsGenerator5
from models.helpers import plot_pattern
import pandas as pd
import numpy as np
import json
from pytz import UTC
from datetime import datetime, timezone
from datetime import date, timedelta


def check_has_same_wavepattern(l, wavepattern_up):
    for i in l:
        eq_dates = np.array_equal(np.array(wavepattern_up.dates), np.array(i[2].dates))
        eq_values = np.array_equal(np.array(wavepattern_up.values), np.array(i[2].values))
        if eq_dates and eq_values:
            return True
    return False


def check_has_same_wavepattern_df(df, wavepattern_up):
    l = df.wave.tolist()
    for i in l:
        eq_dates = np.array_equal(np.array(wavepattern_up.dates), np.array(i.dates))
        eq_values = np.array_equal(np.array(wavepattern_up.values), np.array(i.values))
        if eq_dates and eq_values:
            return True
    return False


def check_in_range(df_symbol_date, nowpoint, time, low, high):
    r = True
    max = np.max(df_symbol_date[time:nowpoint].High.tolist())
    if max >= high:
        return False
    min = np.min(df_symbol_date[time:nowpoint].Low.tolist())
    if min <= low:
        return False
    return r


def agent_filter_and_update_wave_df(df_wave, symbol, nowpoint, symbols_d):
    c1 = df_wave['symbol'] == symbol
    c2 = df_wave['time'] <= nowpoint
    df_wave_f = df_wave[c1 & c2]

    df_symbol = symbols_d[symbol]
    df_symbol = df_symbol[df_symbol['Date'] <= nowpoint]
    df_symbol_date = df_symbol.set_index(['Date'])
    if len(df_wave_f) > 0:
        # filter
        df_wave_f.loc[:, ('valid')] = df_wave_f.apply(
            lambda x: 0 if check_in_range(df_symbol_date, nowpoint, x['time'], x['low'], x['high']) else 1,
            axis=1)
        # update
        df_wave.loc[df_wave['id'].isin(df_wave_f['id'].tolist()), ['valid']] = 1
    return df_wave_f




################
# init
################
symbols = ['AMD', 'btc-usd_1d', 'FB', 'INFY', 'MONDY', 'MTDR']
df_wave = pd.DataFrame([], columns=['id', 'symbol', 'time', 'low', 'w1', 'w2', 'w3', 'w4', 'high', 'wave', 'position', 'valid'])
sdate = date(2019, 3, 22)   # start date
edate = date(2019, 4, 9)   # end date
up_to_count = 5
nowpoint = '2018-12-25'
symbols_d = dict()

################
# data collector
################
def data_collector(symbols):
    for symbol in symbols:
        symbols_d[symbol] = pd.read_csv(symbol+'.csv')
    return symbols_d

################
# wave analyzer
################

def wave_analyzer(df_wave, symbols, symbols_d, up_to_count=None, startpoint=None, nowpoint=None, window=1000):
    for symbol in symbols:
        df = symbols_d[symbol]
        if nowpoint:
            c = df['Date'] <= nowpoint
            if startpoint:
                c = c & df['Date'] >= startpoint
            df = df[c]
        if df.size > window:
            df = df.iloc[-window:]
        idx_start = np.argmin(np.array(list(df['Low'])))
        idx_start_high = np.argmin(np.array(list(df['High'])))
        wa = WaveAnalyzer(df=df, verbose=False)
        up_to_count = up_to_count if up_to_count else 1
        wave_options_impulse = WaveOptionsGenerator5(up_to=up_to_count)  # generates WaveOptions up to [15, 15, 15, 15, 15]

        impulse = Impulse('impulse')
        rules_to_check = [impulse]
        # rules_to_check = [impulse, leading_diagonal]
        # correction = Correction('Correction')
        # rules_to_check = [correction]

        print(f'Start at idx: {idx_start}')
        print(f"will run up to {wave_options_impulse.number / 1e6}M combinations.")

        # set up a set to store already found wave counts
        # it can be the case, that 2 WaveOptions lead to the same WavePattern.
        # This can be seen in a chart, where for example we try to skip more maxima as there are. In such a case
        # e.g. [1,2,3,4,5] and [1,2,3,4,10] will lead to the same WavePattern (has same sub-wave structure, same begin / end,
        # same high / low etc.
        # If we find the same WavePattern, we skip and do not plot it

        wavepatterns_up = set()
        # loop over all combinations of wave options [i,j,k,l,m] for impulsive waves sorted from small, e.g.  [0,1,...] to
        # large e.g. [3,2, ...]

        for new_option_impulse in wave_options_impulse.options_sorted:
            waves_up = wa.find_impulsive_wave(idx_start=idx_start, wave_config=new_option_impulse.values)
            if waves_up:
                wavepattern_up = WavePattern(waves_up, verbose=True)
                for rule in rules_to_check:
                    if wavepattern_up.check_rule(rule):
                        if wavepattern_up in wavepatterns_up:
                            print('################################')
                            print('same wavepattern id:%s' % id(wavepattern_up))
                            print('################################')
                            continue
                        else:
                            wavepatterns_up.add(wavepattern_up)
                            if not check_has_same_wavepattern_df(df_wave, wavepattern_up):
                                w_info = [id(wavepattern_up),
                                          symbol,
                                          wavepattern_up.dates[0],
                                          wavepattern_up.low,
                                          wavepattern_up.values[1],
                                          wavepattern_up.values[3],
                                          wavepattern_up.values[5],
                                          wavepattern_up.values[7],
                                          wavepattern_up.high,
                                          wavepattern_up,
                                          # wavepattern_up.__dict__,
                                          # json.dumps(wavepattern_up.__dict__),
                                          0,
                                          0
                                          ]
                                df_wave = df_wave.append(pd.DataFrame([w_info], columns=df_wave.columns), ignore_index=True)
                                print('len(df_wave):%s' % len(df_wave))
                                print(f'{rule.name} found: {new_option_impulse.values}')
                            # plot_pattern(df=df, wave_pattern=wavepattern_up, title=symbol + str(new_option_impulse))
        return df_wave


################
# wave manager
################
def wave_manager(df_wave, symbols, symbols_d, nowpoint=None):
    if nowpoint is None:
        nowpoint = str(datetime.now().astimezone(UTC))
        # now_time = datetime.now(timezone.utc)
    wave_standby_d = dict()
    for symbol in symbols:
        df_wave_symbol_standby = agent_filter_and_update_wave_df(df_wave, symbol, nowpoint, symbols_d)
        wave_standby_d[symbol] = df_wave_symbol_standby
    print(wave_standby_d, df_wave)
    return wave_standby_d, df_wave


symbols_d = data_collector(symbols)
df_wave = wave_analyzer(df_wave, symbols, symbols_d, up_to_count=up_to_count, nowpoint=nowpoint)
wave_standby_d, df_wave = wave_manager(df_wave, symbols, symbols_d, nowpoint=nowpoint)

################
# trader
################
date_l = pd.date_range(sdate, edate-timedelta(days=1), freq='d')
print(date_l.tolist())
for d in date_l.tolist():
    print(d)
    pass



https://stackoverflow.com/questions/69184604/bybit-api-is-there-a-way-to-place-take-profit-stop-loss-orders-after-a-openin
    
    
    Yes, it is possible to attach these orders after entering a position. In the docs they reference set stop, and this is also included in the test.py doc page within the Bybit python install

here is the link to the docs

Bybit Set Stop

Here is what a stop and TP would look like for a LONG position. Please note that for a long we set what our current pos is for the side argument. (BUY)

# Stop Loss
print(client.LinearPositions.LinearPositions_tradingStop(
    symbol="BTCUSDT", 
    side="Buy", 
    stop_loss=41000).result())

# Take profit
print(client.LinearPositions.LinearPositions_tradingStop(
    symbol="BTCUSDT", 
    side="Buy", 
    take_profit=49000).result())
Additional note: TP orders are conditional orders, meaning they are sent to the order book once triggered, which results in a market order. If you already know your target level, a limit order may be more suitable. This will go to your active orders, which you will have to cancel. We use a sell argument for this one:

# Limit order
print(client.LinearOrder.LinearOrder_new(
    side="Sell",
    symbol="BTCUSDT",
    order_type="Limit",
    qty=0.001,
    price=49000,
    time_in_force="GoodTillCancel",
    reduce_only=True, 
    close_on_trigger=False).result())
Cheers my friend and good luck with your coding and trading!


bybit python Close On Trigger Order?


Another way to is to listen to websocket data. What I do is I subscribe to "execution" topic. This way, every time your order gets executed you receive an event with all the info about the trade. Then, you can have a callback function that places a trade for you.

Here's the link to the api: https://bybit-exchange.github.io/docs/inverse/#t-websocketexecution

Here's how to subscribe:

enter image description here

Here's the sample response:

enter image description here

Share
Follow
