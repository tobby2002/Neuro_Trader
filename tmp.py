from __future__ import annotations
from models.WavePattern import WavePattern
from models.WaveRules import Impulse, LeadingDiagonal, Correction
from models.WaveAnalyzer import WaveAnalyzer
from models.WaveOptions import WaveOptionsGenerator5
from models.helpers import plot_pattern
import pandas as pd
import numpy as np


def check_has_same_wavepattern(w_l, wavepattern_up):
    for i in w_l:
        eq_dates = np.array_equal(np.array(wavepattern_up.dates), np.array(i[2].dates))
        eq_values = np.array_equal(np.array(wavepattern_up.values), np.array(i[2].values))
        if eq_dates and eq_values:
            return True
    return False


def check_has_same_wavepattern_df(df, wavepattern_up):
    w_l = df.wave.tolist()
    for i in w_l:
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


def agent_filter_and_update_wave_df(df_wave, symbol, nowpoint, df_symbols):
    c1 = df_wave['symbol'] == symbol
    c2 = df_wave['time'] <= nowpoint
    df_wave_f = df_wave[c1 & c2]

    df_symbol = df_symbols[symbol]
    df_symbol = df_symbol[df_symbol['Date'] <= nowpoint]
    df_symbol_date = df_symbol.set_index(['Date'])
    if len(df_wave_f) > 0:

        # filter
        df_wave_f.loc[:, ('valid')] = df_wave_f.apply(
            lambda x: 0 if check_in_range(df_symbol_date, nowpoint, x['time'], x['low'], x['high']) else 1,
            axis=1)

        # update
        df_wave_f

        # df_wave_f['value'] = np.where(df_wave_f.index > 20000, 0, df['value'])
    c3 = df_wave_f['valid'] == 0
    return df_wave_f[c3], df_wave

df_wave = pd.DataFrame([], columns=['id', 'symbol', 'time', 'low', 'w1', 'w2', 'w3', 'w4', 'high', 'wave', 'position', 'valid'])

w_l = list()
df_symbols = dict()

for symbol in ['AMD.csv', 'btc-usd_1d.csv', 'FB.csv', 'INFY.csv', 'MONDY.csv', 'MTDR.csv']:
    df = pd.read_csv(symbol)
    df_symbols[symbol] = df

    idx_start = np.argmin(np.array(list(df['Low'])))
    idx_start_high = np.argmin(np.array(list(df['High'])))

    wa = WaveAnalyzer(df=df, verbose=False)
    wave_options_impulse = WaveOptionsGenerator5(up_to=5)  # generates WaveOptions up to [15, 15, 15, 15, 15]

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
            # print('---%s' % id(wavepattern_up))

            for rule in rules_to_check:

                if wavepattern_up.check_rule(rule):
                    if wavepattern_up in wavepatterns_up:
                        # print(id(wavepattern_up))
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
                                      0,
                                      0
                                      ]
                            w_l.append(w_info)
                            df_wave = df_wave.append(pd.DataFrame([w_info], columns=df_wave.columns), ignore_index=True)
                            print(f'{rule.name} found: {new_option_impulse.values}')
                        # plot_pattern(df=df, wave_pattern=wavepattern_up, title=symbol + str(new_option_impulse))

nowpoint = '2021-12-25'
symbol = 'INFY.csv'
df_w_filterd, df_wave = agent_filter_and_update_wave_df(df_wave, symbol, nowpoint, df_symbols)
print(df_w_filterd, df_wave)

                            
                            
                            
        
