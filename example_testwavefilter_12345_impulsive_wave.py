from __future__ import annotations
from models.WavePattern import WavePattern
from models.WaveRules import Impulse, LeadingDiagonal
from models.WaveAnalyzer import WaveAnalyzer
from models.WaveOptions import WaveOptionsGenerator5
from models.helpers import plot_pattern
import pandas as pd
import numpy as np

# df = pd.read_csv(r'data\btc-usd_1d.csv')



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

def agent_filter_symbol_time(df_wave, symbol, timepoint, df_flow):
    df_wave_f = None
    if symbol and timepoint:
        f1 = df_wave['symbol'] == symbol
        f2 = df_wave['time'] >= timepoint
        df_wave_f = df_wave[f1 & f2]
    return df_wave_f

def agent_filter_beyond_low_high(df_wave, symbol, timepoint, df_flow):

    df_wave_f = None
    if symbol and timepoint:
        f1 = df_wave['symbol'] == symbol
        f2 = df_wave['time'] >= timepoint
        df_wave_f = df_wave[f1 & f2]
    df_flow = df_flow[df_flow['Date'] <= timepoint]
    df_wave_f['valid'] = df_wave_f.apply(lambda x: False if np.min(df_flow[df_flow['Date'] > x['time']].Low.tolist() < x['time'].low) else True)
    df_wave_f['valid'] = df_wave_f.apply(lambda x: False if np.max(df_flow[df_flow['Date'] > x['time']].Low.tolist() < x['time'].high) else True)
    return df_wave_f




df_wave = pd.DataFrame([], columns=['id', 'symbol', 'time', 'low', 'w1', 'w2', 'w3', 'w4', 'high', 'wave', 'position', 'valid'])

w_l = list()


symbols = ['TWTR.csv',
            'FB.csv',
           'btc-usd_1d.csv',
           # 'AMD.csv', 'MTDR.csv', 'FSV.csv', 'LB.csv', 'LYFT.csv', 'SINA.csv', 'GOOG.csv', 'TSLA.csv', 'CPRT.csv'
           ]

df_symbols = []
for symbol in symbols:
    print(symbol)
    df = pd.read_csv(symbol)
    df_symbols.append(df)
    idx_start = np.argmin(np.array(list(df['Low'])))

    wa = WaveAnalyzer(df=df, verbose=False)
    wave_options_impulse = WaveOptionsGenerator5(up_to=1)  # generates WaveOptions up to [15, 15, 15, 15, 15]

    impulse = Impulse('impulse')
    rules_to_check = [impulse]

    print(f'Start at idx: {idx_start}')
    print(f"will run up to {wave_options_impulse.number / 1e6}M combinations.")

    # set up a set to store already found wave counts
    # it can be the case, that 2 WaveOptions lead to the same WavePattern.
    # This can be seen in a chart, where for example we try to skip more maxima as there are. In such a case
    # e.g. [1,2,3,4,5] and [1,2,3,4,10] will lead to the same WavePattern (has same sub-wave structure, same begin / end,
    # same high / low etc.
    # If we find the same WavePattern, we skip and do not plot it

    wavepatterns_up = set()
    w_l = list()
    # loop over all combinations of wave options [i,j,k,l,m] for impulsive waves sorted from small, e.g.  [0,1,...] to
    # large e.g. [3,2, ...]
    for new_option_impulse in wave_options_impulse.options_sorted:

        waves_up = wa.find_impulsive_wave(idx_start=idx_start, wave_config=new_option_impulse.values)

        if waves_up:
            wavepattern_up = WavePattern(waves_up, verbose=True)

            for rule in rules_to_check:

                if wavepattern_up.check_rule(rule):
                    if wavepattern_up in wavepatterns_up:
                        continue
                    else:
                        wavepatterns_up.add(wavepattern_up)
                        # if not check_has_same_wavepattern_df(df_wave, wavepattern_up):
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
                                  False,
                                  True
                                  ]
                        w_l.append(w_info)
                        df_wave = df_wave.append(pd.DataFrame([w_info], columns=df_wave.columns), ignore_index=True)
                        print(f'{rule.name} found: {new_option_impulse.values}')

                        # plot_pattern(df=df, wave_pattern=wavepattern_up, title=symbol + str(new_option_impulse))
print('impulsive wave end')


timepoint = '2018-06-14'
# symbol = 'AMD.csv'
df_flow = None
# df_w_filterd = agent_filter_symbol_time(df_wave, symbol, timepoint, df_flow)
df_TWTR = df_symbols[0]
df_flow = df_TWTR
df_w_filterd = agent_filter_beyond_low_high(df_wave, symbol, timepoint, df_flow)
print(df_w_filterd)