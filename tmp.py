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

def agent_filter_wave_df(df_wave, symbol, timepoint, df_flow):
    f1 = df_wave['symbol'] == symbol
    f2 = df_wave['time'] >= timepoint
    df_wave_f = df_wave[f1 & f2]
    return df_wave_f

df_wave = pd.DataFrame([], columns=['id', 'symbol', 'time', 'low', 'w1', 'w2', 'w3', 'w4', 'high', 'wave', 'position', 'valid'])

w_l = list()


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
                                          False,
                                          True
                                          ]
                                w_l.append(w_info)
                                df_wave = df_wave.append(pd.DataFrame([w_info], columns=df_wave.columns), ignore_index=True)
                                print(f'{rule.name} found: {new_option_impulse.values}')
                            


timepoint = '2018-05-29'
symbol = 'AMD.csv'
df_flow = None
df_w_filterd = agent_filter_wave_df(df_wave, symbol, timepoint, df_flow)
print(df_w_filterd)
                            
                            
                            
                            
        
