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

def entry_wave_df(df, df_ohlcv=None):
    r_df = None

    return df

dfw = pd.DataFrame([], columns=['id', 'symbol', 'time', 'low', 'w1', 'w2', 'w3', 'w4', 'high', 'wave', 'position'])




                            wavepatterns_up.add(wavepattern_up)
                            # if not check_has_same_wavepattern(w_l, wavepattern_up):
                            if not check_has_same_wavepattern_df(dfw, wavepattern_up):

                                w_info = [id(wavepattern_up),
                                          symbol,
                                          wavepattern_up.dates[0],
                                          wavepattern_up.low,
                                          wavepattern_up.values[1],
                                          wavepattern_up.values[3],
                                          wavepattern_up.values[5],
                                          wavepattern_up.values[7],
                                          wavepattern_up.high,
                                          wavepattern_up, False]
                                w_l.append(w_info)
                                dfw = dfw.append(pd.DataFrame([w_info], columns=dfw.columns), ignore_index=True)
        
