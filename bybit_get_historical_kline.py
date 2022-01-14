# this code is based on get_historical_data() from python-binance module 
# https://github.com/sammchardy/python-binance
# it also requires pybybit.py available from this page 
# https://note.mu/mtkn1/n/n9ef3460e4085 
# (where pandas & websocket-client are needed) 

import time
import dateparser
import pytz
import json
import csv
import pandas as pd 
from datetime import datetime 


def get_historical_klines(symbol, interval, start_str, end_str=None):
    """Get Historical Klines from Bybit 

    See dateparse docs for valid start and end string formats http://dateparser.readthedocs.io/en/latest/

    If using offset strings for dates add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"

    :param symbol: Name of symbol pair -- BTCUSD, ETCUSD, EOSUSD, XRPUSD 
    :type symbol: str
    :param interval: Bybit Kline interval -- 1 3 5 15 30 60 120 240 360 720 "D" "M" "W" "Y"
    :type interval: str
    :param start_str: Start date string in UTC format
    :type start_str: str
    :param end_str: optional - end date string in UTC format
    :type end_str: str

    :return: list of OHLCV values

    """

    # set parameters for kline() 
    timeframe = str(interval)
    limit    = 200
    start_ts = int(date_to_milliseconds(start_str)/1000)
    end_ts = None
    if end_str:
        end_ts = int(date_to_milliseconds(end_str)/1000)
    else: 
        end_ts = int(date_to_milliseconds('now')/1000)


    # init our list
    output_data = []

    # loop counter 
    idx = 0
    # it can be difficult to know when a symbol was listed on Binance so allow start time to be before list date
    symbol_existed = False
    while True:
        # fetch the klines from start_ts up to max 200 entries 
        temp_dict = bybit.kline(symbol=symbol, interval=timeframe, _from=start_ts, limit=limit)

        # handle the case where our start date is before the symbol pair listed on Binance
        if not symbol_existed and len(temp_dict):
            symbol_existed = True

        if symbol_existed:
            # extract data and convert to list 
            temp_data = [list(i.values())[2:] for i in temp_dict['result']]
            # append this loops data to our output data
            output_data += temp_data

            # update our start timestamp using the last value in the array and add the interval timeframe
            # NOTE: current implementation ignores inteval of D/W/M/Y  for now 
            start_ts = temp_data[len(temp_data) - 1][0] + interval*60

        else:
            # it wasn't listed yet, increment our start date
            start_ts += timeframe

        idx += 1
        # check if we received less than the required limit and exit the loop
        if len(temp_data) < limit:
            # exit the while loop
            break

        # sleep after every 3rd call to be kind to the API
        if idx % 3 == 0:
            time.sleep(0.2)

    return output_data

def get_historical_klines_pd(symbol, interval, start_str, end_str=None):
    """Get Historical Klines from Bybit 

    See dateparse docs for valid start and end string formats 
    http://dateparser.readthedocs.io/en/latest/

    If using offset strings for dates add "UTC" to date string 
    e.g. "now UTC", "11 hours ago UTC"

    :param symbol: Name of symbol pair -- BTCUSD, ETCUSD, EOSUSD, XRPUSD 
    :type symbol: str
    :param interval: Bybit Kline interval -- 1 3 5 15 30 60 120 240 360 720 "D" "M" "W" "Y"
    :type interval: str
    :param start_str: Start date string in UTC format
    :type start_str: str
    :param end_str: optional - end date string in UTC format
    :type end_str: str

    :return: list of OHLCV values

    """

    # set parameters for kline() 
    timeframe = str(interval)
    limit    = 200
    start_ts = int(date_to_milliseconds(start_str)/1000)
    end_ts = None
    if end_str:
        end_ts = int(date_to_milliseconds(end_str)/1000)
    else: 
        end_ts = int(date_to_milliseconds('now')/1000)


    # init our list
    output_data = []

    # loop counter 
    idx = 0
    # it can be difficult to know when a symbol was listed on Binance so allow start time to be before list date
    symbol_existed = False
    while True:
        # fetch the klines from start_ts up to max 200 entries 
        temp_dict = bybit.kline(symbol=symbol, interval=timeframe, _from=start_ts, limit=limit)

        # handle the case where our start date is before the symbol pair listed on Binance
        if not symbol_existed and len(temp_dict):
            symbol_existed = True

        if symbol_existed:
            # extract data and convert to list 
            temp_data = [list(i.values())[2:] for i in temp_dict['result']]
            # append this loops data to our output data
            output_data += temp_data

            # update our start timestamp using the last value in the array and add the interval timeframe
            # NOTE: current implementation does not support inteval of D/W/M/Y
            start_ts = temp_data[len(temp_data) - 1][0] + interval*60

        else:
            # it wasn't listed yet, increment our start date
            start_ts += timeframe

        idx += 1
        # check if we received less than the required limit and exit the loop
        if len(temp_data) < limit:
            # exit the while loop
            break

        # sleep after every 3rd call to be kind to the API
        if idx % 3 == 0:
            time.sleep(0.2)

    # convert to data frame 
    df = pd.DataFrame(output_data, columns=['TimeStamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'TurnOver'])
    df['Date'] = [datetime.fromtimestamp(i).strftime('%Y-%m-%d %H:%M:%S.%d')[:-3] for i in df['TimeStamp']]

    return df

