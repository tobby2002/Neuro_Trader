import os
import time
import pandas as pd
import sqlite3
import threading
import pytz
from binance.client import Client
import datetime as dt
from tqdm import tqdm
import re
from sqlite3 import Error
from datetime import timedelta, datetime, timezone
# from src import allowed_range, retry, delta, load_data, resample
# from src.bitmex import BitMex
from datetime import date, timedelta

import json
with open('config.json', 'r') as f:
    config = json.load(f)

OHLC_DIRNAME = os.path.join(os.path.dirname(__file__), "./data/{}")
OHLC_FILENAME = os.path.join(os.path.dirname(__file__), "./data/{}/data.csv")

'''
http://www.sqlitetutorial.net/sqlite-python/sqlite-python-select/
'''

package_dir = os.path.abspath(os.path.dirname(__file__))
db_dir = os.path.join(package_dir, 'binance_futures.sqlite')
conn = sqlite3.connect(db_dir)



################
# init
################
futures = config['default']['futures']
rule = config['default']['rule']
leverage = config['default']['leverage']

high_target = config['default']['high_target']
low_target = config['default']['low_target']
low_target_w2 = config['default']['low_target_w2']

seed = config['default']['seed']
fee = config['default']['fee']
fee_maker = config['default']['fee_maker']
fee_taker = config['default']['fee_taker']
fee_slippage = config['default']['fee_slippage']

period_days_ago = config['default']['period_days_ago']
period_days_ago_till = config['default']['period_days_ago_till']
period_interval = config['default']['period_interval']

round_trip_flg = config['default']['round_trip_flg']
round_trip_count = config['default']['round_trip_count']
compounding = config['default']['compounding']

exchange = config['default']['exchange']
timeframe = config['default']['timeframe']
up_to_count = config['default']['up_to_count']
long = config['default']['long']
h_fibo = config['default']['h_fibo']
l_fibo = config['default']['l_fibo']
symbol_random = config['default']['symbol_random']
symbol_length = config['default']['symbol_length']

basic_secret_key = config['basic']['secret_key']
basic_secret_value = config['basic']['secret_value']
futures_secret_key = config['futures']['secret_key']
futures_secret_value = config['futures']['secret_value']

printout = config['default']['printout']
symbols = config['symbols']['binance_futures']


class DataManagerfeeder:

    def __init__(self, *args, **kwargs):
        # print('DataManagerfeeder start')
        pass

    def get_conn(self):
        dbpath = db_dir
        try:
            conn = sqlite3.connect(dbpath)
            return conn
        except Error as e:
            print(e)
        return None

    def get_historical_ohlc_data_start_end(self, symbol, start_str, end_str, past_days=None, interval=None, futures=False):
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

            # if not interval:
            #     interval = '1h'  # default interval 1 hour
            # if not past_days:
            #     past_days = 30  # default past days 30.

            start_str = str((pd.to_datetime('today') - pd.Timedelta(str(start_str) + ' days')).date())
            if end_str:
                end_str = str((pd.to_datetime('today') - pd.Timedelta(str(end_str) + ' days')).date())
            else:
                end_str = None
            print(start_str, end_str)
            D = None
            try:
                if futures:
                    D = pd.DataFrame(
                        client.futures_historical_klines(symbol=symbol, start_str=start_str, end_str=end_str,
                                                         interval=interval))
                else:
                    D = pd.DataFrame(client.get_historical_klines(symbol=symbol, start_str=start_str, end_str=end_str,
                                                                  interval=interval))
                pass
            except Exception as e:
                time.sleep(1)
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
            # pass
        return D, start_str, end_str

    def create_newtable(self, conn, exchange, symbol, bin_size):
        cur = conn.cursor()
        '''
        timestamp,open,high,low,close,volume
        2019-03-23 07:53:00+00:00,3995.0,3994.5,3992.5,3992.5,415222
        2019-03-23 07:54:00+00:00,3992.5,3993.0,3992.5,3993.0,82759
        2019-03-23 07:55:00+00:00,3993.0,3993.0,3989.5,3990.0,2429478
        2019-03-23 07:56:00+00:00,3990.0,3990.0,3988.5,3989.0,866484
        
        CREATE TABLE %s(
            timestamp TEXT,
            open NUMERIC,
            high NUMERIC,
            low NUMERIC,
            close NUMERIC,
            volume NUMERIC
        );
        
        /* 데이터 넣기 */
        # INSERT INTO bitmex_1m(timestamp,open,high,low,close,volume)VALUES('2019-03-23 07:53:00+00:00',3995.0,3994.5,3992.5,3992.5,415222);
        # INSERT INTO bitmex_1m(timestamp,open,high,low,close,volume)VALUES('2019-03-23 07:54:00+00:00',3992.5,3993.0,3992.5,3993.0,82759);
        # INSERT INTO bitmex_1m(timestamp,open,high,low,close,volume)VALUES('2019-03-23 07:53:00+00:00',3995.0,3994.5,3992.5,3992.5,415222);
        
        '''
        cur.executescript("""
        /* 테이블이 이미 있다면 제거하기 */
        /* DROP TABLE IF EXISTS bitmex_1m;*/
        /* 테이블 생성하기 */
    
        CREATE TABLE %s_%s_%s(
            Date TIMESTAMP,
            Open REAL,
            High REAL,
            Low REAL,
            Close REAL,
            Volume INTEGER
        );
        """ % (exchange, symbol, bin_size))
        print('create_table : %s_%s_%s' % (exchange, symbol, bin_size))
        # conn.commit()
        # conn.close()

    def create_wave(self, conn):
        cur = conn.cursor()
        '''
        ['id', 'symbol', 'timeframe', 'type', 'time', 'low', 'w1', 'w2', 'w3', 'w4', 'high', 'wave', 'position', 'valid']
        CREATE TABLE wave(
            id TEXT,
            symbol TEXT,
            timeframe TEXT,
            type TEXT,            
            time TIMESTAMP,
            low REAL,
            w1 REAL,
            w2 REAL,
            w3 REAL,
            w4 REAL,
            high REAL,
            wave TEXT,
            position NUMERIC,
            valid NUMERIC
        );

        '''
        cur.executescript("""
        /* 테이블이 이미 있다면 제거하기 */
        /* DROP TABLE IF EXISTS bitmex_1m;*/
        /* 테이블 생성하기 */

        CREATE TABLE wave(
            id TEXT,
            symbol TEXT,
            timeframe TEXT,
            type TEXT,
            time TIMESTAMP,
            low REAL,
            w1 REAL,
            w2 REAL,
            w3 REAL,
            w4 REAL,
            high REAL,
            wave TEXT,
            position NUMERIC,
            valid NUMERIC
        );
        """)
        print('created table : wave')
        # conn.commit()
        # conn.close()

    def load_df_wave(self, conn):
        cur = conn.cursor()
        query = cur.execute("SELECT * From wave")
        cols = [column[0] for column in query.description]
        query_df = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
        return query_df

    def save_wave(self, w_info, conn):
        c = conn.cursor()
        col = [w_info[0], w_info[1], w_info[2], w_info[3], w_info[4], w_info[5], w_info[6], w_info[7], w_info[8], w_info[9], \
        w_info[10], w_info[11], w_info[12], w_info[13]]
        c.execute('INSERT INTO wave(id, symbol, timeframe, "type", "time", low, w1, w2, w3, w4, high, wave, "position", valid ) VALUES (?, ?,?,?,?,?,?,?,?,?,?,?,?,?)', col)
        conn.commit()


    def load_df_wave_startpoint_nowpoint_window(self, conn, timeframe, type, startpoint, nowpoint, window):
        cur = conn.cursor()
        window = window * int(re.sub(r'[^0-9]', '', timeframe))
        window_start_date = str((pd.to_datetime(nowpoint) - pd.Timedelta(str(window) + ' minutes')))
        if startpoint:
            sql = "SELECT * From wave WHERE timeframe = '%s' and type = '%s'  and time >= '%s' and time >= '%s' and time <= '%s'" % (timeframe, type, window_start_date, startpoint, nowpoint)
        else:
            sql = "SELECT * From wave WHERE timeframe = '%s' and type = '%s'  and time >= '%s' and time <= '%s'" % (timeframe, type, window_start_date, nowpoint)
        query = cur.execute(sql)
        cols = [column[0] for column in query.description]
        query_df = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
        return query_df

    def truncate_tablename(self, conn, tablename):
        cur = conn.cursor()
        cur.executescript("""
        DELETE FROM %s;
        VACUUM;
        """ % (tablename))
        conn.commit()

    def drop_table(self, conn, exchange, symbol, bin_size):
        cur = conn.cursor()
        cur.executescript("""
        DROP TABLE IF EXISTS %s_%s_%s;
        """ % (exchange, symbol, bin_size))
        # conn.commit()
        # conn.close()
        print('drop_table : %s_%s_%s' % (exchange, symbol, bin_size))

    def truncate_table(self, conn, exchange, symbol, bin_size):
        cur = conn.cursor()
        cur.executescript("""
        DELETE FROM %s_%s_%s;
        VACUUM;
        """ % (exchange, symbol, bin_size))
        conn.commit()
        conn.close()
        print('truncate_table : %s_%s_%s' % (exchange, symbol, bin_size))


    # def test_fetch_ohlcv_1m(self, mins):
    #     '''
    #     source => DataFrame Type
    #     ------------------------------------------------------------------
    #                                open    high     low   close   volume
    #     2019-06-15 14:29:00+00:00  8669.5  8670.0  8667.0  8667.0  1454667
    #     2019-06-15 14:30:00+00:00  8667.0  8667.5  8667.0  8667.5   424940
    #     :return:
    #     '''
    #     # bitmex = BitMex(threading=False)
    #     bitmex = BitMex(threading=True)
    #     end_time = datetime.now(timezone.utc)
    #     start_time = end_time - 1 * timedelta(minutes=mins)
    #     source = bitmex.fetch_ohlcv('1m', start_time, end_time)
    #     return source

    '''
        # download and save to database
        # conn = get_conn()
        # df = download_df_ohlcv('1m', 129600, threading=True)  # 1m, 1000건, 현재까지 e.g. 60*24=1440 -> 1D -> (1440*30)=43200 1MONTH, 129600 3MONTH,
        # save_to_db(conn, df, 'bitmex_1m', 'replace')
        # show_table(conn, 'bitmex_1m')  # show table content
    '''
    # def download_df_ohlcv(self, bin_size, ohlcv_len, **kwargs):
    #     '''
    #     df => DataFrame Type
    #     ------------------------------------------------------------------
    #                                open    high     low   close   volume
    #     2019-06-15 14:29:00+00:00  8669.5  8670.0  8667.0  8667.0  1454667
    #     2019-06-15 14:30:00+00:00  8667.0  8667.5  8667.0  8667.5   424940
    #     :return:
    #     '''
    #     print('download data from server')
    #     start = time.time()
    #     # bitmex = BitMex(threading=False)
    #     bitmex = BitMex(threading=True)
    #     end_time = datetime.now(timezone.utc)
    #     start_time = end_time - ohlcv_len * delta(bin_size)
    #     df = bitmex.fetch_ohlcv(bin_size, start_time, end_time)
    #     print('download_df_ohlcv time:', time.time() - start)
    #     return df

    def save_to_db(self, conn, df, tablename, replace):  # if_exists='replace' or if_exists='append'
        # start = time.time()
        df.to_sql(tablename, conn, if_exists=replace)
        # print('save_data time:', time.time() - start)

    def load_df(self, conn, table, bin_size):
        cur = conn.cursor()
        query = cur.execute("SELECT * From %s_%s " % (table, bin_size))
        cols = [column[0] for column in query.description]
        query_df = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
        return query_df

    def load_df_candle_startpoint_endpoint(self, conn, table, bin_size, startpoint, endpoint, window):
        cur = conn.cursor()
        window = window * int(re.sub(r'[^0-9]', '', bin_size))
        window_start_date = str((pd.to_datetime(endpoint) - pd.Timedelta(str(window) + ' minutes')))
        if startpoint:
            sql = "SELECT * From %s_%s WHERE Date >= '%s' and Date >= '%s' and Date <= '%s' " % (table, bin_size, window_start_date, startpoint, endpoint)
        else:
            sql = "SELECT * From %s_%s WHERE Date >= '%s' and Date <= '%s' " % (table, bin_size, window_start_date, endpoint)
        query = cur.execute(sql)
        cols = [column[0] for column in query.description]
        query_df = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
        return query_df

    def load_df_ohlc_startpoint_nowpoint_window(self, conn, table, bin_size, startpoint, nowpoint, window):
        cur = conn.cursor()
        if startpoint:
            sql = "SELECT * From %s_%s WHERE Date >= '%s' and Date <= '%s' ORDER BY ROWID DESC LIMIT %d" % (table, bin_size, startpoint, nowpoint, window)
        else:
            sql = "SELECT * From %s_%s WHERE Date <= '%s' ORDER BY ROWID DESC LIMIT %d" % (table, bin_size, nowpoint, window)
        query = cur.execute(sql)
        cols = [column[0] for column in query.description]
        query_df = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
        return query_df

    def load_df_last_n(self, conn, table, bin_size, n):
        cur = conn.cursor()
        query = cur.execute("SELECT * From %s_%s ORDER BY ROWID DESC LIMIT %d" % (table, bin_size, n))
        cols = [column[0] for column in query.description]
        query_df = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
        return query_df[::-1]

    def load_df_last_row(self, conn, table, bin_size):
        cur = conn.cursor()
        query = cur.execute("SELECT ROWID, * From %s_%s ORDER BY ROWID DESC LIMIT 1" % (table, bin_size))
        cols = [column[0] for column in query.description]
        query_df = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
        return query_df


    def utc2local(self, data):
        data['timestamp'] = data['index'].apply(lambda x: x[0:19])
        data.index = pd.DatetimeIndex(data.timestamp, name='timestamp').tz_localize('UTC').tz_convert('Asia/Seoul')
        data.timestamp = data.index
        data = data.reset_index(drop=True)
        return data


    def save_to_csv(self, df, filename):
        print('save_to_csv:', str("%s.csv" % filename))
        start = time.time()
        df.to_csv("%s.csv" % filename, mode='w')  # https://buttercoconut.xyz/74/
        print('download_data time:', time.time() - start)


    def load_last_n_df(self, conn, table, bin_size, n):
        cur = conn.cursor()
        query = cur.execute("select ROWID, * from '%s_%s' order by ROWID desc limit %d" % (table, bin_size, n))
        cols = [column[0] for column in query.description]
        query_df = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
        return query_df

    def show_table(conn, table, bin_size, limit):
        cur = conn.cursor()
        if limit > 0:
            sql = str("select ROWID, * from '%s_%s' order by ROWID desc limit %d" % (table, bin_size, limit))
            cur.execute(sql)
            item_list = cur.fetchall()
        else:
            cur.execute("SELECT ROWID, * FROM %s_%s" % (table, bin_size))
            item_list = cur.fetchall()

        for it in item_list:
            print(it)
        print('%s_%s table, %s 건' % (table, bin_size, len(item_list)))

    def download_save_show_by_1m(self, conn, table, bin_size, ohlcv_len):
        print('download data from server')
        start = time.time()
        source = self.test_fetch_ohlcv_1m(ohlcv_len)  # 1min 짜리 1000건
        print('download_data time:', time.time() - start)

        start = time.time()
        source.to_sql('bitmex_1m', conn, if_exists='replace')
        print('save_data time:', time.time() - start)
        self.show_table(conn, table, ohlcv_len, 10)

    # def autorelay_download_save(self, conn, table, bin_size, ohlcv_len, append):
    #     cur = conn.cursor()
    #     startd = time.time()
    #     bitmex = BitMex(threading=True)
    #     sql = str("select ROWID, * from '%s_%s' order by ROWID desc limit 1" % (table, bin_size))
    #     cur.execute(sql)
    #     item_list = cur.fetchall()
    #     end_time = datetime.now(timezone.utc)
    #     start_time = end_time - ohlcv_len * delta(bin_size)
    #
    #     # 데이터가 있으면 릴레이 다운로드 자동으로 기간설정
    #     if item_list:
    #         lasttime = item_list[0][1]
    #         # print(lasttime)
    #         # print(lasttime[0:16])
    #         last_time = datetime.strptime(lasttime[0:16], '%Y-%m-%d %H:%M')
    #         start_time = last_time + timedelta(minutes=1)
    #     else:
    #         # 데이터가 없으면,
    #         print('No data and will start from ', start_time)
    #         pass
    #
    #     print('start_time:', start_time)
    #     print('end_time:', end_time)
    #
    #     # relay download 디폴트 기간 설정만큼 다운로드 한다.
    #     df = bitmex.fetch_ohlcv(bin_size, start_time, end_time)
    #     print('download_df_ohlcv time: ', time.time() - startd)
    #
    #     # insert to database
    #     df.to_sql(table+'_'+bin_size, conn, if_exists=append)
    #     self.show_table(conn, table, bin_size, 1)

    def run_downloader_by_thread_1m(self):
        print('=====', time.ctime(), '== run downloader by thread ===')
        try:
            # 1036800 => 약 2 년 | 518400=> 약1년 | 129600 => 약3개월 | 43200 => 약1개월
            self.autorelay_download_save(conn, 'bitmex', '1m', 43200, 'append')
            conn.commit()
            conn.close()
        except Exception as e:  # 에러 종류
            print('In run_downloader_by_thread : ex, ', e)
        threading.Timer(1, self.run_downloader_by_thread_1m).start()



################
# data collector
################
def data_init(dm, symbols, exchange, bin_size, start_str, end_str, past_days, futures):  #, drop_n_creat=True):
    end_str = None if end_str is None or (end_str < 0) else end_str
    conn = dm.get_conn()
    for i, symbol in tqdm(enumerate(symbols)):
        start = time.perf_counter()
        print('download %s %s/%s ' % (symbol, str(i), str(len(symbols))))
        try:
            df, start_date, end_date = dm.get_historical_ohlc_data_start_end(symbol, start_str=start_str,
                                                                              end_str=end_str, past_days=past_days,
                                                                              interval=bin_size, futures=futures)

            if df is not None and df.empty is not True:
                symbol = symbol.lower()
                tablename = exchange + '_' + symbol + '_' + bin_size
                dm.drop_table(conn, exchange, symbol, bin_size)
                dm.create_newtable(conn, exchange, symbol, bin_size)  # drop and create new table by tablename
                dm.save_to_db(conn, df, tablename, 'replace')
                # df = dm.load_df(conn, tablename, bin_size)
                # symbols_d[symbol] = df
                print('download %s %s/%s done' % (symbol, str(i), str(len(symbols))))
                finish = time.perf_counter()
                print(f'Finished in {symbol} {round(finish - start, 2)} second(s)')
        except Exception as e:
            print('data_init:%s' % e)
            print('download %s %s - %s fail' % (symbol, str(i), str(len(symbols))))
            continue
    finish = time.perf_counter()
    print(f'Finished in ALL DOWNLOAD : {round(finish - start, 2)} second(s)')
    conn.commit()
    conn.close()
    print('data_init done')
    # return symbols_d

if __name__ == "__main__":
    '''
    # https://docs.python.org/ko/3/library/sqlite3.html
    # https://wikidocs.net/5332
    # https://tariat.tistory.com/9
    '''
    # try:
    #     # from src import allowed_range, retry, delta, load_data, resample
    #     # from src.bitmex import BitMex
    #     from datamanagerfeeder_ing import DataManagerfeeder
    # except:
    #     from ..src import allowed_range, retry, delta, load_data, resample
    #     from ..src.bitmex import BitMex
    #     from ..src.datamanagerfeeder_ing import DataManagerfeeder


    dm = DataManagerfeeder()
    '''
    # drop table
    '''
    # conn = get_conn()
    # drop_table(conn, 'bitmex', '1m')

    '''
    # create table
    '''
    # conn = get_conn()
    # create_newtable(conn, 'bitmex', '1m')  # drop and create new table by tablename
    '''
    # truncate table
    '''
    # conn = get_conn()
    # truncate_table(conn, 'bitmex', '1m')

    '''
    # show table
    '''
    # conn = get_conn()
    # show_table(conn, 'bitmex', '1m', 10)  # 0 --> all list

    '''
    # download and save to database and show data
    '''
    # conn = get_conn()
    # periods = 30000
    # df = download_df_ohlcv('1m', periods, threading=True)  # 1m, periods:1440건, 현재까지 e.g. 60*24=1440 -> 1D / (1440*30)=43200 1MONTH / 129600 3MONTH,

    # save_to_db(conn, df, 'bitmex_1m', 'replace')
    # show_table(conn, 'bitmex, '1m', 10)

    # df = load_df(conn, 'bitmex', '1m')
    # save_to_csv(df, 'bitmex_1m_30000_from_table')

    '''
    # auto relay download & save to database and show data
    '''
    # autorelay_download_save(get_conn(), 'bitmex', '1m', 5, 'append')

    '''
    # By Thread, auto relay download & save to database and show data
    '''
    # dm.run_downloader_by_thread_1m()

    pass

    futures = config['default']['futures']
    rule = config['default']['rule']
    leverage = config['default']['leverage']

    high_target = config['default']['high_target']
    low_target = config['default']['low_target']
    low_target_w2 = config['default']['low_target_w2']

    seed = config['default']['seed']
    fee = config['default']['fee']
    fee_maker = config['default']['fee_maker']
    fee_taker = config['default']['fee_taker']
    fee_slippage = config['default']['fee_slippage']

    period_days_ago = config['default']['period_days_ago']
    period_days_ago_till = config['default']['period_days_ago_till']
    period_interval = config['default']['period_interval']

    round_trip_flg = config['default']['round_trip_flg']
    round_trip_count = config['default']['round_trip_count']
    compounding = config['default']['compounding']

    timeframe = config['default']['timeframe']
    up_to_count = config['default']['up_to_count']
    long = config['default']['long']
    h_fibo = config['default']['h_fibo']
    l_fibo = config['default']['l_fibo']
    symbol_random = config['default']['symbol_random']
    symbol_length = config['default']['symbol_length']

    basic_secret_key = config['basic']['secret_key']
    basic_secret_value = config['basic']['secret_value']
    futures_secret_key = config['futures']['secret_key']
    futures_secret_value = config['futures']['secret_value']

    printout = config['default']['printout']
    printout = config['default']['printout']
    symbols = config['symbols']['binance_futures']


    def print_condition():
        print('-------------------------------')
        print('futures:%s' % str(futures))
        print('rule:%s' % str(rule))
        print('leverage:%s' % str(leverage))
        print('seed:%s' % str(seed))
        print('fee:%s%%' % str(fee * 100))
        print('fee_maker:%s%%' % str(fee_maker * 100))
        print('fee_taker:%s%%' % str(fee_taker * 100))
        print('fee_slippage:%s%%' % str(round(fee_slippage * 100, 4)))
        if futures:
            fee_high = (
                                   fee_taker + fee_taker + fee_slippage) * leverage  # fee_maker buy:0.02%(0.0002) sell:0.02%(0.0002), sleepage:0.01%(0.0001)
            fee_low = (
                                  fee_taker + fee_maker + fee_slippage) * leverage  # fee_maker buy:0.02%(0.0002) sell:0.02%(0.0002), sleepage:0.01%(0.0001)
            print('(fee_high+fee_low)*100/2 + slippage:%s%%' % round(
                (float(fee_high) + float(fee_low)) * 100 / 2 + fee_slippage * 100, 4))

        else:
            fee_high = (
                                   fee_taker + fee_taker + fee_slippage) * leverage  # fee_maker buy:0.02%(0.0002) sell:0.02%(0.0002), sleepage:0.01%(0.0001)
            fee_low = (
                                  fee_taker + fee_maker + fee_slippage) * leverage  # fee_maker buy:0.02%(0.0002) sell:0.02%(0.0002), sleepage:0.01%(0.0001)
            print('(fee_high+fee_low)*100/2 + slippage:%s%%' % round(
                (float(fee_high) + float(fee_low)) * 100 / 2 + fee_slippage * 100, 4))

        print('timeframe: %s' % timeframe)
        print('period_days_ago: %s' % period_days_ago)
        # print('period_days_ago_till: %s' % period_days_ago_till)
        print('period_interval: %s' % period_interval)
        print('round_trip_count: %s' % round_trip_count)
        print('compounding: %s' % compounding)

        print('symbol_random: %s' % symbol_random)
        print('symbol_length: %s' % symbol_length)

        # print('timeframe: %s' % timeframe)
        start_dt = str((pd.to_datetime('today') - pd.Timedelta(str(period_days_ago) + ' days')).date())
        end_dt = str((pd.to_datetime('today') - pd.Timedelta(str(period_days_ago_till) + ' days')).date())
        print('period: %s ~ %s' % (start_dt, end_dt))
        print('up_to_count: %s' % up_to_count)
        # print('long: %s' % long)
        print('h_fibo: %s' % h_fibo)
        print('l_fibo: %s' % l_fibo)
        print('-------------------------------')


    ########################################
    # create create_wave
    # conn = dm.get_conn()
    # dm.create_wave(conn)
    # conn.commit()
    # conn.close()

    ########################################
    # exchange = 'binance_futures'
    past_days = 1
    timeframe = 1  # 1, 5, 15
    timeunit = 'm'
    bin_size = str(timeframe) + timeunit
    start_str = 1
    end_str = None
    # if end_str < 0:
    #     end_str = None
    # data_init(dm, symbols, exchange, bin_size, start_str, end_str, None, futures)




    #########################
    # df, start_date, end_date = dm.get_historical_ohlc_data_start_end(symbol, start_str=start_str,
    #                                                               end_str=end_str, past_days=past_days,
    #                                                               interval=bin_size, futures=futures)
    #
    # tablename = exchange + '_' + bin_size
    #
    # '''
    # # drop table
    # '''
    # conn = dm.get_conn()
    # dm.drop_table(conn, exchange, bin_size)
    #
    # '''
    # # create table
    # '''
    # conn = dm.get_conn()
    # dm.create_newtable(conn,  exchange, bin_size)  # drop and create new table by tablename
    #
    # conn = dm.get_conn()
    # dm.save_to_db(conn, df, tablename, 'replace')
    # dm.show_table(conn, exchange, bin_size, 10)
    #
    # df = dm.load_df(conn, exchange, bin_size)
    #
    # print(df)
    # df_lows = fractals_low_loop(df_all)
    # df_lows = fractals_low_loop(df_lows)













