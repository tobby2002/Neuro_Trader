import ccxt
import json
from time import sleep
import bybitwrapper
from cryptofeed import FeedHandler
from cryptofeed.exchanges import (FTX, Binance, BinanceUS, BinanceFutures, Bitfinex, Bitflyer, AscendEX, Bitmex, Bitstamp, Bittrex, Coinbase, Gateio,
                                  HitBTC, Huobi, HuobiDM, HuobiSwap, Kraken, OKCoin, OKEx, Poloniex, Bybit, KuCoin, Bequant, Upbit, Probit)
with open('config.json', 'r') as f:
    config = json.load(f)
from prettyprinter import pprint


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
window = config['default']['window']

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

plotview = config['default']['plotview']
printout = config['default']['printout']
symbols = config['symbols']['binance_futures']

client = bybitwrapper.bybit(test=False, api_key='o2aZhUESAachytlOy5', api_secret='AZPK3dhKdNsRhHX2s80KxCaEsFzlt5cCuQdK')
pairs = Bybit.symbols()
find_USDT = 'USDT-PERP'
index_USDTs = [i for i in range(len(pairs)) if find_USDT in pairs[i]]
symbols_usdt_perp = []
symbols_usdt = []
symbols_none = []
for s in index_USDTs:
    symbols_usdt_perp.append(pairs[s])
    symbols_usdt.append(pairs[s].split('-')[0]+'USDT')
    symbols_none.append(pairs[s].split('-')[0])
pairs = symbols_usdt
coins = symbols_none


# set binance
exchange_id = 'binance'
exchange_class = getattr(ccxt, exchange_id)
binance = exchange_class({
    'apiKey': None,
    'secret': None,
    'timeout': 30000,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'},
})
binance.load_markets()


def check_positions(symbol):
    positions = client.LinearPositions.LinearPositions_myPosition(symbol=symbol+"USDT").result()
    if positions[0]['ret_msg'] == 'OK':
        for position in positions[0]['result']:
            if position['entry_price'] > 0:
                print("Position found for ", symbol, " entry price of ", position['entry_price'])
                return True, position
            else:
                pass

    else:
        print("API NOT RESPONSIVE AT CHECK ORDER")
        sleep(5)


def fetch_ticker(symbol):
    tickerDump = binance.fetch_ticker(symbol + '/USDT')
    ticker = float(tickerDump['last'])
    return ticker


def fetch_order_size(symbol, order_size_percent_balance, leverage):
    global qty
    wallet_info = client.Wallet.Wallet_getBalance(coin="USDT").result()
    balance = wallet_info[0]['result']['USDT']['wallet_balance']
    ticker = fetch_ticker(symbol)
    qtycalc = (balance / ticker) * leverage
    qty = qtycalc * (order_size_percent_balance / 100)
    return qty


def set_leverage(symbol, leverage, coins=None):
    if coins:
        for coin in coins:
            set = client.LinearPositions.LinearPositions_saveLeverage(symbol=coin+"USDT", buy_leverage=int(leverage), sell_leverage=int(leverage)).result()
            if not set or not (set[0]['ret_msg'] == 'OK' or set[0]['ret_msg'] == 'leverage not modified'):
                print('Could not set all coins for set_leverage especially about coin: %s' % coin)
                return False
    else:
        if symbol:
            set = client.LinearPositions.LinearPositions_saveLeverage(symbol=symbol+"USDT", buy_leverage=int(leverage), sell_leverage=int(leverage)).result()
            if not set or not (set[0]['ret_msg'] == 'OK' or set[0]['ret_msg'] == 'leverage not modified'):
                print('Could not set symbol[%s] set_leverage ' % symbol)
                return False
        else:
            return False
    if coins:
        print("Setting Leverage:",leverage,"x about coins[",coins ,"] before Starting Bot")
    else:
        if symbol:
            print("Setting Leverage:",leverage,"x about symbol [",symbol,"] before trade")
    return True

def place_order(symbol, side, size, price, order_type):
    print('*****************************************************')
    print(symbol, side, " Entry Found!! Placing new order!!")
    print('*****************************************************')
    size = round(size, 3)
    order = client.LinearOrder.LinearOrder_new(side=side, symbol=symbol+"USDT", order_type=order_type, qty=size,
                                               price=price,
                                       time_in_force="GoodTillCancel", reduce_only=False,
                                       close_on_trigger=False).result()

    pprint(order)

def calculate_order(symbol, side):
    position = check_positions(symbol)
    print('position:%s' % position)
    if position != None:
        if position[0] == True:
            position = position[1]
            pnl = float(position['unrealised_pnl'])

            if pnl < 0:
                ticker = fetch_ticker(symbol)
                percent_change = ticker - float(position['entry_price'])
                pnl = (percent_change / ticker) * -100
                print("PNL %", symbol, (-1 * pnl))
                min_order = fetch_order_size(symbol)
                print('min_order:%s'% min_order)
                # multipliers = load_multipliers(coins, symbol)
                multipliers = None
                # size1 = (min_order * multipliers[0])
                # size2 = (min_order * multipliers[1])
                # size3 = (min_order * multipliers[2])
                # size4 = (min_order * multipliers[3])

                # dca = load_dca(coins, symbol)
                # modifiers = load_dca_values(coins, symbol)
                # dca = None
                # modifiers = None
                # print(min_order)

                # print("Current Position Size for ", symbol, " = ", position['size'])
                # if position['size'] <= size1:
                #     size = min_order
                #     place_order(symbol, side, ticker, size)
                # elif size1 < position['size'] <= size2 and pnl > dca[0]:
                #     size = min_order * modifiers[0]
                #     place_order(symbol, side, ticker, size)
                # elif size2 < position['size'] <= size3 and pnl > dca[1]:
                #     size = min_order * modifiers[1]
                #     place_order(symbol, side, ticker, size)
                # elif size3 < position['size'] <= size4 and pnl > dca[2]:
                #     size = min_order * modifiers[2]
                #     place_order(symbol, side, ticker, size)
                # elif size4 < position['size'] and pnl > dca[3]:
                #     size = min_order * modifiers[3]
                #     place_order(symbol, side, ticker, size)
                # else:
                #     print("At Max Size for ", symbol, " Tier or Not Outside Drawdown Settings..")

            else:
                print(symbol, "Position is currently in profit so we wont do anything here    :D")

        else:
            print("SEARCH FOR ME THIS SHOULD NOT HAPPEN GNOME LOL")

    else:
        print("No Open Position Found Yet")
        ticker = fetch_ticker(symbol)
        order_size_percent_balance = 0.1
        min_order = fetch_order_size(symbol, order_size_percent_balance, leverage)
        place_order(symbol, side, ticker, min_order)


# set_leverage(None, leverage, coins=coins)
# set_leverage('BTC', leverage, coins=None)
symbol = 'ADA'
side = 'Buy'
size = 1
price = 1.4810
order_type = 'Limit'
place_order(symbol, side, size, price, order_type)
ordered = {
    'ret_code': 0,
    'ret_msg': 'OK',
    'ext_code': '',
    'ext_info': '',
    'result': {
        'order_id': 'a2c96037-ace3-4325-abf7-daf662f3ae9a',
        'user_id': 17371411,
        'symbol': 'ADAUSDT',
        'side': 'Buy',
        'order_type': 'Limit',
        'price': 1.5178,
        'qty': 1,
        'time_in_force': 'GoodTillCancel',
        'order_status': 'Created',
        'last_exec_price': 0,
        'cum_exec_qty': 0,
        'cum_exec_value': 0,
        'cum_exec_fee': 0,
        'reduce_only': False,
        'close_on_trigger': False,
        'order_link_id': '',
        'created_time': '2022-01-17T15:15:37Z',
        'updated_time': '2022-01-17T15:15:37Z',
        'take_profit': 0,
        'stop_loss': 0,
        'tp_trigger_by': 'UNKNOWN',
        'sl_trigger_by': 'UNKNOWN',
        'position_idx': 1
    },
    'time_now': '1642432537.329549',
    'rate_limit_status': 99,
    'rate_limit_reset_ms': 1642432537324,
    'rate_limit': 100
}
position = {'user_id': 17371411, 'symbol': 'ADAUSDT', 'side': 'Buy', 'size': 1, 'position_value': 1.5116, 'entry_price': 1.5116, 'liq_price': 0.0001, 'bust_price': 0.0001, 'leverage': 3, 'auto_add_margin': 0, 'is_isolated': False, 'position_margin': 412.8019905, 'occ_closing_fee': 8e-08, 'realised_pnl': -0.00090696, 'cum_realised_pnl': -0.00090696, 'free_qty': -1, 'tp_sl_mode': 'Partial', 'unrealised_pnl': -0.0067, 'deleverage_indicator': 2, 'risk_id': 116, 'stop_loss': 0, 'take_profit': 0, 'trailing_stop': 0, 'position_idx': 1, 'mode': 'BothSide'}
calculate_order(symbol, side)

2
nd

{
    'ret_code': 0,
    'ret_msg': 'OK',
    'ext_code': '',
    'ext_info': '',
    'result': {
        'order_id': '6088d8c6-e420-4be7-9b31-de6108393576',
        'user_id': 17371411,
        'symbol': 'ADAUSDT',
        'side': 'Buy',
        'order_type': 'Limit',
        'price': 1.5044,
        'qty': 2,
        'time_in_force': 'GoodTillCancel',
        'order_status': 'Created',
        'last_exec_price': 0,
        'cum_exec_qty': 0,
        'cum_exec_value': 0,
        'cum_exec_fee': 0,
        'reduce_only': False,
        'close_on_trigger': False,
        'order_link_id': '',
        'created_time': '2022-01-17T15:31:51Z',
        'updated_time': '2022-01-17T15:31:51Z',
        'take_profit': 0,
        'stop_loss': 0,
        'tp_trigger_by': 'UNKNOWN',
        'sl_trigger_by': 'UNKNOWN',
        'position_idx': 1
    },
    'time_now': '1642433511.244398',
    'rate_limit_status': 99,
    'rate_limit_reset_ms': 1642433511239,
    'rate_limit': 100
},

<

class 'tuple'>: ({'ret_code': 0, 'ret_msg': 'OK', 'ext_code': '', 'ext_info': '', 'result': [
    {'user_id': 17371411, 'symbol': 'ADAUSDT', 'side': 'Buy', 'size': 3, 'position_value': 4.5052,
     'entry_price': 1.50173333, 'liq_price': 0.0001, 'bust_price': 0.0001, 'leverage': 3, 'auto_add_margin': 0,
     'is_isolated': False, 'position_margin': 412.80019419, 'occ_closing_fee': 2.3e-07, 'realised_pnl': -0.00270312,
     'cum_realised_pnl': -0.00270312, 'free_qty': -3, 'tp_sl_mode': 'Partial', 'unrealised_pnl': -0.0169,
     'deleverage_indicator': 2, 'risk_id': 116, 'stop_loss': 0, 'take_profit': 0, 'trailing_stop': 0, 'position_idx': 1,
     'mode': 'BothSide'},
    {'user_id': 17371411, 'symbol': 'ADAUSDT', 'side': 'Sell', 'size': 0, 'position_value': 0, 'entry_price': 0,
     'liq_price': 0, 'bust_price': 0, 'leverage': 3, 'auto_add_margin': 0, 'is_isolated': False, 'position_margin': 0,
     'occ_closing_fee': 0, 'realised_pnl': -0.20784566, 'cum_realised_pnl': 5.78955289, 'free_qty': 0,
     'tp_sl_mode': 'Partial', 'unrealised_pnl': 0, 'deleverage_indicator': 0, 'risk_id': 116, 'stop_loss': 0,
     'take_profit': 0, 'trailing_stop': 0, 'position_idx': 2, 'mode': 'BothSide'}], 'time_now': '1642433553.927332',
                  'rate_limit_status': 119, 'rate_limit_reset_ms': 1642433553922, 'rate_limit': 120},
                 < bravado.requests_client.RequestsResponseAdapter object at 0x112584a00 >)





