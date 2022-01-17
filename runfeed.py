'''
Copyright (C) 2017-2021  Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.
'''
from decimal import Decimal

from cryptofeed import FeedHandler
from cryptofeed.backends.aggregate import OHLCV
from cryptofeed.defines import CANDLES, BID, ASK, BLOCKCHAIN, FUNDING, GEMINI, L2_BOOK, L3_BOOK, LIQUIDATIONS, OPEN_INTEREST, PERPETUAL, TICKER, TRADES, INDEX
from cryptofeed.exchanges import (FTX, Binance, BinanceUS, BinanceFutures, Bitfinex, Bitflyer, AscendEX, Bitmex, Bitstamp, Bittrex, Coinbase, Gateio,
                                  HitBTC, Huobi, HuobiDM, HuobiSwap, Kraken, OKCoin, OKEx, Poloniex, Bybit, KuCoin, Bequant, Upbit, Probit)
from cryptofeed.symbols import Symbol
from cryptofeed.exchanges.phemex import Phemex
# from cryptofeed.backends.postgres import CandlesPostgres, IndexPostgres, TickerPostgres, TradePostgres, OpenInterestPostgres, LiquidationsPostgres, FundingPostgres, BookPostgres
from cryptofeed.backends.postgresn import CandlesPostgres, IndexPostgres, TickerPostgres, TradePostgres, OpenInterestPostgres, LiquidationsPostgres, FundingPostgres, BookPostgres

async def candle_callback(c, receipt_timestamp):
    print(f"Candle received at {receipt_timestamp}: {c}")

def main():
    postgres_cfg = {'host': '127.0.0.1', 'user': 'postgres', 'db': 'postgres', 'pw': 'postgres'}
    config = {'log': {'filename': 'demo.log', 'level': 'DEBUG', 'disabled': False}}
    f = FeedHandler(config=config)
    pairs = Bybit.symbols()
    find_USDT = 'USDT-PERP'
    index_USDTs = [i for i in range(len(pairs)) if find_USDT in pairs[i]]
    symbols_usdt = []
    for s in index_USDTs:
        symbols_usdt.append(pairs[s])
    pairs = symbols_usdt
    f.add_feed(Bybit(channels=[CANDLES], symbols=pairs, callbacks={CANDLES: CandlesPostgres(**postgres_cfg)}))
    f.run()


if __name__ == '__main__':
    main()
