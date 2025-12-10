import asyncio
import ccxt.async_support as ccxt  
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, List
import logging
from datetime import datetime
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FydEngine")

app = FastAPI(title="FydBlock Trading Engine")
active_bots: Dict[int, dict] = {}

class StrategyConfig(BaseModel):
    upper_price: float
    lower_price: float
    grids: int
    investment: float

class BotRequest(BaseModel):
    bot_id: int
    user_id: int
    exchange: str      
    pair: str          
    api_key: str
    api_secret: str
    passphrase: Optional[str] = None
    strategy: StrategyConfig

class BacktestConfig(BaseModel):
    exchange: str = 'binance'
    pair: str = 'BTC/USDT'
    startDate: str  
    endDate: str    # ✅ Added End Date
    capital: float
    upperPrice: float
    lowerPrice: float
    gridSize: int

# --- BACKTESTER ---
class Backtester:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.exchange_id = config.exchange.lower()
        self.pair = config.pair
        self.timeframe = '1h' 
        self.start_date = config.startDate
        self.end_date = config.endDate # Use this
        self.initial_balance = config.capital
        self.upper_price = config.upperPrice
        self.lower_price = config.lowerPrice
        self.grids = config.gridSize

    async def fetch_historical_data(self):
        exchange_class = getattr(ccxt, self.exchange_id)
        exchange = exchange_class({'enableRateLimit': True})
        
        try:
            start_ts = exchange.parse8601(f"{self.start_date}T00:00:00Z")
            end_ts = exchange.parse8601(f"{self.end_date}T23:59:59Z") # End of Day

            all_ohlcv = []
            current_since = start_ts
            
            # Fetch loop (Pagination)
            while current_since < end_ts:
                ohlcv = await exchange.fetch_ohlcv(self.pair, self.timeframe, current_since, limit=1000)
                if not ohlcv: break
                
                # Filter results
                batch = [c for c in ohlcv if c[0] <= end_ts]
                all_ohlcv += batch
                
                last_time = ohlcv[-1][0]
                if last_time == current_since: break # Avoid infinite loop
                current_since = last_time + 1
                
                if last_time >= end_ts: break

            return all_ohlcv
        finally:
            await exchange.close()

    async def run(self):
        logger.info(f"Running backtest for {self.pair}...")
        ohlcv = await self.fetch_historical_data()
        
        if not ohlcv:
            return {"status": "error", "message": "No historical data found"}

        step = (self.upper_price - self.lower_price) / self.grids
        grid_levels = [self.lower_price + (i * step) for i in range(self.grids + 1)]
        
        balance_usdt = self.initial_balance
        balance_asset = 0
        trade_history = []
        chart_data = []
        
        active_grids = {i: False for i in range(len(grid_levels))}
        investment_per_grid = self.initial_balance / self.grids

        for candle in ohlcv:
            timestamp, open_, high, low, close, volume = candle
            date_str = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M')
            action = None
            
            for i, level in enumerate(grid_levels):
                if low < level and not active_grids[i]:
                    if balance_usdt >= investment_per_grid:
                        amount = investment_per_grid / level
                        balance_usdt -= investment_per_grid
                        balance_asset += amount
                        active_grids[i] = True
                        action = 'buy'
                        trade_history.append({
                            "time": date_str, "type": "Buy", "price": level, "amount": amount, "profit": 0
                        })

                elif high > (level + step) and active_grids[i]:
                    if balance_asset > 0:
                        amount = investment_per_grid / level 
                        revenue = amount * (level + step)
                        profit = revenue - investment_per_grid
                        balance_usdt += revenue
                        balance_asset -= amount
                        active_grids[i] = False
                        action = 'sell'
                        trade_history.append({
                            "time": date_str, "type": "Sell", "price": level + step, "amount": amount, "profit": profit
                        })

            current_value = balance_usdt + (balance_asset * close)
            chart_data.append({
                "date": date_str, "price": close, "value": current_value, "action": action
            })

        total_profit = current_value - self.initial_balance
        roi = (total_profit / self.initial_balance) * 100
        
        return {
            "status": "success",
            "stats": {
                "totalProfit": round(total_profit, 2),
                "roi": round(roi, 2),
                "totalTrades": len(trade_history)
            },
            "history": trade_history,
            "chartData": chart_data 
        }

# --- LIVE BOT ---
class GridBot:
    def __init__(self, config: BotRequest):
        self.bot_id = config.bot_id
        self.pair = config.pair
        self.config = config
        self.exchange = None
        self.running = False
        self.grid_levels = []
        self.last_grid_index = -1 

    async def initialize_exchange(self):
        exchange_class = getattr(ccxt, self.config.exchange.lower())
        self.exchange = exchange_class({
            'apiKey': self.config.api_key,
            'secret': self.config.api_secret,
            'password': self.config.passphrase,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'} 
        })
        await self.exchange.load_markets()

    async def calculate_grids(self):
        upper = self.config.strategy.upper_price
        lower = self.config.strategy.lower_price
        count = self.config.strategy.grids
        step = (upper - lower) / count
        self.grid_levels = [lower + (i * step) for i in range(count + 1)]

    async def run(self):
        self.running = True
        logger.info(f"🚀 Live Bot {self.bot_id} Started")
        try:
            await self.initialize_exchange()
            await self.calculate_grids()
            ticker = await self.exchange.fetch_ticker(self.pair)
            current_price = ticker['last']
            self.last_grid_index = min(range(len(self.grid_levels)), key=lambda i: abs(self.grid_levels[i] - current_price))

            while self.running:
                try:
                    ticker = await self.exchange.fetch_ticker(self.pair)
                    price = ticker['last']
                    if self.last_grid_index < len(self.grid_levels) - 1:
                        next_level = self.grid_levels[self.last_grid_index + 1]
                        if price >= next_level:
                            await self.execute_trade('sell', price)
                            self.last_grid_index += 1
                            continue 
                    if self.last_grid_index > 0:
                        prev_level = self.grid_levels[self.last_grid_index - 1]
                        if price <= prev_level:
                            await self.execute_trade('buy', price)
                            self.last_grid_index -= 1
                            continue
                    await asyncio.sleep(2) 
                except Exception as e:
                    logger.error(f"Bot {self.bot_id} Loop Error: {e}")
                    await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Bot {self.bot_id} Critical Failure: {e}")
        finally:
            if self.exchange: await self.exchange.close()

    async def execute_trade(self, side, price):
        try:
            amount_usdt = self.config.strategy.investment / self.config.strategy.grids
            amount = amount_usdt / price 
            logger.info(f"✅ Bot {self.bot_id}: {side.upper()} executed at ${price}")
        except Exception as e:
            logger.error(f"Bot {self.bot_id} Trade Failed: {e}")

@app.get("/")
def health_check(): return {"status": "online", "active_bots": len(active_bots)}

@app.post("/backtest")
async def run_backtest_endpoint(config: BacktestConfig):
    try:
        backtester = Backtester(config)
        return await backtester.run()
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start")
async def start_bot_endpoint(config: BotRequest, background_tasks: BackgroundTasks):
    if config.bot_id in active_bots: raise HTTPException(status_code=400, detail="Bot already running")
    bot = GridBot(config)
    task = asyncio.create_task(bot.run())
    active_bots[config.bot_id] = { "instance": bot, "task": task, "config": config.dict() }
    return {"message": "Bot started successfully", "bot_id": config.bot_id}

@app.post("/stop/{bot_id}")
async def stop_bot_endpoint(bot_id: int):
    if bot_id not in active_bots: raise HTTPException(status_code=404, detail="Bot not found")
    bot_entry = active_bots[bot_id]
    bot_entry["instance"].running = False
    try: await bot_entry["task"]
    except asyncio.CancelledError: pass
    del active_bots[bot_id]
    return {"message": "Bot stopped successfully"}
