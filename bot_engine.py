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

# --- MODELS ---
class StrategyConfig(BaseModel):
    upper_price: Optional[float] = 0
    lower_price: Optional[float] = 0
    risk_percentage: Optional[float] = 0  # 0.1, 0.2, 0.3 (for Auto)
    trailing_up: Optional[bool] = False   # For Manual
    trailing_down: Optional[bool] = False # For Manual
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
    timeframe: str = '1h'
    startDate: str  
    endDate: str    
    capital: float
    upperPrice: Optional[float] = 0 
    lowerPrice: Optional[float] = 0
    riskPercentage: Optional[float] = 0 # 0.1 = High, 0.2 = Med, 0.3 = Low
    trailingUp: Optional[bool] = False
    trailingDown: Optional[bool] = False
    gridSize: int

# --- BACKTESTER ---
class Backtester:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.exchange_id = config.exchange.lower()
        self.pair = config.pair
        self.timeframe = config.timeframe
        self.start_date = config.startDate
        self.end_date = config.endDate 
        self.initial_balance = config.capital
        self.upper_price = config.upperPrice
        self.lower_price = config.lowerPrice
        self.risk_pct = config.riskPercentage
        self.grids = config.gridSize
        self.trailing_up = config.trailingUp
        self.trailing_down = config.trailingDown

    async def fetch_historical_data(self):
        exchange_class = getattr(ccxt, self.exchange_id)
        exchange = exchange_class({'enableRateLimit': True})
        try:
            start_ts = exchange.parse8601(f"{self.start_date}T00:00:00Z")
            end_ts = exchange.parse8601(f"{self.end_date}T23:59:59Z")
            all_ohlcv = []
            current_since = start_ts
            while current_since < end_ts:
                ohlcv = await exchange.fetch_ohlcv(self.pair, self.timeframe, current_since, limit=1000)
                if not ohlcv: break
                batch = [c for c in ohlcv if c[0] <= end_ts]
                all_ohlcv += batch
                last_time = ohlcv[-1][0]
                if last_time == current_since: break 
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

        # --- 1. AUTO RANGE CALCULATION (If Auto Mode) ---
        first_close = ohlcv[0][4]
        if self.risk_pct > 0:
            logger.info(f"Auto-calculating range based on risk {self.risk_pct*100}% at start price: ${first_close}")
            self.upper_price = first_close * (1 + self.risk_pct)
            self.lower_price = first_close * (1 - self.risk_pct)
        elif self.upper_price == 0 or self.lower_price == 0:
            # Fallback if no risk pct and no price provided
            self.upper_price = first_close * 1.10
            self.lower_price = first_close * 0.90

        # Initial Grid Setup
        step = (self.upper_price - self.lower_price) / self.grids
        grid_levels = [self.lower_price + (i * step) for i in range(self.grids + 1)]
        
        balance_usdt = self.initial_balance
        balance_asset = 0
        cumulative_grid_profit = 0
        trade_history = []
        chart_data = []
        
        active_grids = {i: False for i in range(len(grid_levels))}
        investment_per_grid = self.initial_balance / self.grids
        skip_step = max(1, len(ohlcv) // 1000) 

        for index, candle in enumerate(ohlcv):
            timestamp, open_, high, low, close, volume = candle
            date_str = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M')
            action = None
            
            # --- 2. TRAILING LOGIC (Shift Grids) ---
            grid_shift = 0
            if self.trailing_up and close > self.upper_price:
                # Move everything UP by one step size roughly
                diff = close - self.upper_price
                grid_shift = diff
            elif self.trailing_down and close < self.lower_price:
                # Move everything DOWN
                diff = close - self.lower_price
                grid_shift = diff
            
            if grid_shift != 0:
                self.upper_price += grid_shift
                self.lower_price += grid_shift
                # Re-calculate levels
                grid_levels = [level + grid_shift for level in grid_levels]
                # Note: In a real bot, we'd cancel orders and replace them. 
                # Here we just shift the logic reference.

            # --- 3. TRADE LOGIC ---
            for i, level in enumerate(grid_levels):
                # BUY
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

                # SELL
                elif high > (level + step) and active_grids[i]:
                    if balance_asset > 0:
                        amount = investment_per_grid / level 
                        revenue = amount * (level + step)
                        profit = revenue - investment_per_grid
                        balance_usdt += revenue
                        balance_asset -= amount
                        cumulative_grid_profit += profit
                        active_grids[i] = False
                        action = 'sell'
                        trade_history.append({
                            "time": date_str, "type": "Sell", "price": level + step, "amount": amount, "profit": profit
                        })

            if index % skip_step == 0 or action:
                current_asset_value = balance_asset * close
                total_equity = balance_usdt + current_asset_value
                chart_data.append({
                    "date": date_str, 
                    "price": close, 
                    "totalValue": total_equity, 
                    "assetValue": current_asset_value, 
                    "gridProfit": cumulative_grid_profit, 
                    "action": action
                })

        final_price = ohlcv[-1][4]
        current_asset_value = balance_asset * final_price
        total_equity = balance_usdt + current_asset_value
        total_profit = total_equity - self.initial_balance
        roi = (total_profit / self.initial_balance) * 100
        
        return {
            "status": "success",
            "stats": {
                "totalProfit": round(total_profit, 2),
                "gridProfit": round(cumulative_grid_profit, 2),
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
        # Handle Auto Calculation if Risk Percentage is provided
        risk = self.config.strategy.risk_percentage
        if risk and risk > 0:
            ticker = await self.exchange.fetch_ticker(self.pair)
            current_price = ticker['last']
            logger.info(f"Bot {self.bot_id}: Auto-setting {risk*100}% range. Price: ${current_price}")
            self.config.strategy.upper_price = current_price * (1 + risk)
            self.config.strategy.lower_price = current_price * (1 - risk)
        
        elif self.config.strategy.upper_price == 0:
             # Default fallback
            ticker = await self.exchange.fetch_ticker(self.pair)
            current_price = ticker['last']
            self.config.strategy.upper_price = current_price * 1.10
            self.config.strategy.lower_price = current_price * 0.90

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
            self.last_grid_index = min(range(len(self.grid_levels)), key=lambda i: abs(self.grid_levels[i] - ticker['last']))

            while self.running:
                try:
                    ticker = await self.exchange.fetch_ticker(self.pair)
                    price = ticker['last']
                    
                    # --- TRAILING CHECK (Live) ---
                    # If Trailing Up enabled and price breaks upper bound
                    if self.config.strategy.trailing_up and price > self.grid_levels[-1]:
                        shift = price - self.grid_levels[-1]
                        self.grid_levels = [l + shift for l in self.grid_levels]
                        logger.info(f"Bot {self.bot_id}: Trailing UP. New Range: {self.grid_levels[0]:.2f} - {self.grid_levels[-1]:.2f}")

                    # If Trailing Down enabled and price breaks lower bound
                    if self.config.strategy.trailing_down and price < self.grid_levels[0]:
                        shift = price - self.grid_levels[0]
                        self.grid_levels = [l + shift for l in self.grid_levels]
                        logger.info(f"Bot {self.bot_id}: Trailing DOWN. New Range: {self.grid_levels[0]:.2f} - {self.grid_levels[-1]:.2f}")

                    # Trading Logic (Simplified)
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
            # Execute real order here...
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
