import asyncio
import ccxt.async_support as ccxt  
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, List
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Configure Logging
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
    riskPercentage: Optional[float] = 0 
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
        # Initialize exchange in sandbox/test mode just for data fetching
        exchange_class = getattr(ccxt, self.exchange_id)
        exchange = exchange_class({'enableRateLimit': True})
        
        try:
            # Convert ISO dates to timestamps
            start_ts = exchange.parse8601(f"{self.start_date}T00:00:00Z")
            end_ts = exchange.parse8601(f"{self.end_date}T23:59:59Z")
            
            all_ohlcv = []
            current_since = start_ts
            
            while current_since < end_ts:
                try:
                    ohlcv = await exchange.fetch_ohlcv(self.pair, self.timeframe, current_since, limit=1000)
                    if not ohlcv: break
                    
                    # Filter candles that might exceed end_ts
                    batch = [c for c in ohlcv if c[0] <= end_ts]
                    all_ohlcv += batch
                    
                    last_time = ohlcv[-1][0]
                    # Prevent infinite loops if exchange returns same data
                    if last_time == current_since: break 
                    current_since = last_time + 1 
                    
                    if last_time >= end_ts: break
                    
                    # Respect rate limits
                    await asyncio.sleep(exchange.rateLimit / 1000)
                    
                except Exception as e:
                    logger.error(f"Error fetching candles: {e}")
                    break
                    
            return all_ohlcv
        finally:
            await exchange.close()

    async def run(self):
        logger.info(f"Running backtest for {self.pair}...")
        ohlcv = await self.fetch_historical_data()
        
        if not ohlcv:
            return {"status": "error", "message": "No historical data found or exchange error"}

        # --- 1. AUTO RANGE CALCULATION (If Auto Mode) ---
        first_close = ohlcv[0][4]
        if self.risk_pct and self.risk_pct > 0:
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
        
        # Optimize chart data size (don't return every single minute point for large ranges)
        skip_step = max(1, len(ohlcv) // 500) 

        for index, candle in enumerate(ohlcv):
            timestamp, open_, high, low, close, volume = candle
            date_str = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M')
            action = None
            
            # --- 2. TRAILING LOGIC (Shift Grids) ---
            grid_shift = 0
            if self.trailing_up and close > self.upper_price:
                diff = close - self.upper_price
                grid_shift = diff
            elif self.trailing_down and close < self.lower_price:
                diff = close - self.lower_price
                grid_shift = diff
            
            if grid_shift != 0:
                self.upper_price += grid_shift
                self.lower_price += grid_shift
                # Re-calculate levels
                grid_levels = [level + grid_shift for level in grid_levels]

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

            # Record Chart Data
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
            "history": trade_history, # Return full history for the table
            "chartData": chart_data 
        }

# --- LIVE BOT (REAL TRADING) ---
class GridBot:
    def __init__(self, config: BotRequest):
        self.bot_id = config.bot_id
        self.pair = config.pair
        self.config = config
        self.exchange = None
        self.running = False
        self.grid_levels = []
        self.last_grid_index = -1 
        self.market_precision = None # To store precision data

    async def initialize_exchange(self):
        exchange_class = getattr(ccxt, self.config.exchange.lower())
        self.exchange = exchange_class({
            'apiKey': self.config.api_key,
            'secret': self.config.api_secret,
            'password': self.config.passphrase,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'} 
        })
        
        # Load markets to get precision and limits
        markets = await self.exchange.load_markets()
        if self.pair in markets:
            self.market_precision = markets[self.pair]
        else:
            logger.error(f"Pair {self.pair} not found on {self.config.exchange}")

    async def calculate_grids(self):
        # Handle Auto Calculation
        risk = self.config.strategy.risk_percentage
        if risk and risk > 0:
            ticker = await self.exchange.fetch_ticker(self.pair)
            current_price = ticker['last']
            self.config.strategy.upper_price = current_price * (1 + risk)
            self.config.strategy.lower_price = current_price * (1 - risk)
        
        elif self.config.strategy.upper_price == 0:
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
        logger.info(f"🚀 Live Bot {self.bot_id} Started on {self.config.exchange}")
        try:
            await self.initialize_exchange()
            await self.calculate_grids()
            
            ticker = await self.exchange.fetch_ticker(self.pair)
            # Find the closest grid level to current price to start
            self.last_grid_index = min(range(len(self.grid_levels)), key=lambda i: abs(self.grid_levels[i] - ticker['last']))

            while self.running:
                try:
                    ticker = await self.exchange.fetch_ticker(self.pair)
                    price = ticker['last']
                    
                    # --- TRAILING LOGIC ---
                    if self.config.strategy.trailing_up and price > self.grid_levels[-1]:
                        shift = price - self.grid_levels[-1]
                        self.grid_levels = [l + shift for l in self.grid_levels]
                        logger.info(f"Bot {self.bot_id}: Trailing UP. New Range: {self.grid_levels[0]:.2f} - {self.grid_levels[-1]:.2f}")

                    if self.config.strategy.trailing_down and price < self.grid_levels[0]:
                        shift = price - self.grid_levels[0]
                        self.grid_levels = [l + shift for l in self.grid_levels]
                        logger.info(f"Bot {self.bot_id}: Trailing DOWN. New Range: {self.grid_levels[0]:.2f} - {self.grid_levels[-1]:.2f}")

                    # --- TRADING LOGIC ---
                    # Check Sell (Price went UP to next grid)
                    if self.last_grid_index < len(self.grid_levels) - 1:
                        next_level = self.grid_levels[self.last_grid_index + 1]
                        if price >= next_level:
                            await self.execute_trade('sell', price)
                            self.last_grid_index += 1
                            continue 

                    # Check Buy (Price went DOWN to prev grid)
                    if self.last_grid_index > 0:
                        prev_level = self.grid_levels[self.last_grid_index - 1]
                        if price <= prev_level:
                            await self.execute_trade('buy', price)
                            self.last_grid_index -= 1
                            continue
                            
                    await asyncio.sleep(5) # Poll interval
                except Exception as e:
                    logger.error(f"Bot {self.bot_id} Loop Error: {e}")
                    await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Bot {self.bot_id} Critical Failure: {e}")
        finally:
            if self.exchange: await self.exchange.close()

    async def execute_trade(self, side, price):
        try:
            # 1. Calculate Amount
            # Investment per grid / price = amount of coin
            investment_per_grid = self.config.strategy.investment / self.config.strategy.grids
            amount = investment_per_grid / price

            # 2. Format Precision
            if self.market_precision:
                symbol = self.pair
                # CCXT specific method to format amount/price to exchange requirements
                formatted_amount = self.exchange.amount_to_precision(symbol, amount)
                formatted_price = self.exchange.price_to_precision(symbol, price)
            else:
                formatted_amount = amount
                formatted_price = price

            # 3. Create Order
            # Using 'limit' orders for Grid strategy usually, or 'market' for instant fill
            order = await self.exchange.create_order(
                symbol=self.pair,
                type='limit', # Or 'market' if you want instant execution
                side=side,
                amount=formatted_amount,
                price=formatted_price
            )
            
            logger.info(f"✅ Bot {self.bot_id}: {side.upper()} order placed. ID: {order['id']} at {formatted_price}")
            
            # Optional: Notify backend via webhook about trade
            # await self.notify_backend(order)

        except Exception as e:
            logger.error(f"❌ Bot {self.bot_id} Trade Execution Failed: {e}")

@app.get("/")
def health_check(): 
    return {"status": "online", "active_bots": len(active_bots)}

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
    if config.bot_id in active_bots: 
        raise HTTPException(status_code=400, detail="Bot already running")
    
    bot = GridBot(config)
    # Run the bot loop in background
    task = asyncio.create_task(bot.run())
    active_bots[config.bot_id] = { "instance": bot, "task": task, "config": config.dict() }
    
    return {"message": "Bot started successfully", "bot_id": config.bot_id}

@app.post("/stop/{bot_id}")
async def stop_bot_endpoint(bot_id: int):
    if bot_id not in active_bots: 
        raise HTTPException(status_code=404, detail="Bot not found")
    
    bot_entry = active_bots[bot_id]
    bot_entry["instance"].running = False
    
    try: 
        # Wait briefly for graceful shutdown
        await asyncio.wait_for(bot_entry["task"], timeout=5.0)
    except (asyncio.CancelledError, asyncio.TimeoutError): 
        pass
        
    del active_bots[bot_id]
    return {"message": "Bot stopped successfully"}
