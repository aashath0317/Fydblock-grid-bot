import asyncio
import ccxt.async_support as ccxt  
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, List
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FydEngine")

app = FastAPI(title="FydBlock Trading Engine")
active_bots: Dict[int, dict] = {}

# --- DATA MODELS ---
class StrategyConfig(BaseModel):
    upper_price: Optional[float] = 0
    lower_price: Optional[float] = 0
    risk_percentage: Optional[float] = 0  # 0.1 (High), 0.2 (Med), 0.3 (Low)
    trailing_up: Optional[bool] = False   # For Manual Mode
    trailing_down: Optional[bool] = False # For Manual Mode
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

# ==========================================
# 1. BACKTEST ENGINE (With Auto-Rebalance)
# ==========================================
class Backtester:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.exchange_id = config.exchange.lower()
        self.pair = config.pair
        self.timeframe = config.timeframe
        self.start_date = config.startDate
        self.end_date = config.endDate 
        self.initial_balance = config.capital
        
        # Current Active Settings
        self.curr_upper = config.upperPrice
        self.curr_lower = config.lowerPrice
        self.risk_pct = config.riskPercentage
        self.grids = config.gridSize
        self.trailing_up = config.trailingUp
        self.trailing_down = config.trailingDown

    async def fetch_historical_data(self):
        # Initialize exchange for data fetching only
        exchange_class = getattr(ccxt, self.exchange_id)
        exchange = exchange_class({'enableRateLimit': True})
        
        try:
            start_ts = exchange.parse8601(f"{self.start_date}T00:00:00Z")
            end_ts = exchange.parse8601(f"{self.end_date}T23:59:59Z")
            
            all_ohlcv = []
            current_since = start_ts
            
            while current_since < end_ts:
                try:
                    ohlcv = await exchange.fetch_ohlcv(self.pair, self.timeframe, current_since, limit=1000)
                    if not ohlcv: break
                    
                    batch = [c for c in ohlcv if c[0] <= end_ts]
                    all_ohlcv += batch
                    
                    last_time = ohlcv[-1][0]
                    if last_time == current_since: break 
                    current_since = last_time + 1 
                    
                    if last_time >= end_ts: break
                    await asyncio.sleep(exchange.rateLimit / 1000)
                except Exception as e:
                    logger.error(f"Error fetching candles: {e}")
                    break
                    
            return all_ohlcv
        finally:
            await exchange.close()

    def calculate_grid_levels(self):
        # Standard Grid Calculation
        step = (self.curr_upper - self.curr_lower) / self.grids
        return [self.curr_lower + (i * step) for i in range(self.grids + 1)], step

    async def run(self):
        logger.info(f"Running backtest for {self.pair}...")
        ohlcv = await self.fetch_historical_data()
        
        if not ohlcv:
            return {"status": "error", "message": "No historical data found"}

        # --- INITIAL SETUP ---
        first_close = ohlcv[0][4]
        
        # Set initial range if Auto Mode
        if self.risk_pct and self.risk_pct > 0:
            self.curr_upper = first_close * (1 + self.risk_pct)
            self.curr_lower = first_close * (1 - self.risk_pct)
        elif self.curr_upper == 0:
            self.curr_upper = first_close * 1.10
            self.curr_lower = first_close * 0.90

        grid_levels, step = self.calculate_grid_levels()
        
        balance_usdt = self.initial_balance
        balance_asset = 0
        cumulative_grid_profit = 0
        trade_history = []
        chart_data = []
        
        # Initial Buy-In (50% Asset / 50% Cash)
        # This is crucial: we assume we start neutrally
        initial_buy = (self.initial_balance * 0.5) / first_close
        balance_asset += initial_buy
        balance_usdt -= (self.initial_balance * 0.5)
        
        # Active Grids: 
        # Price is in the middle. 
        # Levels > Price are "Filled" (we have asset to sell). 
        # Levels < Price are "Empty" (we have cash to buy).
        active_grids = {
            i: (level < first_close) # False = Waiting to Buy (Low), True = Waiting to Sell (High)
            for i, level in enumerate(grid_levels)
        }
        
        # Invert logic for grid standard:
        # If level < current price, we should have ALREADY BOUGHT. So we hold asset. Next move is SELL. (active=True)
        # If level > current price, we have NOT BOUGHT. We hold cash. Next move is BUY. (active=False)
        active_grids = {}
        for i, level in enumerate(grid_levels):
            if level < first_close:
                active_grids[i] = True  # We hold this bag, ready to sell
            else:
                active_grids[i] = False # We don't hold, ready to buy

        investment_per_grid = self.initial_balance / self.grids
        skip_step = max(1, len(ohlcv) // 500) 

        # --- SIMULATION LOOP ---
        for index, candle in enumerate(ohlcv):
            timestamp, open_, high, low, close, volume = candle
            date_str = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M')
            action = None
            
            # ---------------------------
            # 1. AUTO-RESET & TRAILING
            # ---------------------------
            reset_needed = False
            
            # AUTO MODE: Re-Center Grid on Breakout
            if self.risk_pct and self.risk_pct > 0:
                if close > self.curr_upper:
                    # RESET UP
                    self.curr_upper = close * (1 + self.risk_pct)
                    self.curr_lower = close * (1 - self.risk_pct)
                    reset_needed = True
                    action = 'reset_up'
                elif close < self.curr_lower:
                    # RESET DOWN
                    self.curr_upper = close * (1 + self.risk_pct)
                    self.curr_lower = close * (1 - self.risk_pct)
                    reset_needed = True
                    action = 'reset_down'

            # MANUAL MODE: Simple Trailing (No Rebalance)
            elif not reset_needed:
                grid_shift = 0
                if self.trailing_up and close > self.curr_upper:
                    grid_shift = close - self.curr_upper
                elif self.trailing_down and close < self.curr_lower:
                    grid_shift = close - self.curr_lower
                
                if grid_shift != 0:
                    self.curr_upper += grid_shift
                    self.curr_lower += grid_shift
                    # Just shift lines, don't rebalance
                    grid_levels = [level + grid_shift for level in grid_levels]

            # APPLY RESET WITH REBALANCE
            if reset_needed:
                # 1. New Grids
                grid_levels, step = self.calculate_grid_levels()
                
                # 2. Total Equity Value
                total_equity = balance_usdt + (balance_asset * close)
                
                # 3. Rebalance to 50/50 Neutral Strategy
                # This ensures we always have ammo to follow the trend up or down
                target_asset_val = total_equity * 0.5
                current_asset_val = balance_asset * close
                
                if current_asset_val < target_asset_val:
                    # BUY MORE
                    diff_usdt = target_asset_val - current_asset_val
                    amount = diff_usdt / close
                    if balance_usdt >= diff_usdt:
                        balance_usdt -= diff_usdt
                        balance_asset += amount
                
                elif current_asset_val > target_asset_val:
                    # SELL EXCESS
                    diff_asset_val = current_asset_val - target_asset_val
                    amount = diff_asset_val / close
                    balance_asset -= amount
                    balance_usdt += (amount * close)

                # 4. Reset Grid States
                # Levels below price = Filled (Ready to Sell)
                # Levels above price = Empty (Ready to Buy)
                active_grids = {}
                for i, level in enumerate(grid_levels):
                    if level < close:
                        active_grids[i] = True 
                    else:
                        active_grids[i] = False
                
                # Update unit size
                investment_per_grid = total_equity / self.grids

            # ---------------------------
            # 2. GRID TRADING EXECUTION
            # ---------------------------
            for i, level in enumerate(grid_levels):
                # BUY: Price hits lower line AND we don't have a bag yet
                if low < level and not active_grids[i]:
                    if balance_usdt >= investment_per_grid:
                        amount = investment_per_grid / level
                        balance_usdt -= investment_per_grid
                        balance_asset += amount
                        active_grids[i] = True
                        if not action: action = 'buy'
                        trade_history.append({
                            "time": date_str, "type": "Buy", "price": level, "amount": amount, "profit": 0
                        })

                # SELL: Price hits upper line AND we hold a bag
                elif high > (level + step) and active_grids[i]:
                    if balance_asset > 0:
                        amount = investment_per_grid / level # Sell 1 grid unit
                        # Safety check if we have enough asset (floating point errors)
                        sell_amt = min(amount, balance_asset)
                        
                        revenue = sell_amt * (level + step)
                        profit = revenue - (sell_amt * level) # Gross profit per grid
                        
                        balance_usdt += revenue
                        balance_asset -= sell_amt
                        cumulative_grid_profit += profit
                        active_grids[i] = False
                        if not action: action = 'sell'
                        trade_history.append({
                            "time": date_str, "type": "Sell", "price": level + step, "amount": sell_amt, "profit": profit
                        })

            # Record Data for Chart
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

        # --- FINAL RESULTS ---
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

# ==========================================
# 2. LIVE TRADING BOT (Async Execution)
# ==========================================
class GridBot:
    def __init__(self, config: BotRequest):
        self.bot_id = config.bot_id
        self.pair = config.pair
        self.config = config
        self.exchange = None
        self.running = False
        self.grid_levels = []
        self.last_grid_index = -1 
        self.market_precision = None 

    async def initialize_exchange(self):
        exchange_class = getattr(ccxt, self.config.exchange.lower())
        self.exchange = exchange_class({
            'apiKey': self.config.api_key,
            'secret': self.config.api_secret,
            'password': self.config.passphrase,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'} 
        })
        
        # Load markets for precision formatting
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
            # Find nearest grid to start
            self.last_grid_index = min(range(len(self.grid_levels)), key=lambda i: abs(self.grid_levels[i] - ticker['last']))

            while self.running:
                try:
                    ticker = await self.exchange.fetch_ticker(self.pair)
                    price = ticker['last']
                    
                    # --- AUTO RESET / TRAILING LOGIC ---
                    risk = self.config.strategy.risk_percentage
                    reset_needed = False
                    
                    # Auto Mode Logic
                    if risk and risk > 0:
                        if price > self.config.strategy.upper_price:
                            logger.info(f"Bot {self.bot_id}: Price broke UPPER. Resetting grid...")
                            self.config.strategy.upper_price = price * (1 + risk)
                            self.config.strategy.lower_price = price * (1 - risk)
                            reset_needed = True
                        elif price < self.config.strategy.lower_price:
                            logger.info(f"Bot {self.bot_id}: Price broke LOWER. Resetting grid...")
                            self.config.strategy.upper_price = price * (1 + risk)
                            self.config.strategy.lower_price = price * (1 - risk)
                            reset_needed = True
                            
                        if reset_needed:
                            # In real bot: await self.exchange.cancel_all_orders(self.pair)
                            await self.calculate_grids() 
                            self.last_grid_index = min(range(len(self.grid_levels)), key=lambda i: abs(self.grid_levels[i] - price))
                            continue # Restart loop with new grids

                    # Manual Trailing Logic
                    elif self.config.strategy.trailing_up and price > self.grid_levels[-1]:
                        shift = price - self.grid_levels[-1]
                        self.grid_levels = [l + shift for l in self.grid_levels]
                    
                    elif self.config.strategy.trailing_down and price < self.grid_levels[0]:
                        shift = price - self.grid_levels[0]
                        self.grid_levels = [l + shift for l in self.grid_levels]

                    # --- GRID EXECUTION LOGIC ---
                    
                    # 1. Price UP -> Sell
                    if self.last_grid_index < len(self.grid_levels) - 1:
                        next_level = self.grid_levels[self.last_grid_index + 1]
                        if price >= next_level:
                            await self.execute_trade('sell', price)
                            self.last_grid_index += 1
                            continue 

                    # 2. Price DOWN -> Buy
                    if self.last_grid_index > 0:
                        prev_level = self.grid_levels[self.last_grid_index - 1]
                        if price <= prev_level:
                            await self.execute_trade('buy', price)
                            self.last_grid_index -= 1
                            continue
                            
                    await asyncio.sleep(5) # Poll every 5s
                    
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
            investment_per_grid = self.config.strategy.investment / self.config.strategy.grids
            amount = investment_per_grid / price

            # 2. Precision Formatting (Crucial for live execution)
            if self.market_precision:
                symbol = self.pair
                formatted_amount = self.exchange.amount_to_precision(symbol, amount)
                formatted_price = self.exchange.price_to_precision(symbol, price)
            else:
                formatted_amount = amount
                formatted_price = price

            # 3. Execute Order
            order = await self.exchange.create_order(
                symbol=self.pair,
                type='limit', # Or 'market'
                side=side,
                amount=formatted_amount,
                price=formatted_price
            )
            
            logger.info(f"✅ Bot {self.bot_id}: {side.upper()} order {order['id']} placed at {formatted_price}")

        except Exception as e:
            logger.error(f"❌ Bot {self.bot_id} Trade Failed: {e}")

# ==========================================
# 3. API ENDPOINTS
# ==========================================

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
        await asyncio.wait_for(bot_entry["task"], timeout=5.0)
    except: 
        pass
        
    del active_bots[bot_id]
    return {"message": "Bot stopped successfully"}
