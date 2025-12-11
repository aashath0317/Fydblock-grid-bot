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
    risk_percentage: Optional[float] = 0
    trailing_up: Optional[bool] = False
    trailing_down: Optional[bool] = False
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

# --- UTILITIES ---
def safe_div(a, b, default=0.0):
    try:
        return a / b
    except Exception:
        return default

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

        # failsafe
        if config.riskPercentage == 0 and config.upperPrice == 0 and config.lowerPrice == 0:
            logger.warning("⚠️ FAILSAFE: Config 0s. Forcing AUTO (10% Risk).")
            self.risk_pct = 0.10
        else:
            self.risk_pct = config.riskPercentage or 0.0

        self.curr_upper = config.upperPrice or 0.0
        self.curr_lower = config.lowerPrice or 0.0
        self.grids = max(1, config.gridSize)
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
                try:
                    ohlcv = await exchange.fetch_ohlcv(self.pair, self.timeframe, current_since, limit=1000)
                    if not ohlcv:
                        break
                    batch = [c for c in ohlcv if c[0] <= end_ts]
                    all_ohlcv += batch
                    last_time = ohlcv[-1][0]
                    if last_time == current_since:
                        break
                    current_since = last_time + 1
                    if last_time >= end_ts:
                        break
                    await asyncio.sleep(exchange.rateLimit / 1000)
                except Exception as e:
                    logger.error(f"Error fetching candles: {e}")
                    break
            return all_ohlcv
        finally:
            await exchange.close()

    def calculate_grid_levels(self):
        # safety
        if self.grids <= 0:
            self.grids = 1
        # avoid zero-width grid
        if self.curr_upper <= self.curr_lower:
            # small padding
            mid = max(self.curr_upper, self.curr_lower, 1.0)
            self.curr_upper = mid * 1.05
            self.curr_lower = mid * 0.95
        step = (self.curr_upper - self.curr_lower) / self.grids
        levels = [self.curr_lower + (i * step) for i in range(self.grids + 1)]
        return levels, step

    async def run(self):
        logger.info(f"Running backtest for {self.pair}...")
        ohlcv = await self.fetch_historical_data()
        if not ohlcv:
            return {"status": "error", "message": "No historical data found"}

        first_close = ohlcv[0][4]

        # Initialize ranges
        if self.risk_pct and self.risk_pct > 0:
            self.curr_upper = first_close * (1 + self.risk_pct)
            self.curr_lower = first_close * (1 - self.risk_pct)
        elif self.curr_upper == 0:
            self.curr_upper = first_close * 1.10
            self.curr_lower = first_close * 0.90

        logger.info(f"🏁 START RANGE: {self.curr_lower:.6f} - {self.curr_upper:.6f}")

        grid_levels, step = self.calculate_grid_levels()
        balance_usdt = self.initial_balance
        balance_asset = 0.0
        cumulative_grid_profit = 0.0
        trade_history: List[dict] = []
        chart_data: List[dict] = []

        # initial buy 50%
        initial_buy = (self.initial_balance * 0.5) / first_close
        balance_asset += initial_buy
        balance_usdt -= (self.initial_balance * 0.5)

        active_grids = {}
        for i, level in enumerate(grid_levels):
            active_grids[i] = True if level < first_close else False

        investment_per_grid = self.initial_balance / self.grids
        skip_step = max(1, len(ohlcv) // 500)

        for index, candle in enumerate(ohlcv):
            timestamp, open_, high, low, close, volume = candle
            date_str = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M')
            action = None
            reset_needed = False

            # --- STRATEGY LOGIC (Backtest) ---
            if self.risk_pct and self.risk_pct > 0:
                # UPPER HIT -> FULL RESET
                if high >= self.curr_upper:
                    self.curr_upper = high * (1 + self.risk_pct)
                    self.curr_lower = high * (1 - self.risk_pct)
                    reset_needed = True
                    action = 'reset_up'

                # LOWER HIT -> SHIFT LOWER ONLY (no full reset)
                elif low <= self.curr_lower:
                    # compute new lower as current low * (1 + risk)
                    new_lower = low * (1 + self.risk_pct)
                    logger.info(
                        f"Backtest: LOWER hit. low={low:.6f} old_lower={self.curr_lower:.6f} new_lower={new_lower:.6f}"
                    )
                    self.curr_lower = new_lower
                    action = 'shift_lower'
                    # regenerate grid_levels with same upper and new lower
                    grid_levels, step = self.calculate_grid_levels()
                    # recompute active grids relative to current close
                    for i, level in enumerate(grid_levels):
                        active_grids[i] = True if level < close else False
                    # adjust investment per grid based on current equity
                    total_equity = balance_usdt + (balance_asset * close)
                    investment_per_grid = total_equity / self.grids

            else:
                # trailing behavior when not risk-based reset
                grid_shift = 0.0
                if self.trailing_up and high >= self.curr_upper:
                    grid_shift = high - self.curr_upper
                elif self.trailing_down and low <= self.curr_lower:
                    grid_shift = low - self.curr_lower
                if grid_shift != 0:
                    self.curr_upper += grid_shift
                    self.curr_lower += grid_shift
                    grid_levels = [level + grid_shift for level in grid_levels]

            if reset_needed:
                grid_levels, step = self.calculate_grid_levels()
                total_equity = balance_usdt + (balance_asset * close)
                target_asset_val = total_equity * 0.5
                current_asset_val = balance_asset * close

                # rebalance towards 50% asset value
                if current_asset_val < target_asset_val:
                    diff_usdt = target_asset_val - current_asset_val
                    amount = diff_usdt / close
                    if balance_usdt >= diff_usdt:
                        balance_usdt -= diff_usdt
                        balance_asset += amount

                elif current_asset_val > target_asset_val:
                    # when upper reset, we allow selling excess (unless it's expand_down case)
                    diff_asset_val = current_asset_val - target_asset_val
                    amount = diff_asset_val / close
                    balance_asset -= amount
                    balance_usdt += (amount * close)

                active_grids = {}
                for i, level in enumerate(grid_levels):
                    active_grids[i] = True if level < close else False
                investment_per_grid = total_equity / self.grids

            # grid buy / sell logic
            for i, level in enumerate(grid_levels):
                # BUY lower grid level when price dips to it
                if low <= level and not active_grids.get(i, False):
                    if balance_usdt >= investment_per_grid:
                        amount = investment_per_grid / level
                        balance_usdt -= investment_per_grid
                        balance_asset += amount
                        active_grids[i] = True
                        action = action or 'buy'
                        trade_history.append({
                            "time": date_str, "type": "Buy", "price": level,
                            "amount": amount, "profit": 0
                        })

                # SELL at next grid level if price rises
                elif high >= (level + step) and active_grids.get(i, False):
                    if balance_asset > 0:
                        amount = investment_per_grid / level
                        sell_amt = min(amount, balance_asset)
                        revenue = sell_amt * (level + step)
                        profit = revenue - (sell_amt * level)
                        balance_usdt += revenue
                        balance_asset -= sell_amt
                        cumulative_grid_profit += profit
                        active_grids[i] = False
                        action = action or 'sell'
                        trade_history.append({
                            "time": date_str, "type": "Sell", "price": (level + step),
                            "amount": sell_amt, "profit": profit
                        })

            # append chart snapshot occasionally or when action occurred
            if index % skip_step == 0 or action:
                current_asset_value = balance_asset * close
                total_equity = balance_usdt + current_asset_value
                chart_data.append({
                    "date": date_str, "price": close, "totalValue": total_equity,
                    "assetValue": current_asset_value, "gridProfit": cumulative_grid_profit,
                    "action": action, "upperLimit": self.curr_upper, "lowerLimit": self.curr_lower
                })

        # final stats
        final_price = ohlcv[-1][4]
        current_asset_value = balance_asset * final_price
        total_equity = balance_usdt + current_asset_value
        total_profit = total_equity - self.initial_balance
        roi = safe_div(total_profit, self.initial_balance) * 100

        return {
            "status": "success",
            "stats": {
                "totalProfit": round(total_profit, 6),
                "gridProfit": round(cumulative_grid_profit, 6),
                "roi": round(roi, 4),
                "totalTrades": len(trade_history)
            },
            "history": trade_history,
            "chartData": chart_data
        }

# --- LIVE GRID BOT ---
class GridBot:
    def __init__(self, config: BotRequest):
        self.bot_id = config.bot_id
        self.pair = config.pair
        self.config = config
        self.exchange = None
        self.running = False
        self.grid_levels: List[float] = []
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
        markets = await self.exchange.load_markets()
        if self.pair in markets:
            self.market_precision = markets[self.pair]
        else:
            logger.error(f"Pair {self.pair} not found on {self.config.exchange}")

    async def calculate_grids(self):
        strategy = self.config.strategy
        risk = strategy.risk_percentage or 0.0
        # set initial upper/lower based on risk or default +/-10%
        if risk and risk > 0:
            ticker = await self.exchange.fetch_ticker(self.pair)
            current_price = ticker['last']
            strategy.upper_price = current_price * (1 + risk)
            strategy.lower_price = current_price * (1 - risk)
        elif strategy.upper_price == 0 or strategy.lower_price == 0:
            ticker = await self.exchange.fetch_ticker(self.pair)
            current_price = ticker['last']
            strategy.upper_price = current_price * 1.10
            strategy.lower_price = current_price * 0.90

        upper = strategy.upper_price
        lower = strategy.lower_price
        count = max(1, strategy.grids)
        # safety if invalid bounds
        if upper <= lower:
            upper = lower * 1.05
            strategy.upper_price = upper
        step = (upper - lower) / count
        self.grid_levels = [lower + (i * step) for i in range(count + 1)]

    async def run(self):
        self.running = True
        logger.info(f"🚀 Live Bot {self.bot_id} Started on {self.config.exchange}")
        try:
            await self.initialize_exchange()
            await self.calculate_grids()
            ticker = await self.exchange.fetch_ticker(self.pair)
            price = ticker['last']
            # find nearest grid index
            if self.grid_levels:
                self.last_grid_index = min(range(len(self.grid_levels)), key=lambda i: abs(self.grid_levels[i] - price))

            while self.running:
                try:
                    ticker = await self.exchange.fetch_ticker(self.pair)
                    price = ticker['last']
                    strategy = self.config.strategy
                    risk = strategy.risk_percentage or 0.0

                    # --- RISK-BASED BEHAVIOR ---
                    if risk and risk > 0:
                        # UPPER HIT -> FULL RESET
                        if price >= strategy.upper_price:
                            logger.info(f"Bot {self.bot_id}: Price hit UPPER → FULL RESET")
                            strategy.upper_price = price * (1 + risk)
                            strategy.lower_price = price * (1 - risk)
                            await self.calculate_grids()
                            # reposition
                            self.last_grid_index = min(range(len(self.grid_levels)), key=lambda i: abs(self.grid_levels[i] - price))
                            continue

                        # LOWER HIT -> SHIFT LOWER ONLY
                        if price <= strategy.lower_price:
                            logger.info(f"Bot {self.bot_id}: Price hit LOWER → SHIFTING LOWER ONLY")
                            # shift only the lower price upward based on risk percentage
                            strategy.lower_price = price * (1 + risk)
                            # recalc grid levels with same upper and new lower
                            upper = strategy.upper_price
                            lower = strategy.lower_price
                            count = max(1, strategy.grids)
                            # safety if invalid
                            if upper <= lower:
                                upper = lower * 1.05
                                strategy.upper_price = upper
                            step = (upper - lower) / count
                            self.grid_levels = [lower + (i * step) for i in range(count + 1)]
                            self.last_grid_index = min(range(len(self.grid_levels)), key=lambda i: abs(self.grid_levels[i] - price))
                            continue

                    # --- TRAILING MODES (when risk not used) ---
                    elif strategy.trailing_up and self.grid_levels and price >= self.grid_levels[-1]:
                        shift = price - self.grid_levels[-1]
                        self.grid_levels = [l + shift for l in self.grid_levels]
                    elif strategy.trailing_down and self.grid_levels and price <= self.grid_levels[0]:
                        shift = price - self.grid_levels[0]
                        self.grid_levels = [l + shift for l in self.grid_levels]

                    # SELL condition: moved up to next grid
                    if self.last_grid_index < len(self.grid_levels) - 1:
                        next_level = self.grid_levels[self.last_grid_index + 1]
                        if price >= next_level:
                            await self.execute_trade('sell', price)
                            self.last_grid_index += 1
                            continue

                    # BUY condition: moved down to previous grid
                    if self.last_grid_index > 0:
                        prev_level = self.grid_levels[self.last_grid_index - 1]
                        if price <= prev_level:
                            await self.execute_trade('buy', price)
                            self.last_grid_index -= 1
                            continue

                    await asyncio.sleep(5)
                except Exception as e:
                    logger.error(f"Bot {self.bot_id} Loop Error: {e}")
                    await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Bot {self.bot_id} Critical Failure: {e}")
        finally:
            if self.exchange:
                await self.exchange.close()

    async def execute_trade(self, side, price):
        try:
            investment_per_grid = self.config.strategy.investment / max(1, self.config.strategy.grids)
            amount = investment_per_grid / price
            symbol = self.pair
            if self.market_precision:
                formatted_amount = self.exchange.amount_to_precision(symbol, amount)
                formatted_price = self.exchange.price_to_precision(symbol, price)
            else:
                formatted_amount = amount
                formatted_price = price
            order = await self.exchange.create_order(symbol=self.pair, type='limit', side=side, amount=formatted_amount, price=formatted_price)
            logger.info(f"✅ Bot {self.bot_id}: {side.upper()} order placed at {formatted_price} (id: {order.get('id')})")
        except Exception as e:
            logger.error(f"❌ Bot {self.bot_id} Trade Failed: {e}")

# --- API ENDPOINTS ---
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
    active_bots[config.bot_id] = {"instance": bot, "task": task, "config": config.dict()}
    return {"message": "Bot started successfully", "bot_id": config.bot_id}

@app.post("/stop/{bot_id}")
async def stop_bot_endpoint(bot_id: int):
    if bot_id not in active_bots:
        raise HTTPException(status_code=404, detail="Bot not found")
    bot_entry = active_bots[bot_id]
    bot_entry["instance"].running = False
    try:
        await asyncio.wait_for(bot_entry["task"], timeout=5.0)
    except Exception:
        pass
    del active_bots[bot_id]
    return {"message": "Bot stopped successfully"}
