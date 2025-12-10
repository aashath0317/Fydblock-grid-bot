import asyncio
import ccxt.async_support as ccxt  # Async version of CCXT
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import logging

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GridBot")

app = FastAPI(title="FydBlock Trading Engine")

# Store active bot instances in memory
# Format: { bot_id: { "task": asyncio.Task, "status": "running", "config": dict } }
active_bots: Dict[int, dict] = {}

# Exchange instances cache to avoid overhead
exchange_cache = {}

# --- DATA MODELS ---
class StrategyConfig(BaseModel):
    upper_price: float
    lower_price: float
    grids: int
    investment: float

class BotRequest(BaseModel):
    bot_id: int
    user_id: int
    exchange: str      # e.g., 'binance', 'bybit'
    pair: str          # e.g., 'BTC/USDT'
    api_key: str
    api_secret: str
    passphrase: Optional[str] = None
    strategy: StrategyConfig

# --- GRID LOGIC CLASS ---
class GridBot:
    def __init__(self, config: BotRequest):
        self.bot_id = config.bot_id
        self.user_id = config.user_id
        self.pair = config.pair
        self.config = config
        self.exchange_id = config.exchange.lower()
        self.exchange = None
        self.running = False
        self.grid_levels = []
        self.last_grid_index = -1 # Tracks where we are in the grid

    async def initialize_exchange(self):
        """Initializes CCXT exchange instance with specific user keys."""
        exchange_class = getattr(ccxt, self.exchange_id)
        self.exchange = exchange_class({
            'apiKey': self.config.api_key,
            'secret': self.config.api_secret,
            'password': self.config.passphrase,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'} 
        })
        # Load markets to ensure symbol exists and get precision info
        await self.exchange.load_markets()

    async def calculate_grids(self):
        """Calculates price levels for the grid."""
        upper = self.config.strategy.upper_price
        lower = self.config.strategy.lower_price
        count = self.config.strategy.grids
        
        step = (upper - lower) / count
        self.grid_levels = [lower + (i * step) for i in range(count + 1)]
        logger.info(f"Bot {self.bot_id}: Calculated {len(self.grid_levels)} grid levels.")

    async def run(self):
        """The Main Infinite Loop"""
        self.running = True
        logger.info(f"🚀 Bot {self.bot_id} Started for {self.pair}")

        try:
            await self.initialize_exchange()
            await self.calculate_grids()

            # Initial Price Check to find starting position
            ticker = await self.exchange.fetch_ticker(self.pair)
            current_price = ticker['last']
            
            # Find closest grid level index
            self.last_grid_index = min(
                range(len(self.grid_levels)), 
                key=lambda i: abs(self.grid_levels[i] - current_price)
            )

            while self.running:
                try:
                    # 1. Fetch Real-Time Price
                    ticker = await self.exchange.fetch_ticker(self.pair)
                    price = ticker['last']

                    # 2. Check Logic
                    # Crossed UP -> SELL
                    if self.last_grid_index < len(self.grid_levels) - 1:
                        next_level = self.grid_levels[self.last_grid_index + 1]
                        if price >= next_level:
                            await self.execute_trade('sell', price)
                            self.last_grid_index += 1
                            continue # Skip sleep to catch rapid moves

                    # Crossed DOWN -> BUY
                    if self.last_grid_index > 0:
                        prev_level = self.grid_levels[self.last_grid_index - 1]
                        if price <= prev_level:
                            await self.execute_trade('buy', price)
                            self.last_grid_index -= 1
                            continue

                    # 3. Wait before next check (prevents rate limiting)
                    await asyncio.sleep(2) 

                except ccxt.NetworkError as e:
                    logger.warning(f"Bot {self.bot_id} Network Error: {e}")
                    await asyncio.sleep(5)
                except Exception as e:
                    logger.error(f"Bot {self.bot_id} Loop Error: {e}")
                    await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"Bot {self.bot_id} Critical Failure: {e}")
        finally:
            if self.exchange:
                await self.exchange.close()
            logger.info(f"🛑 Bot {self.bot_id} Stopped.")

    async def execute_trade(self, side, price):
        """Executes a market order on the exchange."""
        try:
            # Calculate amount based on investment per grid
            # Simple logic: Investment / Grids / Price
            amount_usdt = self.config.strategy.investment / self.config.strategy.grids
            amount = amount_usdt / price 

            # Place Order
            order = await self.exchange.create_order(self.pair, 'market', side, amount)
            logger.info(f"✅ Bot {self.bot_id}: {side.upper()} executed at ${price} (ID: {order['id']})")
            
            # Optional: Call back to Node.js to save trade history to DB
            # await requests.post(NODE_CALLBACK_URL, json=...)

        except Exception as e:
            logger.error(f"Bot {self.bot_id} Trade Failed: {e}")

# --- API ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "online", "active_bots": len(active_bots)}

@app.post("/start")
async def start_bot_endpoint(config: BotRequest, background_tasks: BackgroundTasks):
    if config.bot_id in active_bots:
        raise HTTPException(status_code=400, detail="Bot already running")

    # Instantiate Bot
    bot = GridBot(config)
    
    # Run the bot.run() method as a background task
    # We store the bot instance to control it later
    task = asyncio.create_task(bot.run())
    
    active_bots[config.bot_id] = {
        "instance": bot,
        "task": task,
        "config": config.dict()
    }

    return {"message": "Bot started successfully", "bot_id": config.bot_id}

@app.post("/stop/{bot_id}")
async def stop_bot_endpoint(bot_id: int):
    if bot_id not in active_bots:
        raise HTTPException(status_code=404, detail="Bot not found")

    # Signal the bot to stop looping
    bot_entry = active_bots[bot_id]
    bot_instance = bot_entry["instance"]
    bot_instance.running = False
    
    # Wait for task to finish cleanup
    try:
        await bot_entry["task"]
    except asyncio.CancelledError:
        pass

    del active_bots[bot_id]
    return {"message": "Bot stopped successfully"}

@app.get("/status/{bot_id}")
def get_bot_status(bot_id: int):
    if bot_id in active_bots:
        return {"status": "running", "details": active_bots[bot_id]["config"]}
    return {"status": "stopped"}
