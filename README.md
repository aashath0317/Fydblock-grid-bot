# ⚡ FydBlock Trading Engine

The high-performance, asynchronous trading core for **FydBlock**.  
Built with **Python** and **FastAPI**, this microservice executes algorithmic trading strategies (Grid, DCA) across multiple exchanges simultaneously using `ccxt.async_support`.

It runs as a background engine that receives commands from the main **Node.js backend** to start, stop, and manage user trading bots.

---

## 🚀 Tech Stack

- **Language:** Python 3.9+
- **Framework:** FastAPI (Async, high-performance)
- **Server:** Uvicorn (ASGI)
- **Trading Library:** CCXT (Async)
- **Concurrency:** Python `asyncio`, BackgroundTasks

---

## ✨ Features

- **Multi-User Engine** — Runs isolated trading loops for hundreds of users.
- **Real-Time Execution** — Non-blocking async price updates & order execution.
- **Grid Strategy Built-In** — Buy-Low / Sell-High automation for Spot markets.
- **Multi-Exchange Support** — Binance, Bybit, OKX, KuCoin, and 100+ via CCXT.
- **Fully API-Driven** — Backend controls everything via HTTP endpoints.
- **Stateless Design** — Strategy settings & API keys passed dynamically.

---

## 🛠 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/fydblock_engine.git
cd fydblock_engine
```

### 2. Create a Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

If requirements are missing:
```bash
pip install fastapi uvicorn ccxt pydantic python-dotenv
```

---

## ⚙️ Configuration

This engine is **stateless** — all bot parameters & exchange keys are passed through the `/start` endpoint.

Optional `.env` file for server configuration:

```env
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info
```

---

## 🏃‍♂️ Running the Engine

### Development
```bash
uvicorn bot_engine:app --reload --port 8000
```

### Production (Recommended: PM2)
```bash
pm2 start "uvicorn bot_engine:app --host 0.0.0.0 --port 8000" --name "fydblock-engine"
pm2 save
```

---

## 📡 API Endpoints

Interactive API docs:  
**http://your-server-ip:8000/docs**

### 1. **Start a Bot**
**POST `/start`**

```json
{
  "bot_id": 101,
  "user_id": 55,
  "exchange": "binance",
  "pair": "BTC/USDT",
  "api_key": "user_exchange_api_key",
  "api_secret": "user_exchange_secret",
  "strategy": {
    "upper_price": 65000,
    "lower_price": 55000,
    "grids": 20,
    "investment": 1000
  }
}
```

### 2. **Stop a Bot**
**POST `/stop/{bot_id}`**

Stops the trading loop instantly.

### 3. **Check Bot Status**
**GET `/status/{bot_id}`**

Returns:  
- Is running or stopped  
- Current strategy parameters  
- Exchange + pair info  

### 4. **Health Check**
**GET `/`**

Returns engine uptime and active bot count.

---

## 🤝 Integration with Node.js Backend

Example backend call:

```javascript
await axios.post('http://localhost:8000/start', {
    bot_id: bot.id,
    user_id: user.id,
    exchange: "binance",
    pair: "BTC/USDT",
    api_key: user.apiKey,
    api_secret: user.secret,
    strategy: botConfig
});
```

---

## 📄 License

**MIT License**

---

Let me know if you want:
✅ Docker version  
✅ Folder structure section  
✅ Add WebSocket support for live bot logs  
