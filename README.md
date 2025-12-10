### **How to Deploy & Run**

1.  **Install Dependencies:**

    ```bash
    pip install fastapi uvicorn ccxt
    ```

2.  **Run the Server:**
    Run this command in your terminal (or use PM2 as discussed before):

    ```bash
    uvicorn bot_engine:app --host 0.0.0.0 --port 8000
    ```

3.  **Connect it to Node.js:**
    In your Node.js `userController.js`, simply send a POST request to start a bot:

    ```javascript
    const axios = require('axios');

    // ... inside createBot controller ...
    await axios.post('http://localhost:8000/start', {
        bot_id: newBot.rows[0].bot_id,
        user_id: req.user.id,
        exchange: 'binance', // from DB
        pair: 'BTC/USDT',
        api_key: decryptedKey,
        api_secret: decryptedSecret,
        strategy: {
            upper_price: 60000,
            lower_price: 50000,
            grids: 20,
            investment: 1000
        }
    });
    ```
