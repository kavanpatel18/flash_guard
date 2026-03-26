# ⚡ FlashGuard — Flash Crash Prediction

BiGRU + Attention neural network for real-time NSE flash crash detection.

## Quick Start

```bash
# 1. Enter the project folder
cd flashguard

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Launch the server
python api_server.py

# 4. Open your browser
#    → http://localhost:5000          (landing page)
#    → http://localhost:5000/dashboard.html   (terminal)
```

## Project Structure

```
flashguard/
├── api_server.py                    ← Flask REST API
├── requirements.txt                 ← Python dependencies
├── start.sh                         ← One-click launcher (Linux/macOS)
├── improved_flash_crash_model.keras ← Primary model
├── improved_minute_model.keras      ← Minute-bar model
└── frontend/
    ├── index.html                   ← Landing page
    ├── dashboard.html               ← Trading terminal
    ├── style.css                    ← Cyberpunk theme
    └── app.js                       ← Dashboard logic
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/models` | List loaded models |
| POST | `/api/predict` | Single ticker prediction |
| POST | `/api/portfolio` | Multi-ticker scan |
| POST | `/api/upload` | CSV file analysis |
| POST | `/api/timeline` | Rolling risk timeline |
| GET | `/api/crash-events` | Available crash replays |
| POST | `/api/crash-replay` | Replay a crash event |

## Optional: Upstox Live Data

Paste your Upstox access token into the **UPSTOX ACCESS TOKEN** field in the
dashboard for live NSE minute-bar data. Without it, the app falls back to
yFinance (delayed) or demo data automatically.

## Notes

- Models must be in the **same folder** as `api_server.py`.
- For NSE stocks always include `.NS` suffix (e.g. `RELIANCE.NS`).
- yFinance 1-minute data is only available for the last 7 days.
