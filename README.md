# 🧠 Mental Health Tracker
### Personal AI Diary with Federated Learning

> Your data never leaves your laptop. Only anonymous model weight updates
> are shared to improve the global model — never your diary, never your usage data.

---

## 📁 Project Structure

```
mental-health-tracker/
│
├── collector/
│   ├── data_collector.py     ← Background agent (runs silently)
│   ├── db.py                 ← Local SQLite database
│   ├── categories.py         ← App category classifier
│   ├── fl_client.py          ← Federated learning weight uploader
│   └── setup_autostart.py    ← Windows auto-start setup
│
├── model/                    ← GRU + Fusion model (coming next)
├── backend/                  ← FastAPI server (coming next)
├── frontend/                 ← React UI (coming next)
│
├── data/                     ← Created automatically
│   ├── tracker.db            ← Your local SQLite database
│   └── collector.log         ← Background agent logs
│
├── test_collector.py         ← Run this first to verify setup
└── requirements.txt          ← Python dependencies
```

---

## 🚀 Quick Start

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Verify everything works
```bash
python test_collector.py
```
All checks should show `[OK]`. Fix any `[FAIL]` items before continuing.

### Step 3 — Start collecting data
```bash
python collector/data_collector.py
```
This runs in your terminal. Keep it open, or proceed to Step 4.

### Step 4 — Set up auto-start (recommended)
Run this **once as Administrator**:
```bash
python collector/setup_autostart.py
```
After this, the collector starts automatically every time you log into Windows.
You don't need to do anything — it runs silently in the background.

---

## 📊 What Gets Collected

| Data | How | Stored |
|------|-----|--------|
| Active app name | `psutil` | Local SQLite |
| Window title | Windows API | Local SQLite |
| Time per app | Polling every 5s | Local SQLite |
| Keyboard activity | `pynput` | Count only (not what you typed) |
| Mouse movement | `pynput` | Distance only (not position) |
| Idle time | Input gap detection | Local SQLite |
| Screen time | Sum of active app time | Local SQLite |
| Late night usage | Time of day check | Local SQLite |
| Battery/charging | `psutil` | Local SQLite |

**What is NEVER collected:**
- ❌ What you type (only keystroke count)
- ❌ Screenshots
- ❌ Passwords or sensitive text
- ❌ Exact mouse positions (only distance moved)

---

## 🔒 Privacy Model

```
YOUR LAPTOP                          SERVER
─────────────────────────────        ──────────────────────
Diary entries        ──STAYS──→      ✗ Never sent
App usage data       ──STAYS──→      ✗ Never sent
Window titles        ──STAYS──→      ✗ Never sent
Raw predictions      ──STAYS──→      ✗ Never sent

Model weight delta   ──SENT──→       ✓ Anonymous weight update
+ Gaussian noise                       (differential privacy)
+ Norm clipping                        Aggregated with other users
                     ←──RECEIVED──   ✓ Improved global model
```

---

## 🛠️ Troubleshooting

**Collector won't start:**
```bash
pip install psutil pynput pywin32
```

**No window title being detected:**
- Run as Administrator for better window detection
- Or use the fallback mode (automatic if pywin32 not available)

**Auto-start not working:**
- Run `setup_autostart.py` as Administrator (right-click → Run as Admin)
- Check Task Scheduler: search "Task Scheduler" in Windows search

**Database errors:**
```bash
# Delete and recreate
del data\tracker.db
python test_collector.py
```

---

## 📈 Data Starts Accumulating From Today

The longer you let the collector run, the better your model becomes.
- Day 1–3: Baseline data collection
- Day 4+: Model can start making predictions
- Week 2+: Weekly analysis becomes meaningful
- Month 1+: Personalisation kicks in (federated learning)

---

## 🔜 Coming Next

- [ ] GRU text model
- [ ] Fusion model (text + numerical)
- [ ] FastAPI backend
- [ ] React UI (diary, tracker, insights)
- [ ] Federated learning server (Railway deployment)
