# Smart Tap AI for Public Bus Transport Optimization

A low-cost RFID tap-in / tap-out prototype for city buses, combined with a Decision Tree that predicts passenger **crowd level (LOW / MEDIUM / HIGH)** and turns it into scheduling recommendations (bus headway, extra buses). Includes a Streamlit dashboard that shows an hour-by-hour crowd forecast for a chosen route.

Final-year Major Project (Phase II), Dept. of AI & ML, Vijaya Vittala Institute of Technology, Bengaluru (VTU), 2026-27.
Team: Mounika A, Swathi R, Varsha P S. Guide: Prof. Sindhu K.
A paper related to this project was selected at IFSCEC (IEEE).

| LCD showing a detected card | Assembled prototype | Dashboard |
|---|---|---|
| ![LCD card detected](docs/images/lcd_card_detected.jpg) | ![Prototype](docs/images/assembled_prototype.jpg) | ![Dashboard](docs/images/dashboard_1.jpg) |

## What is in this repository

| Part | Files | What it does |
|---|---|---|
| **ESP32 firmware** | `firmware/smart_tap_firmware/` | Reads cards with an MFRC522 reader, treats the 1st tap as boarding and the 2nd as exit, shows the fare on a 16x2 I2C LCD, and prints machine-readable lines over serial (115200 baud). |
| **Dataset generator** | `src/dataset_generator.py` | Creates a synthetic trip dataset with crowd-level labels (about 5% label noise). |
| **Model training** | `src/train_model.py` | Compares Decision Tree, Random Forest and KNN with 5-fold CV, tunes the Decision Tree with `GridSearchCV`, and saves `decision_tree_model.pkl`. |
| **Prediction + scheduling** | `src/crowd_optimization.py` | `predict_crowd_level()` (model, with a rule-based fallback), `recommend_action()` (headway 20/12/6 min) and a greedy `allocate_buses()`. |
| **Dashboard** | `dashboard/crowd_dashboard.py` | Streamlit app: pick tap-in/tap-out stops, day and passenger count, see the predicted crowd level for every hour. |

## Hardware

ESP32, MFRC522 RFID reader (3.3 V only), 16x2 I2C LCD (address 0x27).

| Part | ESP32 pins |
|---|---|
| MFRC522 (SPI) | SCK 18, MOSI 23, MISO 19, SS 5, RST 4 |
| LCD (I2C) | SDA 21, SCL 22 |

The project report also describes a NEO-6M GPS module, buzzer, status LEDs and a push button, plus a Python `hardware_bridge.py` that connects the ESP32 to the model. **Those are not included in this repository's firmware**, which uses a fixed fare table per demo card (edit the `trips[]` table in the `.ino` file). Fares are demonstration values, not official BMTC fares.

## Machine learning

- **Features:** `hour`, `day_of_week`, `distance_km`, `passenger_count_today`
- **Labels:** LOW / MEDIUM / HIGH, generated from fixed rules (peak hours 7-10 am and 5-8 pm; passenger-count thresholds 15 and 40) plus about 5% random label noise.
- **Data:** synthetic (3,000 rows by default). No real ridership data was used.

### Results from one run

The generator is not seeded, so numbers change slightly every run.

| Model (5-fold CV on the training set) | Accuracy |
|---|---|
| Decision Tree (default) | 0.9338 +/- 0.0091 |
| Random Forest | 0.9688 +/- 0.0051 |
| KNN (k=5) | 0.8058 +/- 0.0165 |

Tuned Decision Tree (`criterion=gini, max_depth=5, min_samples_leaf=5`): **97.67% accuracy on a 600-row held-out test set.**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| LOW | 1.00 | 0.90 | 0.95 | 48 |
| MEDIUM | 0.98 | 0.99 | 0.99 | 380 |
| HIGH | 0.97 | 0.97 | 0.97 | 172 |

Full output: `results/training_log.txt`. Plots: `results/confusion_matrix.png`, `results/feature_importance.png`.

**How to read these numbers:** the labels come from the same rules the model sees as inputs (hour and passenger count), so the score shows that the tree learned the synthetic rules. It is **not** evidence of accuracy on real bus ridership. Class balance is also skewed (about 63% MEDIUM, 29% HIGH, 8% LOW), which is why per-class metrics are reported.

## Run it

```bash
git clone https://github.com/swathi023/smart-tap-ai.git
cd smart-tap-ai
pip install -r requirements.txt

# 1. generate data and train (from the repo root)
python src/dataset_generator.py --rows 3000 --out data/trips.csv
python src/train_model.py --data data/trips.csv --out decision_tree_model.pkl

# 2. try the scheduling layer
python src/crowd_optimization.py

# 3. launch the dashboard (needs decision_tree_model.pkl in the repo root)
streamlit run dashboard/crowd_dashboard.py
```

`train_model.py` also saves `confusion_matrix.png` and `feature_importance.png` in the folder you run it from.

**Firmware:** open `firmware/smart_tap_firmware/smart_tap_firmware.ino` in the Arduino IDE with the ESP32 board package and the `MFRC522` and `LiquidCrystal_I2C` libraries. Close the Serial Monitor before running any script that uses the serial port.

## Limitations

- Synthetic data only; real ridership data is the next step.
- Single-reader toggle logic (1st tap = in, 2nd = out) and at most 5 registered demo cards.
- Greedy bus allocation is locally optimal, not a full fleet optimizer.
- Dashboard stop coordinates are approximate and distances are straight-line estimates; it does not show real BMTC crowd data.

## Future work

Real data collection, separate boarding and exit readers, GPS-based distance, a payment gateway, a passenger mobile app and multi-route fleet scheduling.

## License

MIT, see `LICENSE`.
