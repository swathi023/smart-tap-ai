"""
Smart Tap AI - Synthetic Dataset Generator
---------------------------------------------
Generates simulated tap-in/tap-out trip records with a crowd_level
label (LOW/MEDIUM/HIGH), using the same rule thresholds as the
rule-based fallback in crowd_optimization.py - so the "ground truth"
used to train the model is consistent with the rest of the pipeline.

Run:
    python src/dataset_generator.py --rows 3000 --out trips.csv
"""

import argparse
import csv
import random

from crowd_optimization import PEAK_HOURS, HIGH_PASSENGER_THRESHOLD, MEDIUM_PASSENGER_THRESHOLD

ROUTE_STOPS_KM = [0.0, 3.2, 7.8, 14.5, 21.0]


def label_crowd_level(hour, passenger_count_today):
    is_peak = hour in PEAK_HOURS
    if is_peak and passenger_count_today >= HIGH_PASSENGER_THRESHOLD:
        return "HIGH"
    if is_peak or passenger_count_today >= MEDIUM_PASSENGER_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def generate_row(passenger_count_today):
    hour = random.randint(5, 22)          # bus operating hours
    day_of_week = random.randint(0, 6)
    board, exit_ = random.sample(ROUTE_STOPS_KM, 2)
    distance_km = round(abs(exit_ - board), 2)

    # Small label noise (~5%) so the classifier has to learn real
    # boundaries instead of memorizing a lookup table.
    label = label_crowd_level(hour, passenger_count_today)
    if random.random() < 0.05:
        label = random.choice(["LOW", "MEDIUM", "HIGH"])

    return {
        "hour": hour,
        "day_of_week": day_of_week,
        "distance_km": distance_km,
        "passenger_count_today": passenger_count_today,
        "crowd_level": label,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--out", default="trips.csv")
    args = parser.parse_args()

    rows = []
    passenger_count_today = 0
    for i in range(args.rows):
        if i % 50 == 0:  # simulate the counter resetting across multiple "days"
            passenger_count_today = random.randint(0, 5)
        else:
            passenger_count_today += random.randint(1, 3)
        rows.append(generate_row(passenger_count_today))

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
