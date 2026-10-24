"""
Smart Tap AI - Crowd-Level Prediction & Optimization Layer
------------------------------------------------------------
This module sits directly on top of the trained Decision Tree
classifier. It converts a model's crowd-level prediction into a
concrete scheduling recommendation - this is the "optimization"
half of the project, separate from the "prediction" half.

Two things live here:
  1. predict_crowd_level() - wraps the trained model (or a
     rule-based fallback) and returns one of LOW / MEDIUM / HIGH.
  2. recommend_action() - maps a crowd level to a scheduling
     recommendation (headway, action, human-readable message).

Import this from hardware_bridge.py or from your evaluation/
results notebook - same logic, one source of truth.
"""

CROWD_LEVELS = ["LOW", "MEDIUM", "HIGH"]

# Thresholds used both for (a) labeling synthetic training data and
# (b) the rule-based fallback when no trained model is available yet.
PEAK_HOURS = set(list(range(7, 11)) + list(range(17, 21)))  # 7-10am, 5-8pm
HIGH_PASSENGER_THRESHOLD = 40    # passengers logged so far "today"
MEDIUM_PASSENGER_THRESHOLD = 15


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

def predict_crowd_level(model, hour: int, day_of_week: int,
                        distance_km: float, passenger_count_today: int) -> str:
    """
    Returns one of 'LOW', 'MEDIUM', 'HIGH'.

    If `model` is a trained scikit-learn classifier (loaded via
    joblib), its prediction is used directly - it must have been
    trained on columns in this exact order/name:
        [hour, day_of_week, distance_km, passenger_count_today]
    with target labels 'LOW' / 'MEDIUM' / 'HIGH'.

    If `model` is None, a rule-based fallback is used instead, so
    the pipeline is fully testable before the model exists.
    """
    if model is not None:
        import pandas as pd
        features = pd.DataFrame([{
            "hour": hour,
            "day_of_week": day_of_week,
            "distance_km": distance_km,
            "passenger_count_today": passenger_count_today,
        }])
        pred = str(model.predict(features)[0]).upper()
        return pred if pred in CROWD_LEVELS else "LOW"

    # --- rule-based fallback ---
    is_peak = hour in PEAK_HOURS
    if is_peak and passenger_count_today >= HIGH_PASSENGER_THRESHOLD:
        return "HIGH"
    if is_peak or passenger_count_today >= MEDIUM_PASSENGER_THRESHOLD:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------
# Optimization / recommendation layer
# ---------------------------------------------------------------------

# Simple rule table: crowd level -> recommended headway + action.
# This is intentionally a lookup table, not a solver - call it
# "rule-based resource allocation informed by ML-predicted crowd
# levels" in the report, not "AI-driven optimization".
RECOMMENDATIONS = {
    "LOW": {
        "headway_minutes": 20,
        "action_code": "NORMAL",
        "message": "Standard headway - no change needed.",
    },
    "MEDIUM": {
        "headway_minutes": 12,
        "action_code": "REDUCE_HW",
        "message": "Reduce headway - increase frequency moderately.",
    },
    "HIGH": {
        "headway_minutes": 6,
        "action_code": "ADD_BUS",
        "message": "Deploy additional bus - high crowd predicted.",
    },
}


def recommend_action(crowd_level: str) -> dict:
    """
    Maps a predicted crowd level to a scheduling recommendation.
    Returns a dict with headway_minutes, action_code (short, for
    serial/display use), and message (human-readable, for reports).
    """
    return RECOMMENDATIONS.get(crowd_level.upper(), RECOMMENDATIONS["LOW"])


# ---------------------------------------------------------------------
# Greedy bus allocation across a day's predicted time-slots
# (the "step up" version - a genuine, if simple, optimization pass
#  over a fixed bus fleet, for the Results chapter)
# ---------------------------------------------------------------------

def allocate_buses(slot_crowd_levels: list, total_buses: int, base_buses_per_slot: int = 1) -> list:
    """
    Given a predicted crowd level per time-slot (e.g. per hour of the
    day) and a fixed total bus fleet, greedily allocate extra buses
    to the highest-crowd slots first.

    slot_crowd_levels: list of 'LOW'/'MEDIUM'/'HIGH', one per slot
    total_buses: total buses available across the whole day
    base_buses_per_slot: minimum buses guaranteed per slot

    Returns a list of ints - buses allocated per slot, same order
    as slot_crowd_levels.
    """
    n_slots = len(slot_crowd_levels)
    if n_slots == 0:
        return []

    allocation = [base_buses_per_slot] * n_slots
    remaining = total_buses - base_buses_per_slot * n_slots
    if remaining <= 0:
        return allocation

    severity = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    # Slots sorted by severity (highest first); ties keep original order
    order = sorted(range(n_slots), key=lambda i: -severity.get(slot_crowd_levels[i], 0))

    # Round-robin extra buses across non-LOW slots first, then all slots
    # if buses are still left after every non-LOW slot has 2 extra.
    pass_num = 0
    while remaining > 0:
        made_progress = False
        for i in order:
            if remaining <= 0:
                break
            if slot_crowd_levels[i] == "LOW" and pass_num == 0:
                continue  # skip LOW slots on first pass
            allocation[i] += 1
            remaining -= 1
            made_progress = True
        pass_num += 1
        if not made_progress:  # safety valve
            break

    return allocation


if __name__ == "__main__":
    demo_levels = ["LOW", "LOW", "MEDIUM", "HIGH", "HIGH", "MEDIUM", "LOW"]
    print("Crowd levels per slot:", demo_levels)
    print("Bus allocation (12 total, 1 base each):", allocate_buses(demo_levels, total_buses=12))

    for level in CROWD_LEVELS:
        print(level, "->", recommend_action(level))
