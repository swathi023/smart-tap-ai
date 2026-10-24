"""
Smart Tap AI - Crowd Intelligence Dashboard
Run (from the repository root):
    streamlit run dashboard/crowd_dashboard.py
Required:
    decision_tree_model.pkl
Optional:
    trip_log.csv  (recorded tap events)
Install:
    pip install streamlit plotly pandas scikit-learn joblib
"""
import os
from datetime import datetime
from math import asin, cos, radians, sin, sqrt

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MODEL_PATH = "decision_tree_model.pkl"
LOG_PATH = "trip_log.csv"
FEATURES = ["hour", "day_of_week", "distance_km", "passenger_count_today"]
LEVELS = ["LOW", "MEDIUM", "HIGH"]
SCORE = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
COLOR = {"LOW": "#22C55E", "MEDIUM": "#FBBF24", "HIGH": "#FB7185"}
TEXT_ON = {"LOW": "#052E16", "MEDIUM": "#422006", "HIGH": "#4C0519"}

LOCATIONS = {
    "Majestic": (12.9767, 77.5713),
    "Bagaluru": (13.1180, 77.6700),
    "Indiranagar": (12.9719, 77.6412),
    "Hebbal": (13.0358, 77.5970),
    "Silk Board": (12.9177, 77.6228),
    "Malleshwaram": (13.0035, 77.5646),
    "Shivajinagar": (12.9857, 77.6057),
    "Yelahanka": (13.1007, 77.5963),
}
NAMES = list(LOCATIONS)
TRAINED_DISTANCES = [3.2, 4.6, 6.5, 6.7, 7.8, 11.3, 13.2, 14.5, 17.8, 21.0]
MAX_PASSENGERS = 115
HOURS = list(range(5, 23))
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

st.set_page_config(
    page_title="Smart Tap AI | Crowd Intelligence",
    page_icon="\U0001F68C",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { color-scheme: dark; }
.stApp {
    background:
        radial-gradient(circle at 85% 0%, rgba(14,165,233,.13), transparent 27%),
        radial-gradient(circle at 5% 20%, rgba(99,102,241,.10), transparent 25%),
        #07111F;
    color: #E5EDF7;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1728 0%, #091321 100%);
    border-right: 1px solid rgba(148,163,184,.14);
}
.block-container { padding-top: 1.7rem; padding-bottom: 2rem; max-width: 1500px; }
h1, h2, h3, [data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif !important; }
h1 { letter-spacing: -1.2px; }
p, label, .stCaption { color: #9FB0C5; }
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(19,36,58,.96), rgba(12,25,42,.96));
    border: 1px solid rgba(148,163,184,.16);
    border-radius: 18px;
    padding: 18px 20px;
    box-shadow: 0 12px 28px rgba(0,0,0,.12);
}
div[data-testid="stMetricLabel"] { color: #9FB0C5; }
div[data-testid="stMetricValue"] { color: #F3F7FC; }
.stButton button, .stFormSubmitButton button {
    border-radius: 10px; border: 1px solid #38BDF8;
    background: #0EA5E9; color: #03111E; font-weight: 700;
}
.stButton button:hover, .stFormSubmitButton button:hover {
    background: #38BDF8; color: #03111E; border-color: #7DD3FC;
}
[data-testid="stSelectbox"] > div > div, [data-testid="stNumberInput"] > div > div {
    background: #102036; border-color: rgba(148,163,184,.25); border-radius: 10px;
}
hr { border-color: rgba(148,163,184,.16); }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #E5EDF7; }
.small-muted { color:#8EA2BB; font-size: .88rem; }
.section-heading { font-family:'Space Grotesk',sans-serif; font-size:1.25rem; font-weight:700; color:#EAF2FC; margin: .35rem 0 .2rem 0; }
.route-pill {
    display:inline-block; padding:7px 12px; border:1px solid rgba(56,189,248,.3);
    background:rgba(14,165,233,.10); color:#BAE6FD; border-radius:999px;
    font-size:.85rem; font-weight:600;
}
</style>
""", unsafe_allow_html=True)


def distance_km(a, b):
    la1, lo1 = map(radians, LOCATIONS[a])
    la2, lo2 = map(radians, LOCATIONS[b])
    h = sin((la2 - la1) / 2) ** 2 + cos(la1) * cos(la2) * sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(h))


def model_distance(raw):
    return min(TRAINED_DISTANCES, key=lambda d: abs(d - raw))


def live_passenger_count():
    """Return today's recorded TAP_IN count, or None when unavailable."""
    if not os.path.exists(LOG_PATH):
        return None
    try:
        log = pd.read_csv(LOG_PATH)
    except Exception:
        return None
    if "event" in log.columns:
        log = log[log["event"].astype(str).str.upper() == "TAP_IN"]
    if "timestamp" in log.columns:
        timestamps = pd.to_datetime(log["timestamp"], errors="coerce")
        return int((timestamps.dt.date == datetime.now().date()).sum())
    return len(log)


def hour_label(h):
    return f"{(h - 1) % 12 + 1}:00 {'AM' if h < 12 else 'PM'}"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def to_level(label):
    if isinstance(label, str):
        value = label.upper()
        if value in LEVELS:
            return value
        raise ValueError(f"Unknown crowd label from model: {label}")
    return LEVELS[int(label)]


def predict_day(model, day, passengers, dist):
    frame = pd.DataFrame({
        "hour": HOURS,
        "day_of_week": day,
        "distance_km": dist,
        "passenger_count_today": passengers,
    })
    predictions = model.predict(frame[FEATURES])
    frame["level"] = [to_level(p) for p in predictions]
    frame["score"] = frame["level"].map(SCORE)
    return frame


def trip_chart(frame, current_hour):
    xs = [hour_label(h) for h in frame["hour"]]
    scores = frame["score"].tolist()
    levels = frame["level"].tolist()
    fig = go.Figure()

    # Subtle colored zones behind the curve
    for level in LEVELS:
        score = SCORE[level]
        fig.add_hrect(
            y0=score - 0.5, y1=score + 0.5,
            fillcolor=COLOR[level], opacity=0.075, line_width=0,
            layer="below"
        )

    # Glow layer and main curve
    fig.add_trace(go.Scatter(
        x=xs, y=scores, mode="lines",
        line=dict(color="rgba(56,189,248,.18)", width=13, shape="spline", smoothing=0.8),
        hoverinfo="skip", showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=xs, y=scores, mode="lines",
        line=dict(color="#38BDF8", width=3.5, shape="spline", smoothing=0.8),
        hoverinfo="skip", showlegend=False
    ))

    # Colored markers and detailed hover labels
    custom = [[level, hour_label(hour)] for level, hour in zip(levels, frame["hour"])]
    fig.add_trace(go.Scatter(
        x=xs, y=scores, mode="markers",
        marker=dict(
            size=[15 if h == current_hour else 10 for h in frame["hour"]],
            color=[COLOR[level] for level in levels],
            line=dict(color="#EAF2FC", width=[2.5 if h == current_hour else 1 for h in frame["hour"]]),
        ),
        customdata=custom,
        hovertemplate="<b>%{customdata[1]}</b><br>Crowd status: <b>%{customdata[0]}</b><extra></extra>",
        showlegend=False
    ))

    current_rows = frame[frame["hour"] == current_hour]
    if not current_rows.empty:
        row = current_rows.iloc[0]
        fig.add_annotation(
            x=hour_label(current_hour), y=row["score"],
            text="NOW", showarrow=True, arrowhead=2, arrowsize=1,
            arrowwidth=1.5, arrowcolor="#E2E8F0", ax=0, ay=-38,
            font=dict(color="#EAF2FC", size=10),
            bgcolor="#102036", bordercolor="rgba(148,163,184,.35)",
            borderpad=5
        )

    fig.update_layout(
        height=430,
        margin=dict(l=8, r=18, t=28, b=55),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        font=dict(family="DM Sans, sans-serif", color="#9FB0C5", size=11),
        xaxis=dict(
            categoryorder="array", categoryarray=xs,
            showgrid=False, tickangle=-35, ticks="outside",
            tickcolor="rgba(148,163,184,.25)", linecolor="rgba(148,163,184,.15)"
        ),
        yaxis=dict(
            tickvals=[0, 1, 2], ticktext=["LOW", "MEDIUM", "HIGH"],
            range=[-0.45, 2.45], showgrid=True,
            gridcolor="rgba(148,163,184,.10)", zeroline=False,
            ticks="", linecolor="rgba(148,163,184,.12)"
        ),
        hoverlabel=dict(
            bgcolor="#102036", bordercolor="#38BDF8",
            font=dict(color="#F3F7FC", family="DM Sans, sans-serif", size=12)
        )
    )
    return fig


# Header
left, right = st.columns([3, 1])
with left:
    st.markdown("<div class='small-muted'>SMART TAP AI / TRANSIT ANALYTICS</div>", unsafe_allow_html=True)
    st.title("Crowd Intelligence")
    st.markdown("<span class='route-pill'>\u25CF LIVE-STYLE PREDICTION VIEW</span> &nbsp; <span class='small-muted'>Route-level crowd outlook by hour</span>", unsafe_allow_html=True)
with right:
    st.markdown(
        f"<div style='text-align:right;padding-top:18px;color:#8EA2BB;font-size:.9rem'>"
        f"<span style='color:#38BDF8'>\u25CF</span> MODEL STATUS<br>"
        f"<b style='color:#E5EDF7;font-size:1rem'>Decision Tree</b><br>"
        f"{datetime.now().strftime('%d %b %Y \u00B7 %I:%M %p')}</div>",
        unsafe_allow_html=True
    )

try:
    model = load_model()
except Exception as exc:
    st.error(
        f"Could not load **{MODEL_PATH}**. Place the trained model in the same folder "
        f"as this dashboard and run `train_model.py` first.\n\nDetails: {exc}"
    )
    st.stop()

# Sidebar inputs
if "smart_tap_inputs" not in st.session_state:
    st.session_state.smart_tap_inputs = {
        "tap_in": "Majestic",
        "tap_out": "Bagaluru",
        "day": datetime.now().weekday(),
        "passenger_count": 50,
        "source": "Enter passenger count",
    }

saved = st.session_state.smart_tap_inputs

with st.sidebar:
    st.markdown("## \U0001F68C Trip setup")
    st.caption("Choose a route and set the passenger data for the prediction.")
    with st.form("trip_settings"):
        tap_in = st.selectbox("Tap In location", NAMES, index=NAMES.index(saved["tap_in"]))
        tap_out = st.selectbox("Tap Out location", NAMES, index=NAMES.index(saved["tap_out"]))
        day_name = st.selectbox("Travel day", DAYS, index=saved["day"])
        source = st.radio(
            "Passenger data",
            ["Enter passenger count", "Recorded tap-ins"],
            index=["Enter passenger count", "Recorded tap-ins"].index(saved["source"]),
        )
        passenger_count = st.number_input(
            "Passenger Count",
            min_value=0, max_value=MAX_PASSENGERS,
            value=int(saved["passenger_count"]), step=5,
            help="Used when 'Enter passenger count' is selected."
        )
        submitted = st.form_submit_button("Update dashboard", use_container_width=True)

    if submitted:
        if tap_in == tap_out:
            st.warning("Choose two different locations for Tap In and Tap Out.")
        else:
            st.session_state.smart_tap_inputs = {
                "tap_in": tap_in,
                "tap_out": tap_out,
                "day": DAYS.index(day_name),
                "passenger_count": int(passenger_count),
                "source": source,
            }

    st.divider()
    st.markdown("### Crowd status guide")
    st.markdown(
        f"<div style='padding:9px 12px;border-radius:10px;background:{COLOR['LOW']}22;margin-bottom:7px;color:#BBF7D0'>\u25CF <b>LOW</b> &nbsp; Easier boarding</div>"
        f"<div style='padding:9px 12px;border-radius:10px;background:{COLOR['MEDIUM']}22;margin-bottom:7px;color:#FDE68A'>\u25CF <b>MEDIUM</b> &nbsp; Moderate crowd</div>"
        f"<div style='padding:9px 12px;border-radius:10px;background:{COLOR['HIGH']}22;color:#FECDD3'>\u25CF <b>HIGH</b> &nbsp; Busy conditions</div>",
        unsafe_allow_html=True
    )

saved = st.session_state.smart_tap_inputs
raw_dist = distance_km(saved["tap_in"], saved["tap_out"])
dist = model_distance(raw_dist)

if saved["source"] == "Recorded tap-ins":
    recorded_count = live_passenger_count()
    if recorded_count is None:
        passengers = int(saved["passenger_count"])
        source_label = "Entered count \u00B7 no tap log found"
        st.warning(f"{LOG_PATH} is not available or could not be read. The entered passenger count is being used.")
    else:
        passengers = recorded_count
        source_label = "Recorded tap-ins today"
else:
    passengers = int(saved["passenger_count"])
    source_label = "Entered passenger count"

passengers = min(passengers, MAX_PASSENGERS)
df = predict_day(model, saved["day"], passengers, dist)

now_hour = min(max(datetime.now().hour, HOURS[0]), HOURS[-1])
now_level = df.loc[df["hour"] == now_hour, "level"].iloc[0]
busy_row = df.loc[df["score"].idxmax()]
peak_hours = df[df["score"] == df["score"].max()]["hour"].tolist()
peak_label = hour_label(peak_hours[0])

st.markdown(
    f"<div class='small-muted'>SELECTED ROUTE</div>"
    f"<h2 style='margin-top:2px;margin-bottom:4px'>{saved['tap_in']} "
    f"<span style='color:#38BDF8'>\u2192</span> {saved['tap_out']}</h2>"
    f"<div class='small-muted'>{DAYS[saved['day']]} &nbsp; \u00B7 &nbsp; "
    f"Straight-line estimate {raw_dist:.1f} km &nbsp; \u00B7 &nbsp; Model distance {dist:.1f} km</div>",
    unsafe_allow_html=True
)
st.write("")

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Current crowd level", now_level, help="Model prediction for the current hour.")
with m2:
    st.metric("Passenger Count", f"{passengers:,}", help=f"Data source: {source_label}.")
with m3:
    st.metric("Route distance", f"{raw_dist:.1f} km", f"Model uses {dist:.1f} km")
with m4:
    st.metric("Busiest predicted hour", peak_label, f"{busy_row['level']} crowd")

st.write("")
chart_col, insight_col = st.columns([2.25, 1], gap="large")

with chart_col:
    st.markdown("<div class='section-heading'>Crowd forecast timeline</div>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Hourly prediction \u00B7 hover over a marker for details</div>", unsafe_allow_html=True)
    st.plotly_chart(trip_chart(df, now_hour), use_container_width=True, config={"displayModeBar": False}, key="crowd_forecast")

with insight_col:
    st.markdown("<div class='section-heading'>Route insights</div>", unsafe_allow_html=True)
    st.write("")
    st.markdown(
        f"<div style='background:linear-gradient(145deg,#12253B,#0D1A2B);border:1px solid rgba(148,163,184,.16);border-radius:16px;padding:18px;margin-bottom:12px'>"
        f"<div class='small-muted'>CURRENT STATUS</div>"
        f"<div style='font-size:1.7rem;font-weight:700;color:{COLOR[now_level]};margin:5px 0'>{now_level}</div>"
        f"<div style='color:#9FB0C5;font-size:.9rem'>Prediction for {hour_label(now_hour)}</div></div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div style='background:linear-gradient(145deg,#12253B,#0D1A2B);border:1px solid rgba(148,163,184,.16);border-radius:16px;padding:18px;margin-bottom:12px'>"
        f"<div class='small-muted'>BUSIEST PERIOD</div>"
        f"<div style='font-size:1.35rem;font-weight:700;color:#FBBF24;margin:5px 0'>{peak_label}</div>"
        f"<div style='color:#9FB0C5;font-size:.9rem'>{busy_row['level']} predicted crowd</div></div>",
        unsafe_allow_html=True
    )
    level_counts = df["level"].value_counts()
    st.markdown("<div class='small-muted' style='margin:8px 0'>PREDICTED HOURS BY LEVEL</div>", unsafe_allow_html=True)
    for level in LEVELS:
        count = int(level_counts.get(level, 0))
        percent = count / len(df) * 100
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;margin:10px 0 4px;color:#C7D5E6;font-size:.88rem'>"
            f"<span><span style='color:{COLOR[level]}'>\u25CF</span> {level}</span><b>{count} hrs</b></div>"
            f"<div style='height:7px;background:#1D3047;border-radius:99px;overflow:hidden'>"
            f"<div style='width:{percent:.1f}%;height:100%;background:{COLOR[level]};border-radius:99px'></div></div>",
            unsafe_allow_html=True
        )

st.write("")
st.markdown("<div class='section-heading'>Hourly prediction details</div>", unsafe_allow_html=True)
details = df[["hour", "level"]].copy()
details["Time"] = details["hour"].map(hour_label)
details["Crowd level"] = details["level"]
details["Status"] = details["level"].map({"LOW": "Less crowded", "MEDIUM": "Moderate", "HIGH": "Busy"})
details = details[["Time", "Crowd level", "Status"]]
st.dataframe(
    details,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Time": st.column_config.TextColumn("Time"),
        "Crowd level": st.column_config.TextColumn("Crowd level"),
        "Status": st.column_config.TextColumn("Interpretation"),
    },
)

st.divider()
st.markdown(
    "<div class='small-muted'>Prototype note: predictions are generated locally by the trained Decision Tree model. "
    "Stop coordinates are approximate, trip distance is a straight-line estimate, and results depend on the training dataset. "
    "This dashboard does not represent verified real-time BMTC crowd measurements.</div>",
    unsafe_allow_html=True
)
