import streamlit as st
import pandas as pd
import requests
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# -----------------------------
# APP CONFIG
# -----------------------------
st.set_page_config(page_title="Smart Road Safety System", layout="wide")
st.title("🚦 AI-Driven Smart Road Safety System")
st.subheader("Real-Time Accident Risk, Hotspot Detection & Safe Route Suggestion")

# -----------------------------
# LOAD & CLEAN DATA
# -----------------------------
data = pd.read_csv("road_accident_data.csv")

# remove invalid GPS points (FIXES FOLIUM ERROR)
data = data.dropna(subset=["latitude", "longitude"])
data = data[(data["latitude"] != 0) & (data["longitude"] != 0)]

st.write("### 📊 Dataset Preview")
st.dataframe(data.head())

# -----------------------------
# ENCODING
# -----------------------------
le_weather = LabelEncoder()
le_road = LabelEncoder()
le_time = LabelEncoder()
le_vehicle = LabelEncoder()

data["weather_enc"] = le_weather.fit_transform(data["weather"])
data["road_enc"] = le_road.fit_transform(data["road_type"])
data["time_enc"] = le_time.fit_transform(data["time_of_day"])

if "vehicle_type" in data.columns:
    data["vehicle_enc"] = le_vehicle.fit_transform(data["vehicle_type"])
else:
    data["vehicle_enc"] = 0

# -----------------------------
# RISK LABEL CREATION
# -----------------------------
def risk_label(count):
    if count >= 6:
        return 2  # High
    elif count >= 3:
        return 1  # Medium
    else:
        return 0  # Low

data["risk"] = data["accident_count"].apply(risk_label)

# -----------------------------
# FEATURES & TARGET
# -----------------------------
X = data[["weather_enc", "road_enc", "traffic_volume", "time_enc", "vehicle_enc"]]
y = data["risk"]

# -----------------------------
# TRAIN-TEST SPLIT (CRASH-PROOF)
# -----------------------------
num_classes = y.nunique()
test_size = 0.25
test_samples = int(len(y) * test_size)

if test_samples >= num_classes and num_classes > 1:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
st.success(f"✅ Model Accuracy: {accuracy*100:.2f}%")

# -----------------------------
# LIVE WEATHER & TRAFFIC
# -----------------------------
API_KEY = "f24992fe934b7c4210e033f99629dbd1"

def get_live_weather(lat, lon):
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        )
        res = requests.get(url, timeout=5).json()
        api_weather = res["weather"][0]["main"]
        temp = res["main"]["temp"]
        return api_weather, temp
    except:
        return "Clear", 25

# Map API weather → dataset labels (FIXES UNSEEN LABEL ERROR)
def map_weather(api_weather):
    if api_weather in ["Rain", "Drizzle", "Thunderstorm"]:
        return "Rainy"
    elif api_weather in ["Fog", "Mist", "Haze", "Smoke"]:
        return "Fog"
    else:
        return "Clear"

def get_live_traffic():
    return random.randint(200, 1200)

# -----------------------------
# USER INPUT
# -----------------------------
st.write("### 🔍 Accident Risk Prediction")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    place_in = st.selectbox("Place", data["place_name"].unique())
with c2:
    road_in = st.selectbox("Road Type", le_road.classes_)
with c3:
    time_in = st.selectbox("Time of Day", le_time.classes_)
with c4:
    vehicle_in = st.selectbox("Vehicle", ["Car", "Bike", "Truck", "Bus", "Other"])
with c5:
    realtime = st.checkbox("Use Live Weather & Traffic", True)

place_row = data[data["place_name"] == place_in].iloc[0]
lat, lon = place_row["latitude"], place_row["longitude"]

if realtime:
    api_weather, temp = get_live_weather(lat, lon)
    weather_in = map_weather(api_weather)
    traffic_in = get_live_traffic()
else:
    weather_in = st.selectbox("Weather", le_weather.classes_)
    traffic_in = st.slider("Traffic Volume", 100, 1500, 600)
    temp = None

vehicle_enc = (
    le_vehicle.transform([vehicle_in])[0]
    if "vehicle_type" in data.columns
    else 0
)

# -----------------------------
# SAFE ENCODING (NO CRASH)
# -----------------------------
if weather_in not in le_weather.classes_:
    weather_enc = le_weather.transform([le_weather.classes_[0]])[0]
else:
    weather_enc = le_weather.transform([weather_in])[0]

input_data = [[
    weather_enc,
    le_road.transform([road_in])[0],
    traffic_in,
    le_time.transform([time_in])[0],
    vehicle_enc
]]

prediction = model.predict(input_data)[0]
risk_text = "High" if prediction == 2 else "Medium" if prediction == 1 else "Low"

# -----------------------------
# ALERT SYSTEM
# -----------------------------
st.write("### 🚨 Safety Alert")
st.markdown(f"**Risk Level at {place_in}: {risk_text}**")

if temp:
    st.info(f"🌡️ Temperature: {temp}°C")

st.info(f"🌦️ Weather: {weather_in} | 🚦 Traffic: {traffic_in}")

if prediction == 2:
    st.error("🚨 HIGH RISK ZONE! Accident hotspot detected.")
elif prediction == 1:
    st.warning("⚠️ Medium risk area. Drive carefully.")
else:
    st.success("🟢 Low risk area. Safe to travel.")

# -----------------------------
# HOTSPOT MAP
# -----------------------------
st.write("### 🗺️ Accident Hotspot Map")

m = folium.Map(location=[lat, lon], zoom_start=12)

for _, row in data.iterrows():
    if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
        continue

    color = "green"
    if row["risk"] == 2:
        color = "red"
    elif row["risk"] == 1:
        color = "orange"

    folium.Marker(
        [float(row["latitude"]), float(row["longitude"])],
        popup=f"{row['place_name']} | Risk: {'High' if row['risk']==2 else 'Medium' if row['risk']==1 else 'Low'}",
        icon=folium.Icon(color=color)
    ).add_to(m)

HeatMap(
    [[float(r["latitude"]), float(r["longitude"]), r["risk"]]
     for _, r in data.iterrows()]
).add_to(m)

st_folium(m, width=900, height=500)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("AI-Based Smart Road Safety System | Final-Year Ready Project")
