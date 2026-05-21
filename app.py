import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Loading the dataset
df = pd.read_csv("data/forestfires.csv")

# Creating target column
df["fire_risk"] = (df["area"] > 0).astype(int)

# Input features and target variable
X = df[[
    "FFMC",
    "DMC",
    "DC",
    "ISI",
    "temp",
    "RH",
    "wind",
    "rain"
]]

y = df["fire_risk"]


# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42
)


# Training the model (Random Forest Classifier)
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)


# Streamlit UI
st.title("🔥 AI Forest Fire Detection System")

st.write("Enter environemtal conditions below: ")



# User inputs
temp = st.number_input("Temperature (°C)")
RH = st.number_input("Relative Humidity (%)")
wind = st.number_input("Wind Speed (km/h)")
rain = st.number_input("Rainfall")

# Since people dont usually know the internal wildfire index values, we can set them to default values for testing.

# FFMC = st.number_input("FFMC  (Fine Fuel Moisture Code)")
# DMC = st.number_input("DMC (Duff Moisture Code)")
# DC = st.number_input("DC (Drought Code)")
# ISI = st.number_input("ISI (Initial Spread Index)")

st.write("This AI system predicts wildfire risk using environmental and wildfire-related indicators.")


# Internal wildfire index values for testing
FFMC = 85
DMC = 120
DC = 300
ISI = 10


# Prediction button
if st.button("Predict Fire Risk"):

    sample_data = pd.DataFrame(
        [[FFMC, DMC, DC, ISI, temp, RH, wind, rain]],
        columns=["FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]
    )


    # prediction = model.predict(sample_data)


    # if prediction[0] == 1:
    #     st.error("🔥 HIGH FIRE RISK")
    # else:
    #     st.success("✅ LOW FIRE RISK")


    prediction = model.predict(sample_data)

    probability = model.predict_proba(sample_data)

    fire_probability = probability[0][1] * 100

    st.write(f"🔥 Fire Risk Probability: {fire_probability:.2f}%")

    if fire_probability > 70:
        st.error("🔥 HIGH FIRE RISK")

    elif fire_probability > 40:
        st.warning("⚠️ MODERATE FIRE RISK")

    else:
        st.success("✅ LOW FIRE RISK")
        