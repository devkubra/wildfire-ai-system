import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import requests


API_KEY = "e4b9a053b56f951c838aa7cf356b9e28"


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
city = st.text_input("Enter City: ")


# API URL. This builds a web request to the OpenWeatherMap API to get the current weather data for the specified city. 
url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

# Sending requests to API. This line sends a GET request to the OpenWeatherMap API using the constructed URL. The response from the API is stored in the variable 'response'
response = requests.get(url)


# Converting response into python dictionary.
data = response.json()


# This block is executed only after the user enter the city.
# Extracting weather values.
if city:

    temp = data["main"]["temp"]         # This line extracts the current temperature from the API response and stores it in the variable 'temp'.
    RH = data["main"]["humidity"]       # This line extracts the current humidity from the API response and stores it in the variable 'RH'.
    wind = data["wind"]["speed"]        # This line extracts the current wind speed from the API response and stores it in the variable 'wind'.


    rain = 0        # 0 is the default value for rain, which means no rain. If the API response contains information about rain, we can update this variable accordingly.

    if "rain" in data:
        rain = data["rain"].get("1h", 0)        # If it rains, this line extracts the rainfall in the last hour and updates the 'rain' variable. If there is no rain information, it remains 0.


    # Showing the extracted weather values to the user.
    st.write("Temperature (°C): ", temp)
    st.write("Relative Humidity (%): ", RH)
    st.write("Wind Speed (km/h): ", wind)
    st.write("Rainfall: ", rain)


# Manual inputs...no longer needed since we are getting real-time data from the API. 
# temp = st.number_input("Temperature (°C)")
# RH = st.number_input("Relative Humidity (%)")
# wind = st.number_input("Wind Speed (km/h)")
# rain = st.number_input("Rainfall")


st.write("This AI system predicts wildfire risk using environmental and wildfire-related indicators.")


# Since people dont usually know the internal wildfire index values, we can set them to default values for testing.
# Internal wildfire index values for testing
FFMC = 40
DMC = 30
DC = 50
ISI = 2


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

    if fire_probability < 35:
        st.error("✅ LOW FIRE RISK")

    elif fire_probability < 65:
        st.warning("⚠️ MODERATE FIRE RISK")

    else:
        st.success("🔥 HIGH FIRE RISK")
