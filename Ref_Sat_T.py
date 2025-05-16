import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Title
st.title("🌡️ R32 Saturation Temperature Predictor")
st.markdown("Developed by [Hasan Samir Hasan]")
# Select Refrigerant Type
refrigerant = st.selectbox("Select Refrigerant Type", ["R32", "R410A"])
st.markdown("Enter a pressure (PSIG) to predict the saturation temperature (°C) using polynomial regression.")

# --- 1. Built-in R32 Pressure-Temperature Data
R32_data = {
    "Pressure": [
        11, 14.4, 18.2, 22.3, 26.8, 31.7, 37.1, 42.9, 49.3, 56.1, 63.5, 71.5, 80, 89.2,
        99.1, 109.7, 121, 133, 145.9, 159.5, 174.1, 189.5, 205.8, 223.2, 241.5, 260.9,
        281.3, 302.9, 325.7, 349.6, 374.9, 401.4, 429.3, 458.6, 489.4, 521.8, 555.7, 591.4, 628.8
    ],
    "SatTemp": [
        -40, -37.2, -34.4, -31.7, -28.9, -26.1, -23.3, -20.6, -17.8, -15, -12.2, -9.4, -6.7,
        -3.9, -1.1, 1.7, 4.4, 7.2, 10, 12.8, 15.6, 18.3, 21.1, 23.9, 26.7, 29.4, 32.2, 35,
        37.8, 40.6, 43.3, 46.1, 48.9, 51.7, 54.4, 57.2, 60, 62.8, 65.6
    ]
}

R410_data = {
    "Pressure": [
        5.5, 6, 6.6, 7.1, 7.7, 8.3, 8.9, 9.5, 10.1, 10.8, 11.4, 12.1, 12.7, 13.4, 14.1, 14.8, 15.6,
        16.3, 17.1, 17.8, 18.6, 19.4, 20.2, 21, 21.9, 22.7, 23.6, 24.5, 25.4, 26.3, 27.3, 28.2, 29.2,
        30.2, 31.2, 32.2, 33.2, 34.3, 35.4, 36.5, 37.6, 38.7, 39.9, 41, 42.2, 43.4, 44.6, 45.9, 47.1,
        48.4, 49.7, 51.1, 52.4, 53.8, 55.2, 56.6, 58, 59.5, 60.9, 62.4
    ],
    "SatTemp": [
        -45, -44.4, -43.9, -43.3, -42.8, -42.2, -41.7, -41.1, -40.6, -40, -39.4, -38.9, -38.3, -37.8,
        -37.2, -36.7, -36.1, -35.6, -35, -34.4, -33.9, -33.3, -32.8, -32.2, -31.7, -31.1, -30.6, -30,
        -29.4, -28.9, -28.3, -27.8, -27.2, -26.7, -26.1, -25.6, -25, -24.4, -23.9, -23.3, -22.8, -22.2,
        -21.7, -21.1, -20.6, -20, -19.4, -18.9, -18.3, -17.8, -17.2, -16.7, -16.1, -15.6, -15, -14.4,
        -13.9, -13.3, -12.8, -12.2, -11.7
    ]
}

#Select the appropriate data based on refrigerant type
if refrigerant == "R32":
    data = R32_data
else:
    data = R410_data
df = pd.DataFrame(data)



# --- 2. Train a Polynomial Regression Model
X = df[['Pressure']]
y = df['SatTemp']
degree = 4  # You can try 3 or 5 if desired
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)

# --- 3. User Input
pressure_input = st.number_input("Enter Pressure (PSIG):", min_value=0.0, value=628.0, step=1.0)

# --- 4. Predict
input_poly = poly.transform(np.array([[pressure_input]]))
predicted_temp = model.predict(input_poly)[0]
st.success(f"Predicted Saturation Temperature: {predicted_temp:.2f} °C")

# --- 5. Optional: Show Curve
if st.checkbox("📈 Show Curve"):
    import matplotlib.pyplot as plt

    X_range = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
    y_range = model.predict(poly.transform(X_range))

    fig, ax = plt.subplots()
    ax.scatter(X, y, color='blue', label='Data')
    ax.plot(X_range, y_range, color='red', label='Polynomial Fit')
    ax.set_xlabel('Pressure (PSIG)')
    ax.set_ylabel('Saturation Temp (°C)')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
