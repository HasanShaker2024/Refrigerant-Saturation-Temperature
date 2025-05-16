import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Title
st.title("🌡️ Refrigerant Saturation Temperature Predictor")
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
        5.5, 6, 6.6, 7.1, 7.7, 8.3, 8.9, 9.5, 10.1, 10.8, 11.4, 12.1, 12.7, 13.4, 14.1, 14.8, 15.6, 16.3, 17.1, 17.8,
        18.6, 19.4, 20.2, 21, 21.9, 22.7, 23.6, 24.5, 25.4, 26.3, 27.3, 28.2, 29.2, 30.2, 31.2, 32.2, 33.2, 34.3, 35.4,
        36.5, 37.6, 38.7, 39.9, 41, 42.2, 43.4, 44.6, 45.9, 47.1, 48.4, 49.7, 51.1, 52.4, 53.8, 55.2, 56.6, 58, 59.5, 60.9,
        62.4, 63.9, 65.5, 67.1, 68.6, 70.3, 71.9, 73.5, 75.2, 76.9, 78.7, 80.4, 82.2, 84, 85.8, 87.7, 89.6, 91.5, 93.4,
        95.4, 97.4, 99.4, 101.4, 103.5, 105.6, 107.7, 109.9, 112.1, 114.3, 116.5, 118.8, 121.1, 123.4, 125.8, 128.2, 130.6,
        133, 135.5, 138, 140.6, 143.2, 145.8, 148.4, 151.1, 153.8, 156.5, 159.3, 162.1, 164.9, 167.8, 170.7, 173.7, 176.7,
        179.7, 182.7, 185.8, 188.9, 192.1, 195.3, 198.5, 201.8, 205.1, 208.4, 211.8, 215.2, 218.7, 222.2, 225.7, 229.3,
        232.9, 236.5, 240.2, 244, 247.8, 251.6, 255.4, 259.3, 263.3, 267.3, 271.3, 275.4, 279.5, 283.6, 287.9, 292.1,
        296.4, 300.7, 305.1, 309.5, 314, 318.6, 323.1, 327.7, 332.4, 337.1, 341.9, 346.7, 351.6, 356.5, 361.4, 366.4,
        371.5, 376.6, 381.8, 387, 392.3, 397.6, 403, 408.4, 413.9, 419.4, 425, 430.7, 436.4, 442.1, 447.9, 453.8, 459.8,
        465.8, 471.8, 477.9, 484.1, 490.3, 496.6, 503, 509.4, 515.9, 522.5, 529.1, 535.8, 542.5, 549.3, 556.2, 563.2,
        570.2, 577.3, 584.5, 591.7, 599, 606.4, 613.9
    ],
    "SatTemp": [
        -45, -44.4, -43.9, -43.3, -42.8, -42.2, -41.7, -41.1, -40.6, -40, -39.4, -38.9, -38.3, -37.8, -37.2, -36.7,
        -36.1, -35.6, -35, -34.4, -33.9, -33.3, -32.8, -32.2, -31.7, -31.1, -30.6, -30, -29.4, -28.9, -28.3, -27.8,
        -27.2, -26.7, -26.1, -25.6, -25, -24.4, -23.9, -23.3, -22.8, -22.2, -21.7, -21.1, -20.6, -20, -19.4, -18.9,
        -18.3, -17.8, -17.2, -16.7, -16.1, -15.6, -15, -14.4, -13.9, -13.3, -12.8, -12.2, -11.7, -11.1, -10.6, -10,
        -9.4, -8.9, -8.3, -7.8, -7.2, -6.7, -6.1, -5.6, -5, -4.4, -3.9, -3.3, -2.8, -2.2, -1.7, -1.1, -0.6, 0, 0.6, 1.1,
        1.7, 2.2, 2.8, 3.3, 3.9, 4.4, 5, 5.6, 6.1, 6.7, 7.2, 7.8, 8.3, 8.9, 9.4, 10, 10.6, 11.1, 11.7, 12.2, 12.8, 13.3,
        13.9, 14.4, 15, 15.6, 16.1, 16.7, 17.2, 17.8, 18.3, 18.9, 19.4, 20, 20.6, 21.1, 21.7, 22.2, 22.8, 23.3, 23.9,
        24.4, 25, 25.6, 26.1, 26.7, 27.2, 27.8, 28.3, 28.9, 29.4, 30, 30.6, 31.1, 31.7, 32.2, 32.8, 33.3, 33.9, 34.4,
        35, 35.6, 36.1, 36.7, 37.2, 37.8, 38.3, 38.9, 39.4, 40, 40.6, 41.1, 41.7, 42.2, 42.8, 43.3, 43.9, 44.4, 45,
        45.6, 46.1, 46.7, 47.2, 47.8, 48.3, 48.9, 49.4, 50, 50.6, 51.1, 51.7, 52.2, 52.8, 53.3, 53.9, 54.4, 55, 55.6,
        56.1, 56.7, 57.2, 57.8, 58.3, 58.9, 59.4, 60, 60.6, 61.1, 61.7, 62.2, 62.8, 63.3, 63.9, 64.4, 65, 65.6
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
