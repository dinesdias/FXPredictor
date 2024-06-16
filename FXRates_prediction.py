import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor

# Function definitions for ML model, PPP, and IRP
def ml_model_prediction(data, features):
    # Feature engineering: Create lagged features for previous exchange rates (if needed)
    for i in range(1, 8):  # Example: lag features from 1 day to 7 days
        data[f'Price_lag{i}'] = data['Rate'].shift(-i)
    
    # Drop rows with NaN values resulting from lagging
    data.dropna(inplace=True)
    
    # Define features (lagged prices) and target variable
    X = data[features]
    y = data['Rate']
    
    # Initialize Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X, y)
    
    # Prepare data for predicting
    last_known_data = data.iloc[0][features].values.reshape(1, -1)
    forecasted_exchange_rate = model.predict(last_known_data)[0]
    
    return forecasted_exchange_rate

def purchasing_power_parity(inflation_rate_A, inflation_rate_B, spot_rate, duration_weeks):
    ppp_exchange_rate = spot_rate * ((1 + inflation_rate_A / 100)**(duration_weeks / 52)) / ((1 + inflation_rate_B / 100)**(duration_weeks / 52))
    return ppp_exchange_rate

def interest_rate_parity(interest_rate_A, interest_rate_B, spot_rate, duration_weeks):
    irp_exchange_rate = spot_rate * ((1 + interest_rate_A / 100)**(duration_weeks / 52)) / ((1 + interest_rate_B / 100)**(duration_weeks / 52))
    return irp_exchange_rate

def parse_weights(weights_str):
    weights = [float(w) for w in weights_str.split(',')]
    if len(weights) != 3:
        raise ValueError("Please enter exactly three weights.")
    if not np.isclose(sum(weights), 1.0):
        raise ValueError("The weights must sum to 1.")
    return weights

# Load data
file_path = 'D:/OneDrive/Software Dev/Python/Python Projects/ExchRates/AUD_LKR Historical Data_5yrs_14062024.csv'
data = pd.read_csv(file_path)

# Convert the date column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

# Request user input for frequency type
while True:
    frequency = input("Enter the frequency type (daily, weekly, monthly): ").strip().lower()
    if frequency in ['daily', 'weekly', 'monthly']:
        break
    else:
        print("Invalid frequency type. Please enter 'daily', 'weekly', or 'monthly'.")

# Determine the date range based on frequency
end_date = data['Date'].max()
if frequency == 'daily':
    start_date = end_date - timedelta(days=365)
elif frequency == 'weekly':
    start_date = end_date - timedelta(weeks=104)
elif frequency == 'monthly':
    start_date = end_date - timedelta(days=5*365)

# Filter data based on the date range
filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)].copy()

# Calculate support and resistance levels
def calculate_support_resistance(prices, window=20):
    support = prices.rolling(window=window).min()
    resistance = prices.rolling(window=window).max()
    return support, resistance

# Add support and resistance to filtered_data
filtered_data['Support'], filtered_data['Resistance'] = calculate_support_resistance(filtered_data['Rate'])

# Calculate RSI
def calculate_rsi(prices, window=14):
    deltas = prices.diff().dropna()
    gain = (deltas.where(deltas > 0, 0)).rolling(window=window).mean()
    loss = (-deltas.where(deltas < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

filtered_data['RSI'] = calculate_rsi(filtered_data['Rate'])

# Calculate trend line
z = np.polyfit(mdates.date2num(filtered_data['Date']), filtered_data['Rate'], 1)
p = np.poly1d(z)
filtered_data['Trend'] = p(mdates.date2num(filtered_data['Date']))

# Prompt user for inputs
while True:
    try:
        duration_weeks = int(input("Duration of the forecast (Weeks): "))
        if duration_weeks <= 0:
            print("Please enter a positive number of weeks.")
        elif duration_weeks > 104:
            print("Maximum duration allowed is 104 weeks.")
        else:
            break
    except ValueError:
        print("Invalid input. Please enter a valid number of weeks.")

inflation_rate_A = float(input("Inflation rate of Country A (2 for 2%): "))
inflation_rate_B = float(input("Inflation rate of Country B (3 for 3%): "))
interest_rate_A = float(input("Interest rate of Country A (1 for 1%): "))
interest_rate_B = float(input("Interest rate of Country B (1.5 for 1.5%): "))

while True:
    try:
        weights_str = input("Weights for ML model, PPP, and IRP (e.g:0.4,0.3,0.3): ")
        weights = parse_weights(weights_str)
        break
    except ValueError as e:
        print(e)

# Latest spot rate from historical data
latest_spot_rate = data['Rate'].iloc[0]

# Calculate exchange rate according to ML model
ml_model_rate = ml_model_prediction(filtered_data, ['Price_lag1', 'Price_lag2', 'Price_lag3', 'Price_lag4', 'Price_lag5', 'Price_lag6', 'Price_lag7'])

# Calculate exchange rate according to Purchasing Power Parity (PPP)
ppp_rate = purchasing_power_parity(inflation_rate_A, inflation_rate_B, latest_spot_rate, duration_weeks)

# Calculate exchange rate according to Interest Rate Parity (IRP)
irp_rate = interest_rate_parity(interest_rate_A, interest_rate_B, latest_spot_rate, duration_weeks)

# Calculate weighted average exchange rate based on the inputs
average_exchange_rate = weights[0] * ml_model_rate + weights[1] * ppp_rate + weights[2] * irp_rate

# Round all exchange rates to 2 decimals
average_exchange_rate_rounded = round(average_exchange_rate, 2)
ml_model_rate_rounded = round(ml_model_rate, 2)
ppp_rate_rounded = round(ppp_rate, 2)
irp_rate_rounded = round(irp_rate, 2)
latest_spot_rate_rounded = round(latest_spot_rate, 2)

# Plotting with GridSpec to adjust subplot sizes
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 1)

# Main plot
ax1 = fig.add_subplot(gs[:2, :])
ax1.plot(filtered_data['Date'], filtered_data['Rate'], label='Exchange Rate', color='blue')
ax1.plot(filtered_data['Date'], filtered_data['Support'], label='Support Level', color='green')
ax1.plot(filtered_data['Date'], filtered_data['Resistance'], label='Resistance Level', color='red')
ax1.plot(filtered_data['Date'], filtered_data['Trend'], label='Trend Line', color='orange', linestyle='--')
ax1.axhline(y=ml_model_rate_rounded, color='cyan', linestyle='-.', label=f'ML Model Rate: {ml_model_rate_rounded}')
ax1.axhline(y=ppp_rate_rounded, color='magenta', linestyle='--', label=f'PPP Rate: {ppp_rate_rounded}')
ax1.axhline(y=irp_rate_rounded, color='yellow', linestyle='-', label=f'IRP Rate: {irp_rate_rounded}')
ax1.axhline(y=latest_spot_rate_rounded, color='black', linestyle='-', label=f'Latest Spot Rate: {latest_spot_rate_rounded}')
ax1.axhline(y=average_exchange_rate_rounded, color='lime', linestyle='-', label=f'Average Exchange Rate: {average_exchange_rate_rounded}')
ax1.set_ylabel('Exchange Rate')
ax1.legend()

# RSI plot
ax2 = fig.add_subplot(gs[2, :], sharex=ax1)
ax2.plot(filtered_data['Date'], filtered_data['RSI'], label='RSI', color='purple', linestyle='-.')
ax2.set_ylabel('RSI')
ax2.set_xlabel('Date')

# Annotations for rates
ax1.text(filtered_data['Date'].iloc[0], ml_model_rate_rounded, f'ML Model Rate: {ml_model_rate_rounded}', fontsize=8, va='bottom', ha='left', backgroundcolor='white')
ax1.text(filtered_data['Date'].iloc[0], ppp_rate_rounded, f'PPP Rate: {ppp_rate_rounded}', fontsize=8, va='bottom', ha='left', backgroundcolor='white')
ax1.text(filtered_data['Date'].iloc[0], irp_rate_rounded, f'IRP Rate: {irp_rate_rounded}', fontsize=8, va='bottom', ha='left', backgroundcolor='white')
ax1.text(filtered_data['Date'].iloc[0], ml_model_rate_rounded, f'ML Model Rate: {ml_model_rate_rounded}', fontsize=8, va='bottom', ha='left', backgroundcolor='white')
ax1.text(filtered_data['Date'].iloc[0], ppp_rate_rounded, f'PPP Rate: {ppp_rate_rounded}', fontsize=8, va='bottom', ha='left', backgroundcolor='white')
ax1.text(filtered_data['Date'].iloc[0], irp_rate_rounded, f'IRP Rate: {irp_rate_rounded}', fontsize=8, va='bottom', ha='left', backgroundcolor='white')
ax1.text(filtered_data['Date'].iloc[0], latest_spot_rate_rounded, f'Latest Spot Rate: {latest_spot_rate_rounded}', fontsize=8, va='bottom', ha='left', backgroundcolor='white')
ax1.text(filtered_data['Date'].iloc[0], average_exchange_rate_rounded, f'Average Exchange Rate: {average_exchange_rate_rounded}', fontsize=8, va='bottom', ha='left', backgroundcolor='white')

# RSI plot
ax2.plot(filtered_data['Date'], filtered_data['RSI'], label='RSI', color='purple', linestyle='-.')
ax2.set_ylabel('RSI')
ax2.set_xlabel('Date')

plt.suptitle('AUD to LKR Exchange Rate Analysis')
plt.grid(True)
plt.show()

# Output the results
print(f"Forecasted Exchange Rate (Weighted Average): {average_exchange_rate_rounded}")
print(f"Exchange Rate Forecast by ML Model: {ml_model_rate_rounded}")
print(f"Exchange Rate Forecast by PPP Calculation: {ppp_rate_rounded}")
print(f"Exchange Rate Forecast by IRP Calculation: {irp_rate_rounded}")
print(f"Latest Spot Exchange Radailyte: {latest_spot_rate_rounded}")