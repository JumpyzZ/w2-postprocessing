import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, mean_squared_log_error
from math import sqrt

# Load the observed data
met_path = '/Users/zhu/Desktop/w2-postprocessing/met.csv'
data = pd.read_csv(met_path, delimiter=',', skiprows=1)

# Load the model output data
tsr_path = '/Users/zhu/Desktop/w2-postprocessing/tsr_1_seg31.csv'
model_data = pd.read_csv(tsr_path)
model_data['JDAY'] = model_data['JDAY'].astype(int)

# Merge the two datasets on "JDAY"
merged_data = pd.merge(data[['JDAY', 'TAIR']], model_data[['JDAY', 'T2(C)']], on='JDAY', how='inner')

# Rename the columns for clarity
merged_data.rename(columns={'TAIR': 'Observed_Temp', 'T2(C)': 'Model_Temp'}, inplace=True)

# Calculate the Root Mean Squared Error
rmse = sqrt(mean_squared_error(merged_data['Observed_Temp'], merged_data['Model_Temp']))
# Calculate the Mean Absolute Error
mae = mean_absolute_error(merged_data['Observed_Temp'], merged_data['Model_Temp'])

# Calculate the Max Error
max_err = max_error(merged_data['Observed_Temp'], merged_data['Model_Temp'])

# Create a new figure and set the size of the figure
plt.figure(figsize=(10, 6))

# Plot observed temperature
plt.plot(merged_data['JDAY'], merged_data['Observed_Temp'], label='Observed Temperature')

# Plot model predicted temperature
plt.plot(merged_data['JDAY'], merged_data['Model_Temp'], label='Model Predicted Temperature')

# Set the labels and title
plt.xlabel('Day of the Year')
plt.ylabel('Temperature (C)')
plt.title('Observed vs Model Predicted Temperature')

# Add error metrics to the plot
plt.text(0.02, 0.85, f'RMSE = {rmse:.2f}\nMAE = {mae:.2f}\nMax Error = {max_err:.2f}\n', transform=plt.gca().transAxes)

# Add a legend
plt.legend()

# Show the plot
plt.show()
