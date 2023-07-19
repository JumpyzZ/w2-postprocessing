import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error
from math import sqrt

# 定义文件路径
met_path = '/Users/zhu/Desktop/w2-postprocessing/met.csv'
tsr_path = '/Users/zhu/Desktop/w2-postprocessing/tsr_1_seg31.csv'

# 读取气象数据和模型数据
data = pd.read_csv(met_path, delimiter=',', skiprows=1)
model_data = pd.read_csv(tsr_path)
model_data['JDAY'] = model_data['JDAY'].astype(int)

# 合并数据
merged_data = pd.merge(data[['JDAY', 'TAIR']], model_data[['JDAY', 'T2(C)']], on='JDAY', how='inner')

# 重命名列名
merged_data.rename(columns={'TAIR': 'Observed_Temp', 'T2(C)': 'Model_Temp'}, inplace=True)

# 计算均方根误差（RMSE）、平均绝对误差（MAE）和最大误差（Max Error）
rmse = sqrt(mean_squared_error(merged_data['Observed_Temp'], merged_data['Model_Temp']))
mae = mean_absolute_error(merged_data['Observed_Temp'], merged_data['Model_Temp'])
max_err = max_error(merged_data['Observed_Temp'], merged_data['Model_Temp'])

# 绘制图表
plt.figure(figsize=(10, 6))

plt.plot(merged_data['JDAY'], merged_data['Observed_Temp'], label='Observed Temperature')
plt.plot(merged_data['JDAY'], merged_data['Model_Temp'], label='Model Predicted Temperature')

plt.xlabel('Day of the Year')
plt.ylabel('Temperature (C)')
plt.title('Observed vs Model Predicted Temperature')

# 在图表中添加文本信息，显示RMSE、MAE和最大误差
plt.text(0.02, 0.85, f'RMSE = {rmse:.2f}\nMAE = {mae:.2f}\nMax Error = {max_err:.2f}\n', transform=plt.gca().transAxes)

plt.legend()
plt.show()
