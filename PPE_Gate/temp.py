import subprocess
import time
temps = []
duration = 300 # 10 phút
interval = 1 # đo mỗi 1 giây
for _ in range(duration // interval):
 result = subprocess.check_output(
 ["vcgencmd", "measure_temp"]
 ).decode()
 # temp=52.3'C -> 52.3
 temp = float(result.split('=')[1].split("'")[0])
 temps.append(temp)
 print(f"Current: {temp:.1f}°C")
 time.sleep(interval)
avg_temp = sum(temps) / len(temps)
max_temp = max(temps)
min_temp = min(temps)
print("\n===== Result =====")
print(f"Average Temp : {avg_temp:.2f}°C")
print(f"Maximum Temp : {max_temp:.2f}°C")
print(f"Minimum Temp : {min_temp:.2f}°C")
