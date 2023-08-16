import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read the CSV with numpy
suffix = 'stable64'
filename = suffix + '/data.csv'
data = np.genfromtxt(filename, delimiter=',', skip_header=0, dtype=None, encoding='utf-8')

# Step 2: Extract the first line for the legend
legend_labels = data[0][-3:]

# Step 3: Extract the relevant columns for plotting
# Assuming you want to skip the header, hence [1:]
x_data = data[1:, 1].astype(float)  # Convert to float for plotting
y_data1 = data[1:, -3].astype(float)  # Convert to float for plotting
y_data2 = data[1:, -2].astype(float)  # Convert to float for plotting
y_data3 = data[1:, -1].astype(float)  # Convert to float for plotting

# Step 4: Plot the data using matplotlib
plt.figure(figsize=(10,6))
plt.plot(x_data, y_data1, label=legend_labels[0])
plt.plot(x_data, y_data2, label=legend_labels[1])
plt.plot(x_data, y_data3, label=legend_labels[2])

plt.xlabel("X Axis Label")  # Replace with your x-axis label
plt.ylabel("Y Axis Label")  # Replace with your y-axis label
plt.title("Your Title")  # Replace with your title
plt.legend()
plt.grid(True)
plt.savefig('test.png')
