"""
Usage:
    plot_scalars.py <config_file>
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from docopt import docopt
from configparser import ConfigParser
from pathlib import Path

args = docopt(__doc__)
filename = Path(args['<config_file>'])

from read_config import ConfigEval
config = ConfigEval(filename)
locals().update(config.execute_locals())

# Step 1: Read the CSV with numpy
# suffix = 'stable128L2'
filename = suffix + '/data.csv'
data = np.genfromtxt(filename, delimiter=', ', skip_header=0, dtype=None, encoding='utf-8')

# Step 2: Extract the first line for the legend
legend_labels = data[0][-3:].tolist()
# print(legend_labels)
# sys.exit()

legend_labels[legend_labels.index('x_l2')] = r'$||\mathbf{u}||_2 + ||\mathbf{b}||_2$'
legend_labels[legend_labels.index('u_l2')] = r'$||\mathbf{u}||_2$'
legend_labels[legend_labels.index('b_l2')] = r'$||\mathbf{b}||_2$'

# Step 3: Extract the relevant columns for plotting
# Assuming you want to skip the header, hence [1:]
x_data = data[1:, 1].astype(float)  # Convert to float for plotting
y_data1 = data[1:, -3].astype(float)  # Convert to float for plotting
y_data2 = data[1:, -2].astype(float)  # Convert to float for plotting
y_data3 = data[1:, -1].astype(float)  # Convert to float for plotting

# Step 4: Plot the data using matplotlib
plt.figure(figsize=(4,4))
plt.scatter(x_data, y_data1, label=legend_labels[0])
plt.scatter(x_data, y_data2, label=legend_labels[1])
plt.scatter(x_data, y_data3, label=legend_labels[2])

plt.xlabel("time")  # Replace with your x-axis label
# plt.ylabel("Y Axis Label")  # Replace with your y-axis label
plt.title("MRI norms")  # Replace with your title
plt.legend()
plt.grid(True)
plt.savefig(suffix + '/norms.png')
