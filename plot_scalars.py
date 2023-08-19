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
import os
import h5py

args = docopt(__doc__)
filename = Path(args['<config_file>'])

from read_config import ConfigEval
config = ConfigEval(filename)
locals().update(config.execute_locals())

# import required module
# assign directory
plt.figure(figsize=(4,4))
directory=suffix + '/scalars'
 
colors = { 'x_l2' : 'black', 'u_l2' : 'lime', 'b_l2' : 'blue'}
labels = { 'x_l2' : r'$||\mathbf{u}||_2 + ||\mathbf{b}||_2$', 
          'u_l2' : r'$||\mathbf{u}||_2$', 
          'b_l2' : r'$||\mathbf{b}||_2$'}
 
# iterate over files in
# that directory
labeled=False
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f) and str(f)[-3:] == '.h5':

        # Load the data from the .h5 file
        with h5py.File(f, 'r') as file:
            # Assuming 'data' is the name of the dataset in the .h5 file
            # data = file['tasks'].keys()
            times = file['scales']['sim_time']
            for key in file['tasks'].keys():
                # print(np.shape(times))
                # print(np.shape(file['tasks'][key]))
                # print(' ')
                if (not labeled):
                    plt.scatter(times, file['tasks'][key], label=labels[key], color=colors[key])
                else:
                    plt.scatter(times, file['tasks'][key], color=colors[key])
            # print(np.shape(data))
            labeled=True

        # Extract x and y values
        # x_values = data[:, 0]
        # y_values = data[:, 1]

# sys.exit()

# legend_labels[legend_labels.index('x_l2')] = r'$||\mathbf{u}||_2 + ||\mathbf{b}||_2$'
# legend_labels[legend_labels.index('u_l2')] = r'$||\mathbf{u}||_2$'
# legend_labels[legend_labels.index('b_l2')] = r'$||\mathbf{b}||_2$'

# # Step 3: Extract the relevant columns for plotting
# # Assuming you want to skip the header, hence [1:]
# x_data = data[1:, 1].astype(float)  # Convert to float for plotting
# y_data1 = data[1:, -3].astype(float)  # Convert to float for plotting
# y_data2 = data[1:, -2].astype(float)  # Convert to float for plotting
# y_data3 = data[1:, -1].astype(float)  # Convert to float for plotting

# Step 4: Plot the data using matplotlib
# plt.scatter(x_data, y_data1, label=legend_labels[0])
# plt.scatter(x_data, y_data2, label=legend_labels[1])
# plt.scatter(x_data, y_data3, label=legend_labels[2])

plt.xlabel("time")  # Replace with your x-axis label
# plt.ylabel("Y Axis Label")  # Replace with your y-axis label
plt.title("MRI norms")  # Replace with your title
plt.legend()
# plt.grid(True)
plt.savefig(suffix + '/norms.png')
