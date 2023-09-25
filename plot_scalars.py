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

components = True

colors = { 'x_l2' : 'black', 'u_l2' : 'lime', 'b_l2' : 'blue'}
labels = { 'x_l2' : r'$||\mathbf{u}||^2 + ||\mathbf{b}||^2$', 
          'u_l2' : r'$||\mathbf{u}||^2$', 
          'b_l2' : r'$||\mathbf{b}||^2$'}
 
# iterate over files in
# that directory
labeled=False
time_lst = []
x_lst = []
u_lst = []
b_lst = []

bz_lst = []
bx_lst = []
by_lst = []

uz_lst = []
ux_lst = []
uy_lst = []


for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f) and str(f)[-3:] == '.h5':

        # Load the data from the .h5 file
        with h5py.File(f, 'r') as file:
            # Assuming 'data' is the name of the dataset in the .h5 file
            # data = file['tasks'].keys()
            times = file['scales']['sim_time']
            time_lst += file['scales']['sim_time'][:].tolist()
            x_lst += file['tasks']['x_l2'][:].tolist()
            u_lst += file['tasks']['u_l2'][:].tolist()
            b_lst += file['tasks']['b_l2'][:].tolist()

            if components:
                bz_lst += file['tasks']['bz_l2'][:].tolist()
                bx_lst += file['tasks']['bx_l2'][:].tolist()
                by_lst += file['tasks']['by_l2'][:].tolist()

                uz_lst += file['tasks']['uz_l2'][:].tolist()
                ux_lst += file['tasks']['ux_l2'][:].tolist()
                uy_lst += file['tasks']['uy_l2'][:].tolist()

            # sys.exit()

            # for key in file['tasks'].keys():
            #     # print(np.shape(times))
            #     # print(np.shape(file['tasks'][key]))
            #     # print(' ')
            #     if (not labeled):
            #         plt.scatter(times, np.log(file['tasks'][key]), label=labels[key], color=colors[key])
            #     else:
            #         plt.scatter(times, np.log(file['tasks'][key]), color=colors[key])
            # # print(np.shape(data))
            # labeled=True

        # Extract x and y values
        # x_values = data[:, 0]
        # y_values = data[:, 1]
x_lst = np.array(x_lst).ravel().tolist()
u_lst = np.array(u_lst).ravel().tolist()
b_lst = np.array(b_lst).ravel().tolist()
if components:
    bz_lst = np.array(bz_lst).ravel().tolist()
    bx_lst = np.array(bx_lst).ravel().tolist()
    by_lst = np.array(by_lst).ravel().tolist()

    uz_lst = np.array(uz_lst).ravel().tolist()
    ux_lst = np.array(ux_lst).ravel().tolist()
    uy_lst = np.array(uy_lst).ravel().tolist()

    zipped_lists = zip(time_lst, x_lst, u_lst, b_lst, bz_lst, bx_lst, by_lst, uz_lst, ux_lst, uy_lst)
    sorted_lists = sorted(zipped_lists)
    time_srtd, x_srtd, u_srtd, b_srtd, bz_srtd, bx_srtd, by_srtd, uz_srtd, ux_srtd, uy_srtd = zip(*sorted_lists)

else:
    zipped_lists = zip(time_lst, x_lst, u_lst, b_lst)
    sorted_lists = sorted(zipped_lists)
    time_srtd, x_srtd, u_srtd, b_srtd = zip(*sorted_lists)

plt.plot(time_srtd, x_srtd, label = labels['x_l2'], color=colors['x_l2'], linewidth=4, linestyle='solid')
plt.plot(time_srtd, u_srtd, label = labels['u_l2'], color=colors['u_l2'], linewidth=4, linestyle='dashed')
plt.plot(time_srtd, b_srtd, label = labels['b_l2'], color=colors['b_l2'], linewidth=4, linestyle='dotted')

plt.xlabel("time")  # Replace with your x-axis label
# plt.ylabel("Y Axis Label")  # Replace with your y-axis label
plt.title("MRI energies")  # Replace with your title
plt.legend()
plt.yscale('log')

# plt.grid(True)
outpath = suffix + '/norms.png'
print("output path: " + outpath)
plt.savefig(outpath)
plt.close()
# sys.exit()

colors = { 'uz' : 'magenta', 'ux' : 'lime', 'uy' : 'blue'}
labels = { 'uz' : r'$||\mathbf{u}\cdot\mathbf{\hat{e}_z}||^2$', 'ux' : r'$||\mathbf{u}\cdot\mathbf{\hat{e}_x}||^2$', 'uy' : r'$||\mathbf{u}\cdot\mathbf{\hat{e}_y}||^2$'}


plt.plot(time_srtd, uz_srtd, label=labels['uz'], color=colors['uz'], linewidth=4)
plt.plot(time_srtd, ux_srtd, label=labels['ux'], color=colors['ux'], linewidth=4)
plt.plot(time_srtd, uy_srtd, label=labels['uy'], color=colors['uy'], linewidth=4)

plt.xlabel("time")  # Replace with your x-axis label
plt.title("velocity components")  # Replace with your title
plt.legend()
plt.yscale('log')
outpath = suffix + '/ucomps.png'
print("output path: " + outpath)
plt.savefig(outpath)
plt.close()

colors = { 'bz' : 'magenta', 'bx' : 'lime', 'by' : 'blue'}
labels = { 'bz' : r'$||\mathbf{b}\cdot\mathbf{\hat{e}_z}||^2$', 'bx' : r'$||\mathbf{b}\cdot\mathbf{\hat{e}_x}||^2$', 'by' : r'$||\mathbf{b}\cdot\mathbf{\hat{e}_y}||^2$'}


plt.plot(time_srtd, bz_srtd, label=labels['bz'], color=colors['bz'], linewidth=4)
plt.plot(time_srtd, bx_srtd, label=labels['bx'], color=colors['bx'], linewidth=4)
plt.plot(time_srtd, by_srtd, label=labels['by'], color=colors['by'], linewidth=4)

plt.xlabel("time")  # Replace with your x-axis label
plt.title("magnetic flux components")  # Replace with your title
plt.legend()
plt.yscale('log')
outpath = suffix + '/bcomps.png'
print("output path: " + outpath)
plt.savefig(outpath)
