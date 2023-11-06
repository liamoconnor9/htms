"""
Usage:
    plot_slices.py <config_file>
"""

from docopt import docopt
from configparser import ConfigParser
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker
plt.ioff()
from dedalus.extras import plot_tools
import logging
import sys
logger = logging.getLogger(__name__)
from pathlib import Path

# args = docopt(__doc__)
# filename = Path(args['<config_file>'])

# from read_config import ConfigEval
# config = ConfigEval(filename)
# locals().update(config.execute_locals())


def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot settings
    normal_dir = 'y'
    scale = 2.5
    dpi = 100
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    savename_func = lambda write: 'mid_{:06}.png'.format(write)
    # Layout
    # if (round(ary) > 1):
    # if (False):
    #     nrows, ncols = 6, 1
    #     tasks = ['vx_mid'+normal_dir, 'vy_mid'+normal_dir, 'vz_mid'+normal_dir, 'bx_mid'+normal_dir, 'by_mid'+normal_dir, 'bz_mid'+normal_dir]
    # else:
    #     nrows, ncols = 3, 2
    #     tasks = ['vx_mid'+normal_dir, 'bx_mid'+normal_dir, 'vy_mid'+normal_dir, 'by_mid'+normal_dir, 'vz_mid'+normal_dir, 'bz_mid'+normal_dir]

    # image = plot_tools.Box(Lx, Lz)
    # pad = plot_tools.Frame(0.2, 0.2, 0.1, 0.1)
    # margin = plot_tools.Frame(0.3, 0.2, 0.1, 0.1)

    # # Create multifigure
    # mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
    # fig = mfig.figure
    # fig = plt.figure(figsize=(np.pi / 0.335, 2*np.pi))

    # Specify the dimensions for the two subplots
    # [left, bottom, width, height]
    # axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.3])  # Left subplot
    # axes2 = fig.add_axes([0.1, 0.5, 0.8, 0.3]) 
    # axs = [axes1, axes2]
    # Plot writes
    with h5py.File(filename, mode='r') as file:
        for index in range(start, start+count):
            fig, axs = plt.subplots(1, 3, sharey=True, figsize=(3.2*np.pi, 3*np.pi))
            # fig, axs = plt.subplots(1, 2)
            tasks = ['br', 'bth', 'bz']
            for n, task in enumerate(tasks):
                # Build subfigure axes
                # i, j = divmod(n, ncols)
                # axes = mfig.add_axes(i, j, [0, 0, 1, 1])
                # Call plotting helper (dset axes: [t, x, y, z])
                dset = file['tasks'][task][index, :, :][:, 0, :]
                z_hash = [key for key in file['scales'] if 'z_hash' in key][0]
                r_hash = [key for key in file['scales'] if 'r_hash' in key][0]
                zplt = file['scales'][z_hash]
                xplt = file['scales'][r_hash]

                pc = axs[n].pcolor(xplt, zplt, dset, cmap='seismic')
                plt.colorbar(pc, ax=axs[n], orientation='vertical')
                axs[n].set_title(task)
                if (n == 0):
                    axs[n].set_ylabel('z')
                axs[n].set_xlabel('r')
                axs[n].set_aspect('auto')
                # sys.exit()
                # image_axes = (1, 3)
                # data_slices = (index, slice(None), 0, slice(None))
                # plot_tools.plot_bot(dset, image_axes, data_slices, axes=axes, title=task, even_scale=True)

            # Add time title
            # title = title_func(file['scales/sim_time'][index])
            # title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
            fig.suptitle(r'$t=$' + str(file['scales/sim_time'][index]))
            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
            # fig.clear()
            plt.close(fig)

if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    # args = docopt(__doc__)

    # output_path = pathlib.Path(args['--output']).absolute()
    # dir = args['--dir']
    # suffix = args['--suffix']
    # filename = dir + suffix + '/' + args['--config']
    # config = ConfigParser()
    # config.read(str(filename))

    # global ar, ary, arz
    # Ly = eval(config.get('parameters','Ly'))
    # Lz = eval(config.get('parameters','Lz'))
    # Lx = eval(config.get('parameters','Lx'))

    # ary = Ly / Lx
    # arz = Lz / Lx 

    # Create output directory if needed
    output_path = Path('cyl/frames')
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()

    import glob
                
    post.visit_writes(glob.glob('cyl/slicepoints/*.h5'), main, output=output_path)