"""Created on Thu Jun  4 17:33:58 2020.

@author: guepin

Visualize data from ZHAireS simulations.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import efield_denoising.hdf5lib.hdf5fileinout as hdf5io
import efield_denoising.hdf5lib.ComputePeak2PeakOnHDF5 as ComputeP2P
import glob

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def vis_events(PATH_data, lenefield=1100, num_event=5, num_trace_vis=5):
    """Visualize map event and given number of traces from sim directory.

    PATH_data: simulation folder
    lenefield: fixed length chosen for traces
    num_event_vis: number of events to visualize
    num_trace_vis: number of events to visualize
    """
    # BROWSE SIMULATION FILES (ZHAIRES)
    list_f = glob.glob(PATH_data+'*')
    print('Number of files = %i' % (len(list_f)))

    # All antenna positions
    antenna_all = np.loadtxt(PATH_data+'GP300proposedLayout.dat',
                             usecols=(2, 3, 4))

    for k in range(num_event):
        index = k
        inputfilename = glob.glob(list_f[index] + '/*.hdf5')[0]

        RunInfo = hdf5io.GetRunInfo(inputfilename)
        EventName = hdf5io.GetEventName(RunInfo, 0)
        AntennaInfo = hdf5io.GetAntennaInfo(inputfilename, EventName)
        nantennas = hdf5io.GetNumberOfAntennas(AntennaInfo)
        print("\nNumber of antennas: %i" % nantennas)
        primary = RunInfo['Primary'][0]
        print("Primary particle: %s" % primary)
        energy = RunInfo['Energy'][0]
        print("Energy: %.2f EeV" % energy)
        zenith = 180.-hdf5io.GetEventZenith(RunInfo, 0)
        print("Zenith: %.2f" % zenith)
        azimuth = hdf5io.GetEventAzimuth(RunInfo, 0)-180.
        print("Azimuth: %.2f" % azimuth)

        postab = np.zeros(shape=(nantennas, 3))
        efield_arr = np.zeros(shape=(len(antenna_all), lenefield, 3))
        for i in range(nantennas):
            postab[i, 0] = hdf5io.GetAntennaPosition(AntennaInfo, i)[0]
            postab[i, 1] = hdf5io.GetAntennaPosition(AntennaInfo, i)[1]
            postab[i, 2] = hdf5io.GetAntennaPosition(AntennaInfo, i)[2]

            AntennaID = hdf5io.GetAntennaID(AntennaInfo, i)
            efield_loc = hdf5io.GetAntennaEfield(
                inputfilename, EventName, AntennaID)
            # num_ant = int(AntennaInfo[i][0][1:])
            if len(AntennaInfo[i]['ID']) < 5:
                num_ant = int(AntennaInfo[i]['ID'][1:])
            else:
                num_ant = int(AntennaInfo[i]['ID'][11:])
            efield_arr[num_ant, :, 0] = efield_loc[0:lenefield, 1]
            efield_arr[num_ant, :, 1] = efield_loc[0:lenefield, 2]
            efield_arr[num_ant, :, 2] = efield_loc[0:lenefield, 3]

        # COMPUTE TOTAL PEAK TO PEAK AMPLITUDE
        p2pE = ComputeP2P.get_p2p_hdf5(inputfilename, antennamax='All',
                                       antennamin=0, usetrace='efield')
        p2pE_tot = p2pE[3, :]
        max_data_test = np.max(p2pE_tot)

        antenna_first_num = int(AntennaInfo[0][0][1:])
        diff_x = antenna_all[antenna_first_num, 0]-postab[0, 0]
        diff_y = antenna_all[antenna_first_num, 1]-postab[0, 1]

        PATH_fig = PATH_data+'Figures/'

        # =========================================================================
        # VISUALIZE EVENT ON ANTENNA ARRAY

        plt.figure()
        ax = plt.gca()

        plt.scatter(antenna_all[:, 0]/1e3, antenna_all[:, 1]/1e3, 30,
                    color='grey')

        plt.scatter((postab[:, 0]+diff_x)/1e3, (postab[:, 1]+diff_y)/1e3, 30,
                    c=p2pE_tot/max_data_test,
                    cmap=cm.viridis, vmin=0.0, vmax=0.6)

        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(labelsize=14)

        # ax.set_xlim([-10, 10])
        # ax.set_ylim([-10, 10])
        plt.subplots_adjust(left=0.14)
        ax.set_xlabel(r"$X\,{\rm\,(km)}$", fontsize=14)
        ax.set_ylabel(r"$Y\,{\rm\,(km)}$", fontsize=14)
        ax.axis('equal')
        plt.savefig(PATH_fig+'GP300_footprint_example'+'_'+str(index)+'.pdf')
        # plt.show()

        # =========================================================================
        # VISUALIZE TRACES

        # Find antennas with maximum power
        efield2_arr = np.sum(efield_arr**2, axis=2)
        efield2_arr_max = np.max(efield2_arr, axis=1)
        efield2_ind = np.argsort(efield2_arr_max)
        efield_ord = efield_arr[efield2_ind[::-1], :, :]

        # Visualize one component of the E-field for those traces
        plt.figure()
        ax = plt.gca()
        for i in range(num_trace_vis):
            fig, = plt.plot(efield_ord[i, :, 0], ls='--')
            plt.plot(efield_ord[i, :, 1], ls='-', color=fig.get_color())
            plt.plot(efield_ord[i, :, 2], ls=':', color=fig.get_color())
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(labelsize=14)
        # ax.set_xlim([-10, 10])
        # ax.set_ylim([-10, 10])
        plt.subplots_adjust(left=0.14)
        ax.set_xlabel(r"Time bins", fontsize=14)
        ax.set_ylabel(r"Amplitude E-field", fontsize=14)
        plt.savefig(PATH_fig+'GP300_traces_example'+'_'+str(index)+'.pdf')
        plt.show()


if __name__ == "__main__":
    vis_events(sys.argv[1], int(sys.argv[2]),
               int(sys.argv[3]), int(sys.argv[4]))
