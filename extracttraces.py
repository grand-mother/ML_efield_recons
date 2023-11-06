"""Created on Thu Jun  4 17:33:58 2020.

@author: guepin

Visualize data from ZHAireS simulations.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import hdf5fileinout as hdf5io
import ComputePeak2PeakOnHDF5 as ComputeP2P
import glob

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# =============================================================================
# FUNCTIONS


def save_data(PATH_data, efi):
    """Save traces, reshaping required."""
    # reshaping the array from 3D
    # matrice to 2D matrice.
    efi_reshaped = efi.reshape(efi.shape[0], -1)
    np.savetxt(PATH_data+'data.csv', efi_reshaped)


def read_data(filename, shape2):
    """Load traces, reshaping required."""
    loaded_arr = np.loadtxt(filename)
    # This loadedArr is a 2D array, therefore
    # we need to convert it to the original
    # array shape.reshaping to get original
    # matrice with original shape.
    load_original_arr = loaded_arr.reshape(
        loaded_arr.shape[0], loaded_arr.shape[1] // shape2, shape2)
    return load_original_arr


def extract_efield(inputfilename, lenantenna, lenefield):
    """Extract E-field data from simulations."""
    efield_arr = np.zeros(shape=(lenantenna, lenefield, 3))

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

    for i in range(nantennas):
        AntennaID = hdf5io.GetAntennaID(AntennaInfo, i)
        efield_loc = hdf5io.GetAntennaEfield(inputfilename, EventName,
                                             AntennaID)
        if len(AntennaInfo[i]['ID']) < 5:
            num_ant = int(AntennaInfo[i]['ID'][1:])
        else:
            num_ant = int(AntennaInfo[i]['ID'][11:])
        efield_arr[num_ant, :, 0] = efield_loc[0:lenefield, 1]
        efield_arr[num_ant, :, 1] = efield_loc[0:lenefield, 2]
        efield_arr[num_ant, :, 2] = efield_loc[0:lenefield, 3]

    return efield_arr


def create_efield_file(PATH_loc, nfiles):
    """Produce file with traces."""
    PATH_data = PATH_loc+'GP300Outbox/'
    # PATH_data = PATH_loc+'TheGP300Outbox/'

    # =============================================================================
    # PARAMETERS

    progenitor = 'Proton'
    zenVal = '_'+str(74.8)  # 63.0, 74.8, 81.3, 85.0, 87.1
    eneVal = '_'+str(1.58)  # ?

    # Fixed length chosen for traces
    lenefield = 1100

    # =============================================================================
    # BROWSE SIMULATION FILES (ZHAIRES)

    list_f = glob.glob(PATH_data+'*')
    # list_f = glob.glob(PATH_data+'*'+progenitor+'*')
    # list_f = glob.glob(PATH_data+'*'+progenitor+'*'+zenVal+'*')
    print('Number of files = %i' % (len(list_f)))

    # All antenna positions
    antenna_all = np.loadtxt(PATH_data+'GP300proposedLayout.dat',
                             usecols=(2, 3, 4))

    # =============================================================================
    # CREATE A DATABANK OF E-FIELD TRACES
    # BROWSE EVENTS AND EXTRACT IMPORTANT INFORMATION

    efield_tot = np.empty([0, lenefield, 3])

    for k in range(nfiles):
        index = k
        inputfilename = glob.glob(list_f[index] + '/*.hdf5')[0]
        efield_arr = extract_efield(inputfilename, len(antenna_all), lenefield)

        # Find antennas with maximum power
        efield2_arr = np.sum(efield_arr**2, axis=2)
        efield2_arr_max = np.max(efield2_arr, axis=1)
        efield2_ind = np.argsort(efield2_arr_max)
        efield_ord = efield_arr[efield2_ind[::-1], :, :]

        # Fill efield array with selected traces
        N_ant_save = 10
        efield_tot = np.append(efield_tot,
                               efield_ord[0:N_ant_save, :, :],
                               axis=0)

    # =============================================================================
    # SAVE TO FILE

    save_data(PATH_data, efield_tot)


if __name__ == "__main__":
    create_efield_file(sys.argv[1], int(sys.argv[2]))

# =============================================================================
# HOW TO LOAD DATA

# original = read_data(PATH_data+'data.csv', efield_tot.shape[2])

# # check the shapes:
# print("shape of arr: ", efield_tot.shape)
# print("shape of load_original_arr: ", original.shape)

# # check if both arrays are same or not
# if (original == efield_tot).all():
#     print("Yes, both the arrays are same")
# else:
#     print("No, both the arrays are not same")
