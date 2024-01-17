"""Created on Thu Jun  4 17:33:58 2020.

@author: guepin

Visualize data from ZHAireS simulations.
"""

import sys
import glob
import numpy as np
import uproot
import matplotlib.pyplot as plt
import efield_denoising.hdf5lib.hdf5fileinout as hdf5io
from efield_denoising.hdf5lib.mod_fun import filters

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# =============================================================================
# FUNCTIONS


def save_data(PATH_data, name, efi):
    """Save traces, reshaping required."""
    # reshaping the array from 3D
    # matrice to 2D matrice.
    efi_reshaped = efi.reshape(efi.shape[0], -1)
    np.savetxt(PATH_data+name+'.csv', efi_reshaped)


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
    """Extract E-field data from simulations with hdf5 format."""
    efield_arr = np.zeros(shape=(lenantenna, lenefield, 4))

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

        efield_arr[num_ant, :, 0] = efield_loc[0:lenefield, 0]
        efield_arr[num_ant, :, 1] = efield_loc[0:lenefield, 1]
        efield_arr[num_ant, :, 2] = efield_loc[0:lenefield, 2]
        efield_arr[num_ant, :, 3] = efield_loc[0:lenefield, 3]

    return efield_arr


def create_efield_file(PATH_data, lenefield, nfiles, nantsave):
    """Produce file with traces.

    PATH_data: simulation folder
    lenefield: fixed length chosen for traces (ex. 900 or 1100)
    nfiles: number of events to process
    nantsave: number of traces to save per event
    """
    # =============================================================================
    # BROWSE SIMULATION FILES (ZHAIRES)

    list_f = glob.glob(PATH_data+'*')
    print('Number of files = %i' % (len(list_f)))

    # All antenna positions
    antenna_all = np.loadtxt(PATH_data+'GP300proposedLayout.dat',
                             usecols=(2, 3, 4))

    # =============================================================================
    # CREATE A DATABANK OF E-FIELD TRACES
    # BROWSE EVENTS AND EXTRACT IMPORTANT INFORMATION

    efield_tot = np.empty([0, lenefield, 3])
    efield_filtered_tot = np.empty([0, lenefield, 3])

    for k in range(nfiles):
        index = k
        inputfilename = glob.glob(list_f[index] + '/*.hdf5')[0]
        efield_arr = extract_efield(inputfilename, len(antenna_all), lenefield)

        # Find antennas with maximum power
        efield2_arr = np.sum(efield_arr[:, :, 1:4]**2, axis=2)
        efield2_arr_max = np.max(efield2_arr, axis=1)
        efield2_ind = np.argsort(efield2_arr_max)
        efield_ord = efield_arr[efield2_ind[::-1], :, :]

        # Create filtered traces
        efield_fil = np.zeros(shape=(nantsave, lenefield, 4))
        for i in range(nantsave):
            efield_ord[i, :, 0] *= 1e-9  # ns to s change required
            efield_fil[i, :, :] = filters(efield_ord[i, :, :], 50.e6, 200.e6).T

        # Fill efield array with selected traces
        efield_tot = np.append(efield_tot,
                               efield_ord[0:nantsave, :, 1:4],
                               axis=0)
        efield_filtered_tot = np.append(efield_filtered_tot,
                                        efield_fil[0:nantsave, :, 1:4],
                                        axis=0)

    # =============================================================================
    # SAVE TO FILE

    save_data(PATH_data, "efield_traces", efield_tot)
    save_data(PATH_data, "efield_filtered_traces", efield_filtered_tot)


def extract_tra(inputfilename, quant):
    """Extract E-field data from simulations with root format."""
    # file = uproot.open(PATH_LIB+SIM_NAME+"/gr_"+SIM_NAME+".RawRoot")
    file = uproot.open(inputfilename)
    if quant == 'efield':
        tree = file['tefield']
    elif quant == 'voltage':
        tree = file['tvoltage']
    trace_dict = tree['trace'].arrays().tolist()
    traces = np.array(trace_dict[0]['trace'])
    return traces


def create_trace_db(PATH_data, lentrace, nfiles, nantsave):
    """Produce file with traces with root input.

    PATH_data: simulation folder
    lenefield: fixed length chosen for traces (ex. 900 or 1100)
    nfiles: number of events to process
    nantsave: number of traces to save per event
    """
    # BROWSE SIMULATION FILES (ZHAIRES)
    list_f = glob.glob(PATH_data+'*')
    print('Number of files = %i' % (len(list_f)))

    # CREATE A DATABANK OF TRACES
    efield_tot = np.empty([0, 3, lentrace])
    voltage_tot = np.empty([0, 3, lentrace])
    for k in range(nfiles):
        index = k
        print(k)
        print(list_f[index])
        inputfilename = glob.glob(list_f[index] + '/gr*.RawRoot')[0]
        file = uproot.open(inputfilename)
        if file['tshower']['zenith'].array()[0] > 70.:
            efield_arr = extract_tra(inputfilename, 'efield')
            inputfilename = glob.glob(list_f[index] + '/gr*.Voltage')[0]
            voltage_arr = extract_tra(inputfilename, 'voltage')
            # Find antennas with maximum power
            efield2_arr = np.sum(efield_arr[:, :, :]**2, axis=1)
            efield2_arr_max = np.max(efield2_arr, axis=1)
            efield2_ind = np.argsort(efield2_arr_max)
            efield_ord = efield_arr[efield2_ind[::-1], :, :]
            voltage_ord = voltage_arr[efield2_ind[::-1], :, :]
            efield_tot = np.append(efield_tot,
                                   efield_ord[0:nantsave, :, :lentrace],
                                   axis=0)
            voltage_tot = np.append(voltage_tot,
                                    voltage_ord[0:nantsave, :, :lentrace],
                                    axis=0)
    # Normalize and transpose
    efield_max = np.max(np.abs(efield_tot))
    print("efield_max = %.2e" % efield_max)
    voltage_max = np.max(np.abs(voltage_tot))
    print("voltage_max = %.2e" % voltage_max)
    efield_tot = np.transpose(efield_tot/efield_max, axes=(0, 2, 1))
    voltage_tot = np.transpose(voltage_tot/voltage_max, axes=(0, 2, 1))

    # SAVE TO FILE
    save_data(PATH_data, "efield_traces", efield_tot)
    save_data(PATH_data, "voltage_traces", voltage_tot)


if __name__ == "__main__":
    create_efield_file(sys.argv[1], int(sys.argv[2]),
                        int(sys.argv[3]), int(sys.argv[4]))

# path_loc = '/Users/claireguepin/Projects/GRAND/GP300LibraryXi2023/'
# create_trace_db(path_loc, 1000, 711, 5)

# print(np.shape(read_data("/Users/claireguepin/Projects/GRAND/GP300LibraryXi2023/voltage_traces.csv", 3)))

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
