"""Created on Thu Jun  4 17:33:58 2020.

@author: guepin

Create data files from ZHAireS simulations.
Info in csv file, traces in tensors, compressed, with pt extension.
One line of info per antenna saved.
"""

import sys
import glob
import numpy as np
import uproot
import pandas
import torch
import matplotlib.pyplot as plt
from scipy.signal import hilbert

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def get_peak_tim_amp_hil(efield):
    """Peak amplitude and time using hilbert envelope."""
    # Enveloppe de hilbert x, y, z channels
    hilbert_amp = np.abs(hilbert(efield))
    # Find best peakamp for the 3 channels
    peakamplitude = max([max(hilbert_amp[0, :]),
                         max(hilbert_amp[1, :]),
                         max(hilbert_amp[2, :])])
    # Get the index of time of the peak amplitude
    ipeak = np.where(hilbert_amp == peakamplitude)[1][0]
    return [ipeak, peakamplitude]


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


def create_data_db(PATH_data, lentrace, nfiles, nantsave):
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
    # ADD number to refer to simulation number
    # ADD primary, energy, zenith and azimuth
    # ADD file name
    info_tot = np.empty([0, 11])
    efield_tot = np.empty([0, 3, lentrace])
    voltage_tot = np.empty([0, 3, lentrace])
    for k in range(nfiles):
        index = k
        print("\nFile number %i / %i" % (k+1, nfiles))
        # print(list_f[index])

        inputfilename = glob.glob(list_f[index] + '/gr*.RawRoot')[0]
        file = uproot.open(inputfilename)
        prim = file['tshower']['primary_type'].array()[0]
        ener = file['tshower']['energy_primary'].array()[0]
        zeni = file['tshower']['zenith'].array()[0]
        azim = file['tshower']['azimuth'].array()[0]
        positions = np.array(file['trun']['du_xyz'].arrays()[0]['du_xyz'])
        # print("Primary= %s" % prim)
        # print("Energy= %.2e" % ener)
        # print("Zenith angle = %.2f" % zeni)
        # print("Azimuth angle = %.2f" % azim)

        if zeni > 70. and ener > 1e8:

            # Find antennas with maximum power in efield
            efield_arr = extract_tra(inputfilename, 'efield')
            efield2_sum = np.sum(efield_arr**2)
            efield2_arr = np.sum(efield_arr[:, :, :]**2, axis=1)
            efield2_arr_max = np.max(efield2_arr, axis=1)
            efield2_ind = np.argsort(efield2_arr_max)
            efield_ord = efield_arr[efield2_ind[::-1], :, :]

            # Voltage
            inputfilename = glob.glob(list_f[index] + '/gr*.Voltage')[0]
            voltage_arr = extract_tra(inputfilename, 'voltage')
            voltage2_sum = np.sum(voltage_arr**2)
            voltage_ord = voltage_arr[efield2_ind[::-1], :, :]

            # Positions
            positions_ord = positions[efield2_ind[::-1], :]

            for i in range(nantsave):
                val_peak = get_peak_tim_amp_hil(efield_ord[i, :, :lentrace])
                efield_ind_tim_peak = val_peak[0]
                efield_val_amp_peak = val_peak[1]
                info = np.array([np.array([prim, ener, zeni, azim,
                                           efield2_sum, voltage2_sum,
                                           efield_ind_tim_peak,
                                           efield_val_amp_peak,
                                           positions_ord[i, 0],
                                           positions_ord[i, 1],
                                           positions_ord[i, 2]])])
                info_tot = np.append(info_tot, info, axis=0)

            efield_tot = np.append(efield_tot,
                                   efield_ord[0:nantsave, :, :lentrace],
                                   axis=0)

            voltage_tot = np.append(voltage_tot,
                                    voltage_ord[0:nantsave, :, :lentrace],
                                    axis=0)

    # Transpose

    efield_tot = np.transpose(efield_tot, axes=(0, 2, 1))
    voltage_tot = np.transpose(voltage_tot, axes=(0, 2, 1))

    # SAVE TO FILE

    print(np.shape(info_tot))
    info_df = pandas.DataFrame(info_tot,
                               columns=["Primary", "Energy",
                                        "Zenith", "Azimuth",
                                        "Sum E^2", "Sum V^2",
                                        "Ind peak time",
                                        "Val peak amplitude",
                                        "DU x", "DU y", "DU z"])
    info_df.to_csv(PATH_data+"info"+'.csv')

    print(np.shape(efield_tot))
    efield_tensor = torch.tensor(efield_tot)
    torch.save(efield_tensor, PATH_data+"efield_traces"+'.pt')

    print(np.shape(voltage_tot))
    voltage_tensor = torch.tensor(voltage_tot)
    torch.save(voltage_tensor, PATH_data+"voltage_traces"+'.pt')


# if __name__ == "__main__":
#     create_efield_file(sys.argv[1], int(sys.argv[2]),
#                         int(sys.argv[3]), int(sys.argv[4]))

path_loc = '/Users/claireguepin/Projects/GRAND/GP300LibraryXi2023_proton/'
create_data_db(path_loc, 1000, 1737, 5)
# create_data_db(path_loc, 1000, 10, 5)

# path_loc = '/Users/claireguepin/Projects/GRAND/GP300LibraryXi2023_iron/'
# create_data_db(path_loc, 1000, 1724, 5)

# =============================================================================
# HOW TO LOAD DATA

# efield_tensor = torch.load(path_loc+"efield_traces"+'.pt')
