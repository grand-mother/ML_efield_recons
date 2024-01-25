"""
Visualize histograms, ZHAireS simulations.
Data files prepared for IA, reconstruction or classification.

Created on Mon Jan 22 18:08:03 2024

@author: claireguepin
"""

import numpy as np
import pandas
import torch
import matplotlib.pyplot as plt
from scipy.signal import hilbert

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

PATH_gen = '/Users/claireguepin/Projects/GRAND/'

PATH_data_p = PATH_gen+'GP300LibraryXi2023_proton/'
info_p = pandas.read_csv(PATH_data_p+"info_p"+'.csv')

PATH_data = PATH_gen+'GP300LibraryXi2023_iron/'
info_Fe = pandas.read_csv(PATH_data+"info_Fe"+'.csv')

# General information about air showers

plt.figure()
ax = plt.gca()
plt.hist(np.log10(np.array(info_p["Energy"])), bins=10, alpha=0.5,
         label='proton')
plt.hist(np.log10(np.array(info_Fe["Energy"])), bins=10, alpha=0.5,
         label='iron')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params(labelsize=14)
ax.set_title(r"$\log_{10} E_{\rm primaire}$ histogram", fontsize=14)
ax.legend(prop={'size': 12}, frameon=False)
plt.savefig(PATH_gen+"Hist_energy.pdf")

plt.figure()
ax = plt.gca()
plt.hist(np.array(info_p["Zenith"]), bins=10, alpha=0.5, label='proton')
plt.hist(np.array(info_Fe["Zenith"]), bins=10, alpha=0.5, label='iron')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params(labelsize=14)
ax.set_title(r"Zenith histogram", fontsize=14)
ax.legend(prop={'size': 12}, frameon=False)
plt.savefig(PATH_gen+"Hist_zenith.pdf")

plt.figure()
ax = plt.gca()
plt.hist(np.array(info_p["Azimuth"]), bins=10, alpha=0.5, label='proton')
plt.hist(np.array(info_Fe["Azimuth"]), bins=10, alpha=0.5, label='iron')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params(labelsize=14)
ax.set_title(r"Azimuth histogram", fontsize=14)
ax.legend(prop={'size': 12}, frameon=False)
plt.savefig(PATH_gen+"Hist_azimuth.pdf")

# Energy content of footprint

plt.figure()
ax = plt.gca()
plt.hist(np.log10(np.array(info_p["Sum E^2"])), bins=10, alpha=0.5,
         label='proton')
plt.hist(np.log10(np.array(info_Fe["Sum E^2"])), bins=10, alpha=0.5,
         label='iron')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params(labelsize=14)
ax.set_title(r"$\log_{10}\sum E^2$ histogram", fontsize=14)
ax.legend(prop={'size': 12}, frameon=False)
plt.savefig(PATH_gen+"Hist_E2.pdf")

plt.figure()
ax = plt.gca()
plt.hist(np.log10(np.array(info_p["Sum V^2"])), bins=10, alpha=0.5,
         label='proton')
plt.hist(np.log10(np.array(info_Fe["Sum V^2"])), bins=10, alpha=0.5,
         label='iron')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params(labelsize=14)
ax.set_title(r"$\log_{10}\sum V^2$ histogram", fontsize=14)
ax.legend(prop={'size': 12}, frameon=False)
plt.savefig(PATH_gen+"Hist_V2.pdf")

plt.show()

# Trace and hilbert

efield_tensor = torch.load(PATH_data_p+"efield_traces_p"+'.pt')
efield_ex1 = np.array(efield_tensor[0, :, :])

# Enveloppe de hilbert x, y, z channels
hilbert_amp = np.abs(hilbert(efield_ex1.T))
# Find best peakamp for the 3 channels
peakamplitude = max([max(hilbert_amp[0, :]),
                     max(hilbert_amp[1, :]),
                     max(hilbert_amp[2, :])])
# Get the time of the peak amplitude
times = np.arange(0, len(efield_ex1)*0.5, 0.5)
ipeak = np.where(hilbert_amp == peakamplitude)[1][0]
peaktime = times[ipeak]

plt.figure()
ax = plt.gca()
plt.plot(times, efield_ex1[:, 0])
plt.plot(times, efield_ex1[:, 1])
plt.plot(times, efield_ex1[:, 2])

plt.plot(times, hilbert_amp[0, :], ls='--')
plt.plot(times, hilbert_amp[1, :], ls=':')
plt.plot(times, hilbert_amp[2, :], ls='-.')

# plt.plot(times, np.sqrt(efield_ex1[:, 0]**2
#                         + efield_ex1[:, 1]**2
#                         + efield_ex1[:, 2]**2), ls='--')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params(labelsize=14)
# ax.set_title(r"$\log_{10}\sum V^2$ histogram", fontsize=14)
# ax.legend(prop={'size': 12}, frameon=False)
# plt.savefig(PATH_gen+"Hist_V2.pdf")
