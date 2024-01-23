"""
Visualize histograms, ZHAireS simulations.
Data files prepared for IA, reconstruction or classification.

Created on Mon Jan 22 18:08:03 2024

@author: claireguepin
"""

import numpy as np
import pandas
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

PATH_gen = '/Users/claireguepin/Projects/GRAND/'

PATH_data = PATH_gen+'GP300LibraryXi2023_proton/'
info_p = pandas.read_csv(PATH_data+"info"+'.csv')

PATH_data = PATH_gen+'GP300LibraryXi2023_iron/'
info_Fe = pandas.read_csv(PATH_data+"info"+'.csv')

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
