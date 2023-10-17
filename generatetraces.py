"""
Created on Tue Oct 17 2023.

Generate train, val and test datasets, with simple functions for case study
No noise included here
Next we will use realistic traces
"""

# =============================================================================
# MODULES

import matplotlib.pyplot as plt
import numpy as np
import random
import torch


# =============================================================================
# DATASET

def dataset(ts, ndat):
    """Create dataset, case study: sin convolved with gaussian."""
    train_dataset = []
    for idat in range(ndat_train):
        # Parameters traces: random or fixed values
        # tssig = 512
        tssig = random.uniform(512-10, 512+10)
        # omega = 0.05
        omega = random.uniform(0.03, 0.05)
        # sigma = 500
        sigma = random.uniform(400, 500)
        trace = np.sin(2.*np.pi*(ts-tssig)*omega)*np.exp(-(ts-tssig)**2/sigma)
        train_dataset.append([trace])
    return torch.tensor(np.array(train_dataset))


# Time bins
nts = 1024
ts = np.arange(0, nts, 1)

ndat_train = 256
ndat_valid = 64
ndat_test = 10
train_ds_tensor = dataset(ts, ndat_train)
valid_ds_tensor = dataset(ts, ndat_valid)
test_ds_tensor = dataset(ts, ndat_test)

# This signal does not include noise
# This can be done during training using a dedicated function

# =============================================================================
# VISUALIZE FAKE SIGNAL

fig = plt.figure()
ax = plt.gca()
plt.subplots_adjust(left=0.13)
for i in range(3):
    plt.plot(ts, np.array(train_ds_tensor)[i, 0, :], ls='-', lw=2)
ax.tick_params(labelsize=14)
# plt.xlabel(r'Simulation number', fontsize=14)
# plt.ylabel(r'$\log_{10} (E)$', fontsize=14)
# ax.set_xlim([-10000,10000])
# ax.set_ylim([-10000,10000])
plt.show()

# =============================================================================
# DATALOADER

batch_size = 1
train_loader = torch.utils.data.DataLoader(train_ds_tensor,
                                           batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(valid_ds_tensor,
                                           batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_ds_tensor,
                                          batch_size=1,
                                          shuffle=True)
