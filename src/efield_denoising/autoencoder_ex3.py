"""
Reconstruct only peak amplitude and peak time.

Attempt to include conv2d, and then several antennas at once.

From tutorial https://ai.plainenglish.io/denoising-autoencoder
-in-pytorch-on-mnist-dataset-a76b8824e57e.

Created on Mon Oct 16
"""

# =============================================================================
# Modules
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch
from torch import nn

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# =============================================================================
# CREATE DATASETS

path = '/Users/claireguepin/Projects/GRAND/'\
    + 'ML_efield_recons/Recons_Peak_Voltage_Train5000_Epoch100_conv3d_RAdam/'

voltage = torch.load(path+'voltage_traces_p.pt')
info = pandas.read_csv(path+'info_p.csv')

# PARAMETERS
indmax = 1000.
tbin = 0.5  # ns

nsim = 1054
nant = 5
num_train = 800
num_valid = 100

# FEATURES

volmax = torch.max(torch.abs(voltage))
voltage /= volmax
voltage = voltage[:, ::4, :]
ntime = np.shape(voltage)[1]
print(np.shape(voltage))

pos = np.array([[info["DU x"], info["DU y"], info["DU z"]]])
posmax = np.max(pos)
pos /= posmax
pos = torch.tensor(pos.repeat(ntime, 0))
pos = torch.permute(pos, (2, 0, 1))
print(np.shape(pos))

# Concatenate voltage and antenna positions
features = torch.cat((voltage, pos), dim=2)
print(np.shape(features))
# Then separate into batches of 5 traces per event (5270/5 = 1054)
features = torch.reshape(features, (nsim, nant, ntime, 6))
print(np.shape(features))

vol = np.array([[np.sqrt(info["Sum V^2"]/max(info["Sum V^2"]))]])
voltot = torch.tensor(vol.repeat(ntime, 1)).mT
voltot = torch.reshape(voltot, (nsim, nant, ntime, 1))
print(np.shape(voltot))

# LABELS

ind = np.array([[info["Ind peak time"]]]).astype(float)
ind /= indmax
ind = torch.tensor(ind.repeat(ntime, 1)).mT
ind = torch.reshape(ind, (nsim, nant, ntime, 1))
print(np.shape(ind))

val = np.log10(np.array([[info["Val peak amplitude"]]]))
val = torch.tensor(val.repeat(ntime, 1)).mT
valmax = torch.max(torch.abs(val))
val /= valmax
val = torch.reshape(val, (nsim, nant, ntime, 1))
print(np.shape(val))

# Concatenate everything
data = torch.cat((features, voltot, ind, val), dim=3)
data = torch.permute(data, (0, 1, 3, 2))
print(np.shape(data))

train, valid, test = torch.split(data, (num_train, num_valid,
                                        nsim-num_valid-num_train), 0)

batch_size = 1

train_loader = torch.utils.data.DataLoader(train,
                                           batch_size=batch_size,
                                           shuffle=True)

valid_loader = torch.utils.data.DataLoader(valid,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test,
                                          batch_size=1,
                                          shuffle=True)

# Visualize distributions of labels
# Make sure they are not going to biais results (how?)

# The index distribution does not seem flat, which is intriguing

# ind = np.array([[info["Ind peak time"]]]).astype(float)[0, 0, :]
# # ind /= indmax
# plt.figure()
# ax = plt.gca()
# plt.hist(ind, bins=10, alpha=0.5)
# # plt.hist(ind_reco, bins=10, alpha=0.5,
# #          label='iron')
# ax.xaxis.set_ticks_position('both')
# ax.yaxis.set_ticks_position('both')
# ax.tick_params(labelsize=14)

# In log10 val peak alplitude looks like a Gaussian

# # val = np.array([[info["Val peak amplitude"]]])[0, 0, :]
# val = np.log10(np.array([[info["Val peak amplitude"]]]))[0, 0, :]
# valmax = max(np.abs(val))
# val /= valmax
# plt.figure()
# ax = plt.gca()
# plt.hist(val, bins=10, alpha=0.5)
# # plt.hist(ind_reco, bins=10, alpha=0.5,
# #          label='iron')
# ax.xaxis.set_ticks_position('both')
# ax.yaxis.set_ticks_position('both')
# ax.tick_params(labelsize=14)

# =============================================================================
# ENCODER AND DECODER


class Net(nn.Module):
    """Network structure."""

    def __init__(self, fc2_input_dim):
        super().__init__()

        # Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv3d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv3d(8, 16, 2, stride=2, padding=0),
            nn.ReLU(True),
        )

        # Flatten layer
        self.flatten = nn.Flatten(start_dim=0)
        # Linear section
        self.encoder_lin = nn.Sequential(
            # nn.Linear(16 * 250+1, fc2_input_dim),
            nn.Linear(16 * 62+1, fc2_input_dim),
            nn.ReLU(True),
        )
        self.encoder_lin2 = nn.Sequential(
            nn.Linear(fc2_input_dim, 10),
        )

    def forward(self, x):
        """Forward."""
        # print("ENCODE")
        # print(np.shape(x))
        xloc = x[:, 0, 6, 0]
        x = x[:, :, :6, :]
        # print(np.shape(x))
        x = self.encoder_cnn(x)
        # print(np.shape(x))
        x = self.flatten(x)
        # print(np.shape(x))
        x = torch.cat((x, xloc))
        x = self.encoder_lin(x)
        x = self.encoder_lin2(x)
        x = torch.reshape(x, (5, 2))
        # print(np.shape(x))
        return x


def train_epoch_den(encoder, device, dataloader, loss_fn, optimizer):
    """Training function."""
    # Set train mode for both the encoder and the decoder
    encoder.train()
    # decoder.train()
    train_loss = []
    # Iterate the dataloader
    iterloc = 0
    for data in dataloader:
        iterloc += 1
        features = data[:, :, :7, :]
        # print(np.shape(features))
        labels = data[0, :, 7:10, 0]
        # print(np.shape(labels))

        # Move tensor to the proper device
        labels = labels.to(device)
        features = features.to(device)

        # Encode data
        recons_data = encoder(features)
        # Evaluate loss
        loss = loss_fn(recons_data, labels)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # scheduler.step()

        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


def test_epoch_den(encoder, device, dataloader, loss_fn):
    """Testing function."""
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    # decoder.eval()
    # No need to track the gradients
    with torch.no_grad():
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for data in dataloader:
            features = data[:, :, :7, :]
            labels = data[0, :, 7:10, 0]

            # Move tensor to the proper device
            features = features.to(device)

            # Encode data
            recons_data = encoder(features)

            # Append the network output and the original trace to the lists
            conc_out.append(recons_data.cpu())
            conc_label.append(labels.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data


# =============================================================================
# Define the loss function
loss_fn = torch.nn.MSELoss()

# Define an optimizer (both for the encoder and the decoder!)
# lr = 0.0001

# Set the random seed for reproducible results
torch.manual_seed(0)

# Initialize the network
net = Net(fc2_input_dim=16).double()
params_to_optimize = net.parameters()

# optimtype = torch.optim.Adam(params_to_optimize)
# optimtype = torch.optim.NAdam(params_to_optimize)
optimtype = torch.optim.RAdam(params_to_optimize)
# optimtype = torch.optim.RAdam(params_to_optimize, weight_decay=1e-3)

# we can also implement momentum decay, results are quite sensitive to it
# optimtype = torch.optim.NAdam(params_to_optimize, momentum_decay=0.003)

# optimtype = torch.optim.SGD(params_to_optimize)

# scheduler = torch.optim.lr_scheduler.CyclicLR(optimtype,
#                                               base_lr=0.01,
#                                               max_lr=0.1)

# Check if the GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f'Selected device: {device}')

# Move both the encoder and the decoder to the selected device
net.to(device)
# decoder.to(device)

# =============================================================================
# TRAINING

num_epochs = 10
history_da = {'train_loss': [], 'val_loss': []}

print('\n TRAINING and VALIDATION')

for epoch in range(num_epochs):
    # print('EPOCH %d/%d' % (epoch + 1, num_epochs))
    # Training (use the training function)
    train_loss = train_epoch_den(
        encoder=net,
        device=device,
        dataloader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimtype)
    # Validation  (use the testing function)
    val_loss = test_epoch_den(
        encoder=net,
        device=device,
        dataloader=valid_loader,
        loss_fn=loss_fn)
    # Print Validationloss
    history_da['train_loss'].append(train_loss)
    history_da['val_loss'].append(val_loss)
    print('\n EPOCH {}/{} \t train loss {:.3e} \t val loss {:.3e}'
          . format(epoch + 1, num_epochs, train_loss, val_loss))

# =============================================================================
# VISUALIZE LOSS

indobj = 10  # 5ns
valobj = 0.10  # 5-10%

it_train = 0
MSE = 0
for train_sig in train_loader:
    it_train += 5
    # ampl = 10**np.array(train_sig[0, :, 8, 0])
    # MSE += 5.*(indobj/indmax)**2 + np.sum((valobj*ampl)**2)
    MSE += 5.*(indobj/indmax)**2 + 5.*np.log10(1.-valobj)**2
MSE /= it_train

fig = plt.figure()
ax = plt.gca()
plt.subplots_adjust(left=0.13)
plt.plot(history_da['train_loss'], ls='-', lw=2, label="Training loss")
plt.plot(history_da['val_loss'], ls='-', lw=2, label="Validation loss")
plt.plot(np.ones(len(history_da['train_loss']))*MSE, ls='--', color='k')
ax.tick_params(labelsize=14)
plt.xlabel(r'Epoch', fontsize=14)
plt.ylabel(r'Loss', fontsize=14)
plt.legend(frameon=False, fontsize=14)
# ax.set_xlim([-10000,10000])
# ax.set_ylim([-10000,10000])
plt.title("Final train loss = %.2e and validation loss = %.2e "
          % (history_da['train_loss'][-1], history_da['val_loss'][-1]),
          fontsize=14)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
plt.savefig(path+"Loss.pdf")
plt.show()

fig = plt.figure()
ax = plt.gca()
plt.subplots_adjust(left=0.13)
plt.plot(np.log10(history_da['train_loss']), ls='-', lw=2,
         label="Training loss")
plt.plot(np.log10(history_da['val_loss']), ls='-', lw=2,
         label="Validation loss")
plt.plot(np.log10(np.ones(len(history_da['train_loss']))*MSE),
         ls='--', color='k')
ax.tick_params(labelsize=14)
plt.xlabel(r'Epoch', fontsize=14)
plt.ylabel(r'$\log_{10}$ Loss', fontsize=14)
plt.legend(frameon=False, fontsize=14)
# ax.set_xlim([-10000,10000])
# ax.set_ylim([-10000,10000])
plt.title("Final train loss = %.2e and validation loss = %.2e "
          % (history_da['train_loss'][-1], history_da['val_loss'][-1]),
          fontsize=14)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
plt.savefig(path+"Log10Loss.pdf")
plt.show()

# =============================================================================
# VISUALIZE SOME TRAIN and TEST results


def vis_hist(data_loader, namefig):
    """Visualize histograms."""
    ind_real = np.empty([0])
    ind_reco = np.empty([0])
    val_real = np.empty([0])
    val_reco = np.empty([0])
    for data in data_loader:
        ind_real = np.append(ind_real, data[0, :, 7, 0])
        val_real = np.append(val_real, data[0, :, 8, 0])
        features = data[:, :, 0:7, :]
        features = features.to(device)
        net.eval()
        with torch.no_grad():
            rec_sig = (net(features))
        ind_reco = np.append(ind_reco, rec_sig[:, 0])
        val_reco = np.append(val_reco, rec_sig[:, 1])

    plt.figure()
    ax = plt.gca()
    plt.hist((ind_real-ind_reco)*indmax*tbin, bins=10, alpha=0.5)
    # plt.hist(ind_reco, bins=10, alpha=0.5,
    #          label='iron')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(labelsize=14)
    mean = np.mean((ind_real-ind_reco)*indmax*tbin)
    std = np.std((ind_real-ind_reco)*indmax*tbin)
    ax.set_title(r"Histogram Time (i_{\rm real}-i_{\rm reco})"
                 + "\n"+r"Mean: %.1f, Std: %.1f" % (mean, std), fontsize=14)
    # ax.legend(prop={'size': 12}, frameon=False)
    plt.savefig(path+"Hist_ind_"+namefig+".pdf")

    plt.figure()
    ax = plt.gca()
    plt.hist((10**val_real-10**val_reco)/10**val_real, bins=10, alpha=0.5)
    # plt.hist(ind_reco, bins=10, alpha=0.5,
    #          label='iron')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(labelsize=14)
    mean = np.mean((val_real-val_reco)/val_real)
    std = np.std((val_real-val_reco)/val_real)
    ax.set_title(r"Histogram Amplitude"
                 + r"(A_{\rm real}-A_{\rm reco})/A_{\rm real}"
                 + "\n"+r"Mean: %.1e, Std: %.1e" % (mean, std), fontsize=14)
    # ax.legend(prop={'size': 12}, frameon=False)
    plt.savefig(path+"Hist_val_"+namefig+".pdf")


# TRAIN
vis_hist(train_loader, 'train')

# VALIDATION
vis_hist(valid_loader, 'valid')

# TEST
vis_hist(test_loader, 'test')
