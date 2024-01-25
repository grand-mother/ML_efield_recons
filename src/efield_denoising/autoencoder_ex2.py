"""
Reconstruct only peak amplitude and peak time.

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
    + 'ML_efield_recons/Recons_Peak_Voltage_Train5000_Epoch50/'

info = pandas.read_csv(path+'info_p.csv')

ind = np.array([[info["Ind peak time"]]])
ind = torch.tensor(ind.repeat(3, 0).repeat(1000, 1).T)
indmax = 1000
print(np.shape(ind))
val = np.array([[info["Val peak amplitude"]]])
val = torch.tensor(val.repeat(3, 0).repeat(1000, 1).T)
valmax = torch.max(torch.abs(val))
print(np.shape(val))
vol = np.array([[np.sqrt(info["Sum V^2"]/max(info["Sum V^2"]))]])
voltot = torch.tensor(vol.repeat(3, 0).repeat(1000, 1).T)
print(np.shape(voltot))
voltage = torch.load(path+'voltage_traces_p.pt')
volmax = torch.max(torch.abs(voltage))
print(np.shape(voltage))
data = torch.stack((ind/indmax, val/valmax, voltage/volmax, voltot), dim=3)
print(np.shape(data))
data = torch.permute(data, (0, 2, 1, 3))
print(np.shape(data))

train, valid, test = torch.split(data, (5000, 100, 170), 0)

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

# =============================================================================
# ENCODER AND DECODER


class Net(nn.Module):
    """Network structure."""

    def __init__(self, fc2_input_dim):
        super().__init__()

        # Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(3, 8, 3, stride=2, padding=0),
            nn.ReLU(True),
            nn.Conv1d(8, 16, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        # Flatten layer
        self.flatten = nn.Flatten(start_dim=0)
        # Linear section
        # self.encoder_lin = nn.Sequential(
        #     nn.Linear(16 * 249 + 1, fc2_input_dim),
        #     nn.ReLU(True),
        #     nn.Linear(fc2_input_dim, 2)
        # )
        self.encoder_lin = nn.Sequential(
            nn.Linear(16 * 249+1, fc2_input_dim),
            nn.ReLU(True),
        )
        self.encoder_lin2 = nn.Sequential(
            nn.Linear(fc2_input_dim, 2)
        )

    def forward(self, x):
        # print("ENCODE")
        print(np.shape(x))
        xloc = x[:, 0, 0, 1]
        x = x[:, :, :, 0]
        print(np.shape(x))
        x = self.encoder_cnn(x)
        print(np.shape(x))
        # print(np.shape(x))
        # print(np.shape(x))
        x = self.flatten(x)
        # print(np.shape(x))
        x = torch.cat((x, xloc))
        x = self.encoder_lin(x)
        x = self.encoder_lin2(x)
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
    for image in dataloader:
        iterloc += 1
        image_batch = image[0, 0, 0, 0:2]
        image_noisy = image[:, :, :, 2:4]

        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        image_noisy = image_noisy.to(device)

        # Encode data
        recons_data = encoder(image_noisy)

        # Evaluate loss
        loss = loss_fn(recons_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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
        for image in dataloader:
            image_batch = image[0, 0, 0, 0:2]
            image_noisy = image[:, :, :, 2:4]

            # Move tensor to the proper device
            image_noisy = image_noisy.to(device)

            # Encode data
            recons_data = encoder(image_noisy)

            # Append the network output and the original trace to the lists
            conc_out.append(recons_data.cpu())
            conc_label.append(image_batch.cpu())
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
net = Net(fc2_input_dim=8).double()
params_to_optimize = net.parameters()

# optimtype = torch.optim.Adam(params_to_optimize)
# optimtype = torch.optim.NAdam(params_to_optimize)
optimtype = torch.optim.RAdam(params_to_optimize)

# we can also implement momentum decay, results are quite sensitive to it
# optimtype = torch.optim.NAdam(params_to_optimize, momentum_decay=0.003)

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

num_epochs = 100
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

# it_train = 0
# MSE = 0
# for train_sig in train_loader:
#     it_train += 1
#     efield = train_sig[:, :, :, 0]
#     MSE += np.sum(np.array(efield[0]**2))/len(efield[0][0])/len(efield[0])
# print("MSE with recons of null function = %.2e" % (MSE/it_train))

fig = plt.figure()
ax = plt.gca()
plt.subplots_adjust(left=0.13)
plt.plot(history_da['train_loss'], ls='-', lw=2, label="Training loss")
plt.plot(history_da['val_loss'], ls='-', lw=2, label="Validation loss")
# plt.plot(np.ones(len(history_da['train_loss']))*MSE/it_train,
#          ls='--', color='k')
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
# plt.plot(np.ones(len(history_da['train_loss']))*MSE/it_train,
#          ls='--', color='k')
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

# TRAIN

ind_real = np.empty([0])
ind_reco = np.empty([0])
val_real = np.empty([0])
val_reco = np.empty([0])
for image in train_loader:
    ind_real = np.append(ind_real, image[0, 0, 0, 0])
    val_real = np.append(val_real, image[0, 0, 0, 1])
    voltage = image[:, :, :, 2:4]
    voltage = voltage.to(device)
    net.eval()
    with torch.no_grad():
        rec_sig = (net(voltage))
    ind_reco = np.append(ind_reco, rec_sig[0])
    val_reco = np.append(val_reco, rec_sig[1])

plt.figure()
ax = plt.gca()
plt.hist((ind_real-ind_reco)*1000, bins=10, alpha=0.5)
# plt.hist(ind_reco, bins=10, alpha=0.5,
#          label='iron')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params(labelsize=14)
mean = np.mean((ind_real-ind_reco)*1000)
std = np.std((ind_real-ind_reco)*1000)
ax.set_title(r"Histogram Time index (i_{\rm real}-i_{\rm reco})"
             + "\n"+r"Mean: %.1f, Std: %.1f" % (mean, std), fontsize=14)
# ax.legend(prop={'size': 12}, frameon=False)
plt.savefig(path+"Hist_ind_train.pdf")

plt.figure()
ax = plt.gca()
plt.hist((val_real-val_reco)/val_real, bins=10, alpha=0.5)
# plt.hist(ind_reco, bins=10, alpha=0.5,
#          label='iron')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params(labelsize=14)
mean = np.mean((val_real-val_reco)/val_real)
std = np.std((val_real-val_reco)/val_real)
ax.set_title(r"Histogram Amplitude (A_{\rm real}-A_{\rm reco})/A_{\rm real}"
             + "\n"+r"Mean: %.1e, Std: %.1e" % (mean, std), fontsize=14)
# ax.legend(prop={'size': 12}, frameon=False)
plt.savefig(path+"Hist_val_train.pdf")

# TEST

ind_real = np.empty([0])
ind_reco = np.empty([0])
val_real = np.empty([0])
val_reco = np.empty([0])
for image in test_loader:
    ind_real = np.append(ind_real, image[0, 0, 0, 0])
    val_real = np.append(val_real, image[0, 0, 0, 1])
    voltage = image[:, :, :, 2:4]
    voltage = voltage.to(device)
    net.eval()
    with torch.no_grad():
        rec_sig = (net(voltage))
    ind_reco = np.append(ind_reco, rec_sig[0])
    val_reco = np.append(val_reco, rec_sig[1])

plt.figure()
ax = plt.gca()
plt.hist((ind_real-ind_reco)*1000, bins=10, alpha=0.5)
# plt.hist(ind_reco, bins=10, alpha=0.5,
#          label='iron')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params(labelsize=14)
mean = np.mean((ind_real-ind_reco)*1000)
std = np.std((ind_real-ind_reco)*1000)
ax.set_title(r"Histogram Time index (i_{\rm real}-i_{\rm reco})"
             + "\n"+r"Mean: %.1f, Std: %.1f" % (mean, std), fontsize=14)
# ax.legend(prop={'size': 12}, frameon=False)
plt.savefig(path+"Hist_ind_test.pdf")


plt.figure()
ax = plt.gca()
plt.hist((val_real-val_reco)/val_real, bins=10, alpha=0.5)
# plt.hist(ind_reco, bins=10, alpha=0.5,
#          label='iron')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params(labelsize=14)
mean = np.mean((val_real-val_reco)/val_real)
std = np.std((val_real-val_reco)/val_real)
ax.set_title(r"Histogram Amplitude (A_{\rm real}-A_{\rm reco})/A_{\rm real}"
             + "\n"+r"Mean: %.1e, Std: %.1e" % (mean, std), fontsize=14)
# ax.legend(prop={'size': 12}, frameon=False)
plt.savefig(path+"Hist_val_test.pdf")
