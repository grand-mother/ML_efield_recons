"""
From tutorial https://ai.plainenglish.io/denoising-autoencoder-in-pytorch-on-mnist-dataset-a76b8824e57e.

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

# =============================================================================
# CREATE DATASETS

path = '/Users/claireguepin/Projects/GRAND/'\
    + 'ML_efield_recons/ReconsId_Train5000_OptimRAdam_d8_Epoch10_Voltage/'

info = pandas.read_csv(path+'info_p.csv')
maxi = np.sqrt(max(info["Sum E^2"]))*1e-2
vol = np.array(np.sqrt(info["Sum V^2"]/max(info["Sum V^2"])))
volt = np.repeat(vol[:, np.newaxis], 1000, axis=1)
voltagetot = torch.tensor(np.repeat(volt[:, :, np.newaxis], 3, axis=2))

efield = torch.load(path+'efield_traces_p.pt')
print(np.shape(efield))
voltage = torch.load(path+'voltage_traces_p.pt')
print(np.shape(voltage))
data = torch.stack((efield, voltage, voltagetot), dim=3)
print(np.shape(data))
data = torch.permute(data, (0, 2, 1, 3))/maxi
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


class Encoder(nn.Module):
    """Encoder structure."""

    def __init__(self, encoded_space_dim, fc2_input_dim):
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
        self.encoder_lin = nn.Sequential(
            nn.Linear(16 * 249, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, encoded_space_dim)
        )

    def forward(self, x):
        # print("ENCODE")
        # print(np.shape(x))
        xloc = x[:, 0, 0, 1]
        x = x[:, :, :, 0]
        # x = torch.cat((x[:, :, :, 0], xloc))
        # print(np.shape(x))
        x = self.encoder_cnn(x)
        # print(np.shape(x))
        x = self.flatten(x)
        # xloc = np.repeat(xloc, len(x))
        # x = torch.cat((x, xloc))
        # print(np.shape(x))
        x = self.encoder_lin(x)
        x = torch.cat((x, xloc))
        return x


class Decoder(nn.Module):
    """Decoder structure."""

    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim+1, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, 16 * 249),
        )
        self.unflatten = nn.Unflatten(dim=0, unflattened_size=(1, 16, 249))
        self.decoder_conv = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 8, 3, stride=2, padding=0,
                               output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose1d(8, 3, 3, stride=2, padding=0, output_padding=1)
        )

    def forward(self, x):
        # print(np.shape(x))
        x = self.decoder_lin(x)
        # print(np.shape(x))
        x = self.unflatten(x)
        # print(np.shape(x))
        x = self.decoder_conv(x)
        # print(np.shape(x))
        # x = 1.5*torch.tanh(x)
        return x


def train_epoch_den(encoder, decoder, device, dataloader, loss_fn, optimizer):
    """Training function."""
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader
    iterloc = 0
    for image in dataloader:
        iterloc += 1
        image_batch = image[:, :, :, 0]
        image_noisy = image[:, :, :, 1:3]

        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        image_noisy = image_noisy.to(device)

        # Encode data
        encoded_data = encoder(image_noisy)
        # Decode data
        decoded_data = decoder(encoded_data)

        # if (iterloc % 200) == 0:
        #     plt.figure()
        #     ax = plt.gca()
        #     plt.subplots_adjust(left=0.13)
        #     plt.plot(image_batch[0, :], ls='-', lw=2, label='Efield')
        #     plt.plot(image_noisy[0, :], ls='-', lw=2, label='Voltage')
        #     plt.plot(np.array(decoded_data.detach().numpy())[0, :],
        #               ls='-', lw=2, label='Signal rec')
        #     ax.tick_params(labelsize=14)
        #     # plt.xlabel(r'Simulation number', fontsize=14)
        #     # plt.ylabel(r'$\log_{10} (E)$', fontsize=14)
        #     plt.legend(frameon=False, fontsize=14)
        #     # ax.set_xlim([-10000,10000])
        #     # ax.set_ylim([-10000,10000])
        #     plt.show()

        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


def test_epoch_den(encoder, decoder, device, dataloader, loss_fn):
    """Testing function."""
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    # No need to track the gradients
    with torch.no_grad():
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image in dataloader:
            image_batch = image[:, :, :, 0]
            image_noisy = image[:, :, :, 1:3]

            # Move tensor to the proper device
            image_noisy = image_noisy.to(device)

            # Encode data
            encoded_data = encoder(image_noisy)
            # Decode data
            decoded_data = decoder(encoded_data)

            # Append the network output and the original trace to the lists
            conc_out.append(decoded_data.cpu())
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

# Initialize the two networks
d = 8

# model = Autoencoder(encoded_space_dim=encoded_space_dim)
encoder = Encoder(encoded_space_dim=d, fc2_input_dim=8).double()
decoder = Decoder(encoded_space_dim=d, fc2_input_dim=8).double()
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

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
encoder.to(device)
decoder.to(device)

# =============================================================================
# TRAINING

num_epochs = 200
history_da = {'train_loss': [], 'val_loss': []}

print('\n TRAINING and VALIDATION')

for epoch in range(num_epochs):
    # print('EPOCH %d/%d' % (epoch + 1, num_epochs))
    # Training (use the training function)
    train_loss = train_epoch_den(
        encoder=encoder,
        decoder=decoder,
        device=device,
        dataloader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimtype)
    # Validation  (use the testing function)
    val_loss = test_epoch_den(
        encoder=encoder,
        decoder=decoder,
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

it_train = 0
MSE = 0
for train_sig in train_loader:
    it_train += 1
    efield = train_sig[:, :, :, 0]
    MSE += np.sum(np.array(efield[0]**2))/len(efield[0][0])/len(efield[0])
print("MSE with recons of null function = %.2e" % (MSE/it_train))

fig = plt.figure()
ax = plt.gca()
plt.subplots_adjust(left=0.13)
plt.plot(history_da['train_loss'], ls='-', lw=2, label="Training loss")
plt.plot(history_da['val_loss'], ls='-', lw=2, label="Validation loss")
plt.plot(np.ones(len(history_da['train_loss']))*MSE/it_train,
         ls='--', color='k')
ax.tick_params(labelsize=14)
# plt.xlabel(r'Simulation number', fontsize=14)
# plt.ylabel(r'$\log_{10} (E)$', fontsize=14)
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

# =============================================================================
# VISUALIZE SOME TESTS

it_train = 0
for train_sig in train_loader:
    it_train += 1
    efield = train_sig[:, :, :, 0]
    voltage = train_sig[:, :, :, 1:3]
    voltage = voltage.to(device)
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        rec_sig = decoder(encoder(voltage))

    imax = int(np.argmax(efield[0][0]))
    fig = plt.figure()
    ax = plt.gca()
    plt.subplots_adjust(left=0.13)
    # plt.plot(efield[0], ls='-', lw=3, label="Efield")
    # plt.plot(voltage[0], ls='-', lw=3, label="Signal")
    # plt.plot(rec_sig[0], ls='-', lw=2,
    #          label="Reconstructed signal")
    plt.plot(efield[0][0][imax-50:imax+50], ls='-', lw=3, label="Efield")
    plt.plot(rec_sig[0][0][imax-50:imax+50], ls='-', lw=2,
             label="Reconstructed signal")
    ax.tick_params(labelsize=14)
    # plt.xlabel(r'Simulation number', fontsize=14)
    # plt.ylabel(r'$\log_{10} (E)$', fontsize=14)
    plt.legend(frameon=False, fontsize=14, loc=1)
    # ax.set_xlim([-10000,10000])
    # ax.set_ylim([-10000,10000])
    MSEloc = np.sum(np.array(rec_sig[0]-efield[0])**2)\
        / len(efield[0][0])/len(efield[0])
    ax.set_title("MSE = %.2e" % MSEloc, fontsize=14)
    plt.savefig(path+"Train_ex"+str(it_train)+".pdf")
    plt.show()

    if it_train > 4:
        break

it_test = 0
for test_sig in test_loader:
    it_test += 1
    efield = test_sig[:, :, :, 0]
    voltage = test_sig[:, :, :, 1:3]
    voltage = voltage.to(device)
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        rec_sig = decoder(encoder(voltage))

    imax = int(np.argmax(efield[0][0]))
    fig = plt.figure()
    ax = plt.gca()
    plt.subplots_adjust(left=0.13)
    plt.plot(efield[0][0][imax-50:imax+50], ls='-', lw=3, label="Signal")
    plt.plot(rec_sig[0][0][imax-50:imax+50], ls='-', lw=2,
             label="Reconstructed signal")
    ax.tick_params(labelsize=14)
    # plt.xlabel(r'Simulation number', fontsize=14)
    # plt.ylabel(r'$\log_{10} (E)$', fontsize=14)
    plt.legend(frameon=False, fontsize=14, loc=1)
    # ax.set_xlim([-10000,10000])
    # ax.set_ylim([-10000,10000])
    MSEloc = np.sum(np.array(rec_sig[0]-efield[0])**2)\
        / len(efield[0][0])/len(efield[0])
    ax.set_title("MSE = %.2e" % MSEloc, fontsize=14)
    plt.savefig(path+"Test_ex"+str(it_test)+".pdf")
    plt.show()

    if it_test > 4:
        break
