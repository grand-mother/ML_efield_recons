"""
From tutorial https://ai.plainenglish.io/denoising-autoencoder-in-pytorch-on-mnist-dataset-a76b8824e57e.

Created on Mon Oct 16
"""

# =============================================================================
# Modules
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from generatetraces import dataset


# =============================================================================
# CREATE DATASETS

nts = 1024
ts = np.arange(0, nts, 1)

ndat_train = 256
ndat_valid = 64
ndat_test = 10

train_ds_tensor = dataset(ts, ndat_train)
valid_ds_tensor = dataset(ts, ndat_valid)
test_ds_tensor = dataset(ts, ndat_test)

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

batch_size = 1
train_loader = torch.utils.data.DataLoader(train_ds_tensor,
                                           batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(valid_ds_tensor,
                                           batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_ds_tensor,
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
            nn.Conv1d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(True)
        )
        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        # Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(16 * 256, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, encoded_space_dim)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):
    """Decoder structure."""

    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, 16 * 256),
            nn.ReLU(True)
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(16, 256))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(16, 8, 3, stride=2, padding=1,
                               output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.tanh(x)
        return x


def add_noise(inputs, noise_factor=0.3):
    """Add noise to traces."""
    noisy = inputs + torch.randn_like(inputs) * noise_factor
    noisy = torch.clip(noisy, -1., 1.)
    return noisy


def train_epoch_den(encoder, decoder, device, dataloader, loss_fn, optimizer,
                    noise_factor=0.3):
    """Training function."""
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader
    for image_batch in dataloader:
        image_noisy = add_noise(image_batch, noise_factor).double()
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        image_noisy = image_noisy.to(device)

        # Encode data
        encoded_data = encoder(image_noisy)
        # Decode data
        decoded_data = decoder(encoded_data)

        # fig = plt.figure()
        # ax = plt.gca()
        # plt.subplots_adjust(left=0.13)
        # plt.plot(ts, np.array(image_noisy)[0, 0, :], ls='-', lw=2,
        #           label='Signal+random noise')
        # plt.plot(ts, np.array(image_batch)[0, 0, :], ls='-', lw=2,
        #           label='Signal')
        # plt.plot(ts, np.array(decoded_data.detach().numpy())[0, 0, :],
        #           ls='-', lw=2, label='Signal rec')
        # ax.tick_params(labelsize=14)
        # # plt.xlabel(r'Simulation number', fontsize=14)
        # # plt.ylabel(r'$\log_{10} (E)$', fontsize=14)
        # plt.legend(frameon=False, fontsize=14)
        # # ax.set_xlim([-10000,10000])
        # # ax.set_ylim([-10000,10000])
        # plt.show()

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


def test_epoch_den(encoder, decoder, device, dataloader, loss_fn,
                   noise_factor=0.3):
    """Testing function."""
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    # No need to track the gradients
    with torch.no_grad():
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch in dataloader:
            # Move tensor to the proper device
            image_noisy = add_noise(image_batch, noise_factor).double()
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
d = 3

# model = Autoencoder(encoded_space_dim=encoded_space_dim)
encoder = Encoder(encoded_space_dim=d, fc2_input_dim=8).double()
decoder = Decoder(encoded_space_dim=d, fc2_input_dim=8).double()
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

# optimtype = torch.optim.Adam(params_to_optimize, lr=lr)
optimtype = torch.optim.NAdam(params_to_optimize, momentum_decay=0.003)
# optimtype = torch.optim.RAdam(params_to_optimize)

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

noise_factor = 0.001
num_epochs = 20
history_da = {'train_loss': [], 'val_loss': []}

for epoch in range(num_epochs):
    # print('EPOCH %d/%d' % (epoch + 1, num_epochs))
    # Training (use the training function)
    train_loss = train_epoch_den(
        encoder=encoder,
        decoder=decoder,
        device=device,
        dataloader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimtype,
        noise_factor=noise_factor)
    # Validation  (use the testing function)
    val_loss = test_epoch_den(
        encoder=encoder,
        decoder=decoder,
        device=device,
        dataloader=valid_loader,
        loss_fn=loss_fn,
        noise_factor=noise_factor)
    # Print Validationloss
    history_da['train_loss'].append(train_loss)
    history_da['val_loss'].append(val_loss)
    print('\n EPOCH {}/{} \t train loss {:.5f} \t val loss {:.5f}'
          . format(epoch + 1, num_epochs, train_loss, val_loss))
    # plot_ae_outputs_den(encoder,decoder,noise_factor=noise_factor)

# =============================================================================
# VISUALIZE LOSS

fig = plt.figure()
ax = plt.gca()
plt.subplots_adjust(left=0.13)
plt.plot(history_da['train_loss'], ls='-', lw=2, label="Training loss")
plt.plot(history_da['val_loss'], ls='-', lw=2, label="Validation loss")
ax.tick_params(labelsize=14)
# plt.xlabel(r'Simulation number', fontsize=14)
# plt.ylabel(r'$\log_{10} (E)$', fontsize=14)
plt.legend(frameon=False, fontsize=14)
# ax.set_xlim([-10000,10000])
# ax.set_ylim([-10000,10000])
plt.show()


# =============================================================================
# VISUALIZE SOME TESTS

for test_sig in test_loader:
    test_sig = torch.reshape(test_sig[0], (1, 1, 1024))
    test_nois = add_noise(test_sig, noise_factor)
    test_nois = test_nois.to(device)
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        rec_sig = decoder(encoder(test_nois))

    fig = plt.figure()
    ax = plt.gca()
    plt.subplots_adjust(left=0.13)
    plt.plot(ts, test_nois[0][0], ls='-', lw=2)
    plt.plot(ts, test_sig[0][0], ls='-', lw=2)
    plt.plot(ts, rec_sig[0][0], ls='-', lw=2)
    ax.tick_params(labelsize=14)
    # plt.xlabel(r'Simulation number', fontsize=14)
    # plt.ylabel(r'$\log_{10} (E)$', fontsize=14)
    # plt.legend(frameon=False, fontsize=14)
    # ax.set_xlim([-10000,10000])
    # ax.set_ylim([-10000,10000])
    plt.show()
    break
