
import pandas as pd
import torch
import numpy as np
import sys
sys.path += ["../"]

from src.utils import dum_utils as dumu
from src.modules import dum as dumm

import logging
import coloredlogs
import argparse
import os

from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

coloredlogs.install(
    level='DEBUG',
    fmt="%(asctime)s %(name)s] [%(levelname)s] [%(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
)
logger = logging.getLogger("AL demand forecast MLP1")


def str2bool(s):
    if s.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif s.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



class CustomDataset(Dataset):
    def __init__(self, X, y):
        
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        image = self.X[idx]
        label = self.y[idx]

        return image, label




# train_dataset = CustomDataset(X_train, Y_train)
# test_dataset = CustomDataset(X_test, Y_test)


# step 1 load the data

path_data = '/Users/ab212678/Documents/GRAND/sims/claire_sims/data_v2'

ef_p = torch.load(os.path.join(path_data, 'efield_traces_p.pt')).float() 
ef_fe = torch.load(os.path.join(path_data, 'efield_traces_Fe.pt')).float()

#ef_p += 10
n_p = ef_p.shape[0]
n_fe = ef_fe.shape[0]

y_p = torch.ones(n_p).float()
y_fe = torch.zeros(n_fe).float()


X = torch.vstack([ef_p, ef_fe])
Y = torch.hstack([y_p, y_fe])


#X = X.unsqueeze(dim=1)
X = X.transpose(2, 1)

n_total = n_p + n_fe
perm = np.random.permutation(n_total)

X = X[perm]
Y = Y[perm]

train_test_fraction = 0.7

train_id = perm[:int(n_total*train_test_fraction)]
test_id = perm[int(n_total*train_test_fraction):]

X_test = X[test_id]
Y_test = Y[test_id]

X_train = X[train_id]
Y_train = Y[train_id]



train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)




def run_model_on_dataloader(dataloader, model, device):
    model.eval()
    preds = []
    ys = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            preds.append(pred)
            ys.append(y)
           

    preds = torch.squeeze(torch.stack(preds))
    ys = torch.squeeze(torch.stack(ys))
    return preds, ys


class classif_model1(nn.Module):
    def __init__(self):
        super(classif_model1, self).__init__()
        self.conv1 = nn.Conv1d(3, 32, kernel_size=5)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32, 8, kernel_size=5)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(1976, 50)
        self.fc2 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)

        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.dropout(x, training=self.training)
        x = x.view(-1, 1976)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
        #return F.log_softmax(x)
        


train_dataloader = DataLoader(
    train_dataset,
    batch_size=10)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=10)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = classif_model1().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-3)

scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=5e-5,
    max_lr=1e-3,
    step_size_up=5000,
    mode='triangular2'
    )


loss_list = []
test_mae = []
lr = []
#  train
train_accuracy = []
test_accuracy = []
loss_fn = nn.BCELoss()





model.train()


for i in range(0, 300):
    loss_per_epoch = []
    print("epoch = {}".format(i))
    for batch, (X_, y_) in enumerate(train_dataloader):
        #print("batch = {}".format(batch))
        X_ = X_.to(device)
        #X_ = X_.transpose(2, 1)
        y_ = torch.unsqueeze(y_, dim=1)
        #y_ = y_.type(torch.LongTensor).to(device)
        
        pred = model(X_)

        loss = loss_fn(pred, y_)  # / y.abs()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        lr.append(scheduler.get_last_lr())

        loss_ = loss.item()
        #print(loss_)
        loss_per_epoch.append(loss_)
    if i % 5 == 0:

        a, b = run_model_on_dataloader(test_dataloader, model, device)
        c, d = run_model_on_dataloader(train_dataloader, model, device)


        aa = a.view(-1, 1)
        aa[aa>0.5] = 1
        aa[aa<=0.5] = 0
        bb = b.view(-1, 1)
        cc = c.view(-1, 1)
        cc[cc>0.5] = 1
        cc[cc<=0.5] = 0

        dd = d.view(-1, 1)

        test_acc = (aa[:, 0] == bb[:, 0]).sum().numpy()/aa.shape[0]
        train_acc = (cc[:, 0] == dd[:, 0]).sum().numpy()/cc.shape[0]
        print('test_acc = {}'.format(test_acc))
        print('train_acc = {}'.format(train_acc))
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)

    loss_mean = np.array(loss_per_epoch).mean()
    print(loss_mean)
    print(lr[-1])
    loss_list.append(loss_mean)




a, b = run_model_on_dataloader(test_dataloader, model, device)
c, d = run_model_on_dataloader(train_dataloader, model, device)


aa = a.view(-1, 1)
aa[aa>0.5] = 1
aa[aa<=0.5] = 0
bb = b.view(-1, 1)
cc = c.view(-1, 1)
cc[cc>0.5] = 1
cc[cc<=0.5] = 0

dd = d.view(-1, 1)

test_acc = (aa[:, 0] == bb[:, 0]).sum().numpy()/aa.shape[0]
train_cc = (cc[:, 0] == dd[:, 0]).sum().numpy()/cc.shape[0]





if False:

    def parse_args():
        ap = argparse.ArgumentParser(description='train a simple mlp')
        ap.add_argument(
            "--file_path",
            type=str,
            required=True,
            help="AL parquet file path"
        )
        ap.add_argument(
            "--output_path",
            type=str,
            help="results path"
        )
        ap.add_argument(
            "--n_units",
            type=int,
            default=10,
            help="number if units per layer"
        )
        ap.add_argument(
            "--n_epoch",
            type=int,
            default=10,
            help="number of epochs"
        )
        ap.add_argument(
            "--onnx-export",
            help="Export model in ONNX format.",
            dest='onnx_export',
            action='store_true'
        )
        args = ap.parse_args()
        return args


    if __name__ == "__main__":
        args = parse_args()

        os.makedirs(args.output_path, exist_ok=True)

        dg = pd.read_parquet(args.file_path)

        cat_0, cat_1, cat_2 = cd.get_new_categories()
        dg = cd.rename_categorical_content(dg, cat_0, cat_1, cat_2)
        triplets = cd.get_existing_category_triplet(dg, cat_0, cat_1, cat_2)
        dg_, triplets_n0 = cd.normalize_bigdg(dg, triplets)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(device))

        tfile = os.path.join(
            args.output_path,
            'CAI_AL_MLP_U{units:d}_E{epochs:d}'.format(
                units=args.n_units,
                epochs=args.n_epoch) + '_{suffix}.{ext}'
            )

        mlp_models.train_mlp_model(
            dg_,
            device,
            args.n_epoch,
            args.n_units,
            args.output_path,
            tfile,
            onnx_export=args.onnx_export)
