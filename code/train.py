# do random seeding
import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import FinCLData
import argparse
from model import *
import pickle
import torch.optim as optim
import copy
import time
import os
import random
import optimizers
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score

from tqdm import tqdm
from empyrical import sharpe_ratio, calmar_ratio
import pandas as pd
df = pd.DataFrame()

start_time = time.strftime("%Y%m%d-%H%M%S")
print(start_time)
parser = argparse.ArgumentParser(description="HTLSTM Model")

parser.add_argument(
    "--lr",
    default=0.0005,
    type=float,
    help="Learning rate to use for training (default: 0.0005)",
)
parser.add_argument(
    "--num_epochs",
    default=500,
    type=int,
    help="Number of epochs to run for training (default: 500)",
)
parser.add_argument(
    "--seed",
    default=2020,
    type=int,
    help="Seed for experiment (default: 2020)",
)
parser.add_argument(
    "--decay",
    default=1e-5,
    type=float,
    help="Weight decay to use for training (default: 1e-5)",
)
parser.add_argument(
    "--batch_size",
    default=128,
    type=int,
    help="Batch Size use for training the model (default: 128)",
)
parser.add_argument(
    "--task",
    default="movement",
    help="Task on which to train model [movement, volatility] (default: movement)",
)

parser.add_argument(
    "--data",
    default="stock",
    help="data to be used [stock, china] (default: stock)",
)

parser.add_argument(
    "--name",
    default="htlstm_run_1",
    type=str,
    help="exp_name",
)
args = parser.parse_args()

# torch.manual_seed(args.seed)
# np.random.seed(args.seed)
# random.seed(args.seed)
# not doing seeding on CUDA
device = torch.device("cuda")

if args.data == "stock":
    traindata = FinCLData("../data/train_data.pkl")
    valdata = FinCLData("../data/test_data.pkl")
elif args.data == "china":
    traindata = FinCLData("../data/train_data_chinese.pkl")
    valdata = FinCLData("../data/test_data_chinese.pkl")

trainloader = torch.utils.data.DataLoader(
    traindata, batch_size=args.batch_size, shuffle=True, num_workers=8
)
valloader = torch.utils.data.DataLoader(
    valdata, batch_size=args.batch_size, shuffle=True, num_workers=8
)


if args.data == "stock":
    testdata = FinCLData("../data/test_data.pkl")
elif args.data == "china":
    testdata = FinCLData("../data/test_data_chinese.pkl")

if args.data == "stock":
    testdata_1 = FinCLData("../data/testdata_stocknet_price.pkl")
elif args.data == "china":
    testdata_1 = FinCLData("../data/testdata_china_price.pkl")

# print ('Train Data: ', len())

dataloaders = {"train": trainloader, "val": valloader}

if args.task == "movement":
    criterion = nn.CrossEntropyLoss()
    criterion_2 = nn.SoftMarginLoss()
elif args.task == "volatility":
    criterion = nn.MSELoss()

if os.path.exists("saved_models") == False:
    os.mkdir("saved_models")

loss_history = {"train": [], "val": []}
accuracy_history = {"train": [], "val": []}
mcc_history = {"train": [], "val": []}
f1_history = {"train": [], "val": []}


since = time.time()
best_loss = 9999999
best_epoch = 0
best_mcc = -1
best_acc = 0
best_f1 = -1
best_mcc_when_acc = 0
best_f1_when_acc = 0
best_acc_when_mcc = 0
best_f1_when_mcc = 0
best_acc_when_f1 = 0
best_mcc_when_f1 = 0

if args.task == "movement":
    outdim = 2
elif args.task == "volatility":
    outdim = 1

if args.data == "stock":
    model = HTLSTM(
        text_embed_dim=768,
        intraday_hiddenDim=256,
        interday_hiddenDim=256,
        intraday_numLayers=1,
        interday_numLayers=1,
        maxlen=30,
        outdim=outdim,
        device=device
    )
elif args.data == "china":
    print("training on china data")
    model = HTLSTM(
        text_embed_dim=768,
        intraday_hiddenDim=256,
        interday_hiddenDim=256,
        intraday_numLayers=1,
        interday_numLayers=1,
        maxlen=10,
        outdim=outdim,
        device=device
    )
model.to(device)
optimizer = optimizers.RiemannianAdam(model.parameters(), lr=args.lr, weight_decay=args.decay)


lrs = [5e-4]

for epoch in tqdm(range(args.num_epochs)):
    print("start_time: ", start_time)
    print("Epoch {}/{}".format(epoch, args.num_epochs - 1))
    print("-" * 10)
    for g in optimizer.param_groups:
        g['lr'] = lrs[0]
    print ('Learning rate: ', curr_lr)

    # Each epoch has a training and validation phase
    for phase in ["train", "val"]:
        if phase == "train":
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        running_loss = 0.0

        truelabels = []
        predlabels = []
        # Iterate over data.
        for batch_data in dataloaders[phase]:
            embedding_data = batch_data["embedding"]
            # print ('Embedding data: ', embedding_data.shape)
            # embedding_data = embedding_data.type(torch.DoubleTensor).to(device)
            embedding_data = embedding_data.to(device)
            
            if args.task == "movement":
              truelabels.extend(batch_data["movement_label"].numpy())
              target = batch_data["movement_label"]
              target = target.type(torch.LongTensor).to(device)
            elif args.task == "volatility":
              target = batch_data["volatility"]
              target[torch.isnan(target)] = 0
              target[torch.isinf(target)] = 0
              target = target.type(torch.FloatTensor).to(
                  device).unsqueeze(-1)

            length = batch_data["length_data"]

            time_feats = batch_data["time_feature"].to(device).squeeze(-1)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(phase == "train"):
                outputs, margin_output = model(embedding_data, length, time_feats)

                # print ('Outputs: ', outputs.shape)
                # print ('Targets: ', target.shape)
                cse_loss = criterion(outputs, target)
                if args.task == "movement":
                    margin_target = target.detach().clone()
                    margin_target[margin_target==0] = -1
                    margin_loss = criterion_2(margin_output.squeeze(), margin_target)
                    loss = cse_loss + 0.3*margin_loss
                else:
                    loss = cse_loss
                
                # backward + optimize only if in training phase
                if phase == "train":
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * target.size(0)
            if args.task == "movement":           
              predlabels.extend(torch.argmax(
                  outputs, dim=-1).cpu().detach().numpy())

        if args.task == "movement":
          predlabels = np.array(predlabels)
          truelabels = np.array(truelabels)

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        loss_history[phase].append(epoch_loss)
        
        if args.task == "movement":
          epoch_accuracy = accuracy_score(truelabels, predlabels)
        #   print ('Pred labels: ', predlabels)
          epoch_f1 = f1_score(truelabels, predlabels, labels=[0,1])
          epoch_mcc = matthews_corrcoef(truelabels, predlabels)
          accuracy_history[phase].append(epoch_accuracy)
          mcc_history[phase].append(epoch_mcc)
          f1_history[phase].append(epoch_f1)

        if args.task == "movement":

          if phase == "val" and epoch_accuracy > best_acc:
              best_acc = epoch_accuracy
              best_epoch = epoch
              best_mcc_when_acc = epoch_mcc
              best_f1_when_acc = epoch_f1
              best_loss = epoch_loss
              best_model_wts_epoch = copy.deepcopy(model.state_dict())

          if phase == "val" and epoch_mcc > best_mcc:
              best_mcc = epoch_mcc
              best_acc_when_mcc = epoch_accuracy
              best_f1_when_mcc = epoch_f1
              best_model_wts_mcc = copy.deepcopy(model.state_dict())
          
          if phase == "val" and epoch_f1 > best_f1:
              best_f1 = epoch_f1
              best_acc_when_f1 = epoch_accuracy
              best_mcc_when_f1 = epoch_mcc
              best_model_wts_f1 = copy.deepcopy(model.state_dict())
              

          if phase == "val":
              torch.save(
                  {
                      "model_wts": model.state_dict(),
                      "current_epoch": epoch,
                      "loss_history": loss_history,
                      "mcc_history": mcc_history,
                      "accuracy_history": accuracy_history,
                      "args": args,
                  },
                  "../saved_models/" + str(args.name) + '_' + str(args.data) + '_'+ str(args.num_epochs) + ".pth",
              )
          print(
              "{} Epoch: {} Acc: {:.4f} MCC: {:.4f} F1: {:.4f} Loss: {:.4f} ".format(
                  phase, epoch, epoch_accuracy, epoch_mcc, epoch_f1, epoch_loss,
              )
          )

        elif args.task == "volatility":

          if phase == "val" and epoch_loss < best_loss:
              best_loss = epoch_loss
              best_epoch = epoch
              best_model_wts = copy.deepcopy(model.state_dict())
              

          if phase == "val":
              torch.save(
                  {
                      "model_wts": model.state_dict(),
                      "current_epoch": epoch,
                      "loss_history": loss_history,
                      "best_model_wts": best_model_wts,
                      "args": args,
                  },
                  "../saved_models/" + str(args.name) + '_' + str(args.data) + '_'+ str(args.num_epochs) +'_'+ str(start_time) + ".pth",
              )

          print(
              "{} Epoch: {} Loss: {:.4f} ".format(
                  phase, epoch, epoch_loss,
              )
          )

time_elapsed = time.time() - since
print(
    "Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60)
)
print("\n", start_time)
