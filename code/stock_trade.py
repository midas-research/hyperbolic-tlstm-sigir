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
from sklearn.metrics import matthews_corrcoef, accuracy_score
from empyrical import downside_risk, sortino_ratio
import pandas as pd
df = pd.DataFrame()

start_time = time.strftime("%Y%m%d-%H%M%S")
print(start_time)
parser = argparse.ArgumentParser(description="HTLSTM Stock Trading")

parser.add_argument(
    "--model_path",
    type=str,
    help="Path to saved model checkpoint",
)

parser.add_argument(
    "--data",
    default="stock",
    help="data to be used [stock, china] (default: stock)",
)


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# trading data
testdata = FinCLData("../data/test_data.pkl")
# target price data
testdata_1 = FinCLData("../data/testdata_stocknet_price.pkl")

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

model_path = (
    args.model_path  # without curr
)

model_price.load_state_dict(
    torch.load(model_price_path)["model_wts"]
)
model_price.eval()
model_price.to(device)

trading_threshold = 0.6

with torch.no_grad():
    print ('Threshold: ', trading_threshold)
    profits = []
    for i in tqdm(range(len(testdata))):
        embedding = testdata[i]["embedding"].to(device).unsqueeze(0)
        length_data = testdata[i]["length_data"].to(device).unsqueeze(0)
        time_feats = testdata[i]["time_feature"].to(device).squeeze(-1).unsqueeze(0)

        cp_last = testdata_1[i]["cp_last"]
        cp_target = testdata_1[i]["cp_target"]

        outputs_price, _ = model_price(embedding, length_data, time_feats)
        outputs_probs = torch.nn.functional.softmax(outputs_price, dim=-1).squeeze(0)
        if (
            torch.max(outputs_probs) > trading_threshold
        ):
            if torch.argmax(outputs_price) == 1:
                profits.append(cp_target - cp_last)
            else:
                profits.append(cp_last - cp_target)

    profits = np.array(profits)
    sortino = sortino_ratio(profits)
    downside = downside_risk(profits)
    print ("Downside Risk: ", downside)
    print ("Sortino Ratio: ", sortino)
