import argparse
import json
import os
import os.path as osp
import pickle as pkl
import time

import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch
import torch.nn as nn
import torch_geometric
from model import ParticleNet
from torch_geometric.loader import DataListLoader, DataLoader
from utils import load_data, make_roc, save_model

plt.style.use(hep.style.CMS)
plt.rcParams.update({"font.size": 20})

matplotlib.use("Agg")

# Ignore divide by 0 errors
np.seterr(divide="ignore", invalid="ignore")

# define the global base device
world_size = torch.cuda.device_count()
multi_gpu = world_size >= 2
if world_size:
    device = torch.device("cuda:0")
    for i in range(world_size):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    torch.backends.cudnn.benchmark = True
else:
    device = "cpu"
    print("Device: CPU")


# define argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="../data/toptagging/", help="dataset path")
parser.add_argument("--outpath", type=str, default="../experiments/", help="output folder")
parser.add_argument("--model_prefix", type=str, default="ParticleNet_model", help="directory to hold the model and plots")
parser.add_argument("--overwrite", dest="overwrite", action="store_true", help="overwrites the model")
parser.add_argument("--n_epochs", type=int, default=3, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--patience", type=int, default=20, help="patience before early stopping")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--nearest", type=int, default=16, help="k nearest neighbors in gravnet layer")
parser.add_argument("--depth", type=int, default=1, help="depth of DNN in each EdgeConv block")
parser.add_argument("--dropout", dest="dropout", action="store_true")
parser.add_argument("--quick", dest="quick", action="store_true", help="perform quick training and testing")

args = parser.parse_args()


@torch.no_grad()
def validation_run(multi_gpu, device, model, loader):
    with torch.no_grad():
        optimizer = None
        ret = train(
            multi_gpu,
            device,
            model,
            loader,
            optimizer,
        )
    return ret


def train(multi_gpu, device, model, loader, optimizer):
    """
    A training/validation run over a given epoch that gets called in the training_loop() function.
    When optimizer is set to None, it freezes the model for a validation_run.
    """

    is_train = not (optimizer is None)

    criterion = nn.BCELoss()
    sig = nn.Sigmoid()

    # initialize loss and time counters
    losses, t = 0, 0

    for batch in loader:
        if multi_gpu:
            batch = batch
        else:
            batch = batch.to(device)

        # run forward pass
        t0 = time.time()
        preds, targets = model(batch)
        t1 = time.time()
        t += t1 - t0

        loss = criterion(sig(preds), targets.reshape(-1, 1).float())

        # backprop
        if is_train:  # not run during a validation run
            for param in model.parameters():
                # better than calling optimizer.zero_grad()
                # according to https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
                param.grad = None
            loss.backward()
            optimizer.step()

        losses = losses + loss.detach()

    print(f"Average inference time per batch is {round((t / len(loader)), 3)}s")

    losses = (losses / (len(loader))).cpu().item()

    return losses


def training_loop(
    multi_gpu,
    device,
    model,
    train_loader,
    valid_loader,
    n_epochs,
    patience,
    optimizer,
    outpath,
):
    """
    Main function to perform training. Will call the train() and validation_run() functions every epoch.

    Args:
        model: a pytorch model wrapped by DistributedDataParallel (DDP)
        train_loader: a pytorch Dataloader that loads .pt files for training when you invoke the get() method
        valid_loader: a pytorch Dataloader that loads .pt files for validation when you invoke the get() method
        patience: number of stale epochs allowed before stopping the training
        optimizer: optimizer to use for training (by default: Adam)
        outpath: path to store the model weights and training plots
    """

    # create directory to hold loss plots
    if not os.path.exists(f"{outpath}/loss_plots/"):
        os.makedirs(f"{outpath}/loss_plots/")

    # create directory to hold the model state at each epoch
    if not os.path.exists(f"{outpath}/epoch_weights/"):
        os.makedirs(f"{outpath}/epoch_weights/")

    t0_initial = time.time()

    losses_train, losses_valid = [], []

    best_val_loss = 99999.9
    stale_epochs = 0

    for epoch in range(n_epochs):
        t0 = time.time()

        if stale_epochs > patience:
            print("breaking due to stale epochs")
            break

        # training step
        print("---->Initiating a training run")
        model.train()
        losses = train(
            multi_gpu,
            device,
            model,
            train_loader,
            optimizer,
        )

        losses_train.append(losses)

        # validation step
        print("---->Initiating a validation run")
        model.eval()
        losses = validation_run(multi_gpu, device, model, valid_loader)

        losses_valid.append(losses)

        # early-stopping
        if losses < best_val_loss:
            best_val_loss = losses
            stale_epochs = 0

            try:
                state_dict = model.module.state_dict()
            except AttributeError:
                state_dict = model.state_dict()
            torch.save(state_dict, f"{outpath}/best_epoch_weights.pth")

            with open(f"{outpath}/best_epoch.json", "w") as fp:  # dump best epoch
                json.dump({"best_epoch": epoch}, fp)
        else:
            stale_epochs += 1

        t1 = time.time()

        epochs_remaining = n_epochs - (epoch + 1)
        time_per_epoch = (t1 - t0_initial) / (epoch + 1)
        eta = epochs_remaining * time_per_epoch / 60

        print(
            f"epoch={epoch + 1} / {n_epochs} "
            + f"train_loss={round(losses_train[epoch], 4)} "
            + f"valid_loss={round(losses_valid[epoch], 4)} "
            + f"stale={stale_epochs} "
            + f"time={round((t1-t0)/60, 2)}m "
            + f"eta={round(eta, 1)}m"
        )

        # save the model's weights
        try:
            state_dict = model.module.state_dict()
        except AttributeError:
            state_dict = model.state_dict()
        torch.save(state_dict, f"{outpath}/epoch_weights/epoch_{epoch+1}_weights.pth")

        # make loss plots
        fig, ax = plt.subplots()
        ax.plot(range(len(losses_train)), losses_train, label="training")
        ax.plot(range(len(losses_valid)), losses_valid, label="validation")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend(loc="best")
        plt.savefig(f"{outpath}/loss_plots/losses_epoch_{epoch}.pdf")
        plt.close(fig)

        with open(f"{outpath}/loss_plots/losses_epoch_{epoch}.pkl", "wb") as f:
            pkl.dump((losses_train, losses_valid), f)

        print("----------------------------------------------------------")
    print(f"Done with training. Total training time is {round((time.time() - t0_initial)/60,3)}min")


if __name__ == "__main__":
    """
    e.g.
    python run_training.py --overwrite --quick --model_prefix='ParticleNet_model' --dataset="toptagging/"

    """

    # setup the input/output dimension of the model
    num_features = 7  # we have 7 input features
    num_classes = 1  # we have one output node

    outpath = osp.join(args.outpath, args.model_prefix)

    if not osp.isdir(outpath):
        os.makedirs(outpath)

    model_kwargs = {
        "for_LRP": False,
        "node_feat_size": num_features,
        "num_classes": num_classes,
        "k": args.nearest,
        "depth": args.depth,
        "dropout": True if args.dropout else False,
    }

    model = ParticleNet(**model_kwargs)

    print(model)
    print(f"Model prefix: {args.model_prefix}")

    # save model_kwargs and hyperparameters
    save_model(
        args,
        outpath,
        model_kwargs,
        model.kernel_sizes,
        model.fc_size,
        model.dropout,
        args.depth,
    )

    # save the weights before training for lrp comparisons
    try:
        state_dict = model.module.state_dict()
    except AttributeError:
        state_dict = model.state_dict()
    torch.save(state_dict, f"{outpath}/before_training_weights.pth")

    # Load the training datafiles
    print("- loading datafiles for training...")
    data_train = load_data(args.dataset, "train", 12, args.quick)
    data_valid = load_data(args.dataset, "val", 4, args.quick)

    # make convenient dataloaders and use DataParallel if multi_gpu is on
    if multi_gpu:
        train_loader = DataListLoader(data_train, batch_size=args.batch_size)
        valid_loader = DataListLoader(data_valid, batch_size=args.batch_size)
        model = torch_geometric.nn.DataParallel(model)
    else:
        train_loader = DataLoader(data_train, batch_size=args.batch_size)
        valid_loader = DataLoader(data_valid, batch_size=args.batch_size)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"- training over {args.n_epochs} epochs")
    training_loop(
        multi_gpu,
        device,
        model,
        train_loader,
        valid_loader,
        args.n_epochs,
        args.patience,
        optimizer,
        outpath,
    )

    # load the best trained model for testing
    with open(f"{outpath}/model_kwargs.pkl", "rb") as f:
        model_kwargs = pkl.load(f)

    state_dict = torch.load(f"{outpath}/best_epoch_weights.pth", map_location=device)

    model = ParticleNet(**model_kwargs)
    model.load_state_dict(state_dict)

    print("- loading datafiles for testing...")
    data_test = load_data(args.dataset, "test", 4, args.quick)

    if multi_gpu:
        test_loader = DataListLoader(data_test, batch_size=args.batch_size, shuffle=True)
        model = torch_geometric.nn.DataParallel(model)
    else:
        test_loader = DataLoader(data_test, batch_size=args.batch_size, shuffle=True)

    model.to(device)
    model.eval()

    print("- making predictions")
    y_score = None
    y_test = None
    for i, batch in enumerate(test_loader):
        if multi_gpu:
            batch = batch
        else:
            batch = batch.to(device)

        preds, targets = model(batch)
        preds = preds.detach().cpu()

        if y_score is None:
            y_score = preds[:].detach().cpu().reshape(-1)
            y_test = targets.detach().cpu()
        else:
            y_score = torch.cat([y_score, preds[:].detach().cpu().reshape(-1)])
            y_test = torch.cat([y_test, targets.detach().cpu()])

    # save the predictions
    print("- saving predictions")
    torch.save(y_test, f"{outpath}/y_test.pt")
    torch.save(y_score, f"{outpath}/y_score.pt")

    # Compute ROC curve
    print("- making Roc curves")
    make_roc(y_test, y_score, f"{outpath}/Roc_curve.pdf")
