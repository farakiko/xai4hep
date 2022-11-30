import os
import os.path as osp
import pickle as pkl
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from explainer import LRP

# this script builds a toy dataset, trains a simple FFN model on the dataset, and tests LRP


class FCN(nn.Module):
    """
    Showcase an example of an fully connected network model, with a skip connection, that can be explained by LRP
    """

    def __init__(self, input_dim=3, hidden_dim=256, embedding_dim=40, output_dim=2):
        super(FCN, self).__init__()

        self.act = nn.ReLU

        self.nn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            self.act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            self.act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            self.act(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.nn2 = nn.Sequential(
            nn.Linear(input_dim + embedding_dim, hidden_dim),
            self.act(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, X):
        embedding = self.nn1(X)
        return self.nn2(torch.cat([X, embedding], axis=1))


def build_toy_dataset():
    print("Building a toy dataset with a highly discriminatory feature (feature #3)")

    # only 2 classes (binary classification)
    class1_y = np.zeros([1000, 1])
    class2_y = np.ones([1000, 1])

    # model should pick feature # 3 as the discriminant feature (more relevant)
    class1_x1 = np.random.uniform(low=0.0, high=1.0, size=1000)
    class2_x1 = np.random.uniform(low=0.0, high=1.0, size=1000)

    class1_x2 = np.random.uniform(low=0.0, high=1.0, size=1000)
    class2_x2 = np.random.uniform(low=0.0, high=1.0, size=1000)

    class1_x3 = np.random.uniform(low=0.0, high=1.0, size=1000)
    class2_x3 = np.random.uniform(low=-1.0, high=0.0, size=1000)

    # concatenate features and classes
    x1_all = np.concatenate([class1_x1, class2_x1]).reshape(-1, 1)
    x2_all = np.concatenate([class1_x2, class2_x2]).reshape(-1, 1)
    x3_all = np.concatenate([class1_x3, class2_x3]).reshape(-1, 1)

    y_all = np.concatenate([class1_y, class2_y]).reshape(-1, 1)
    dataset = np.concatenate([x1_all, x2_all, x3_all, y_all], axis=1)

    np.random.shuffle(dataset)

    # build dataset
    dataset = torch.from_numpy(dataset)

    return dataset


def quick_train(device, model, epochs, dataset, batch_size):
    print("Training a model")

    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    for epoch in range(epochs):
        print(f"Epoch {epoch}")

        model.train()
        for i, batch in enumerate(train_loader):
            X = batch[:, :-1]
            Y = batch[:, -1]

            # Forwardprop
            preds = model(X.float().to(device))

            loss = torch.nn.functional.cross_entropy(preds.float(), Y.long().to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # torch.save(model.state_dict(), "weights.pth")

    Num, Deno = 0, 0
    for i, batch in enumerate(train_loader):
        X = batch[:, :-1]
        Y = batch[:, -1]

        # Forwardprop
        preds = model(X.float().to(device))

        Num = Num + (preds.argmax(axis=1) == Y).sum()
        Deno = Deno + len(preds)
    acc = Num / Deno
    print(f"Accuracy of the model is {round(acc.item(), 3)}")
    return acc.item()


if __name__ == "__main__":
    # Check if the GPU configuration and define the global base device
    if torch.cuda.device_count() > 0:
        print(f"Will use {torch.cuda.device_count()} gpu(s)")
        print("GPU model:", torch.cuda.get_device_name(0))
        device = torch.device("cuda:0")
    else:
        print("Will use cpu")
        device = torch.device("cpu")

    dic = {}
    dic["acc"], dic["Rscores"] = [], []

    # get sample dataset
    dataset = build_toy_dataset()
    samples = dataset[:, :-1].float()

    # train sample model
    model = FCN(input_dim=dataset[:, :-1].shape[-1])
    model.train()
    acc = quick_train(device, model, epochs=1, dataset=dataset, batch_size=4)
    print("------------------------------------------")

    # run lrp on sample model
    model.eval()
    lrp_instance = LRP(device, model, epsilon=1e-9)
    Rscores0 = lrp_instance.explain(samples, neuron_to_explain=0)
    Rscores1 = lrp_instance.explain(samples, neuron_to_explain=1)

    print("------------------------------------------")

    normalized_Rscores = Rscores0.absolute() / Rscores0.absolute().sum(
        axis=1, keepdim=True
    )
    avg_normalized_Rscores = (
        normalized_Rscores.sum(axis=0, keepdim=True) / normalized_Rscores.shape[0]
    )
    print(f"Average normalized Rscores per feature: \n {avg_normalized_Rscores}")
    print("As expected, feature # 3 is the most relevant :)")

    print("------------------------------------------")

    sample = 25
    print("Checking conservation of Rscores for a random sample")
    print("R_input ", Rscores0[sample].sum().item())
    print("R_output", model(samples.to(device))[sample][0].item())
