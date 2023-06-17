import argparse
import json
import os
import os.path as osp
import pickle as pkl
import time
import warnings

import torch
from explainer import LRP_ParticleNet
from particlenet import ParticleNet, load_data, make_dr_Mij_plots, scaling_up
from torch_geometric.loader import DataLoader

warnings.filterwarnings("ignore")

# Check if the GPU configuration and define the global base device
if torch.cuda.device_count() > 0:
    print("Will use GPU model:", torch.cuda.get_device_name(0))
    device = torch.device("cuda:0")
else:
    print("Will use CPU")
    device = torch.device("cpu")

# this script runs lrp on a trained ParticleNet model

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="./data/toptagging/", help="path to datafile")
parser.add_argument("--outpath", type=str, default="./experiments/", help="path to the trained model directory")
parser.add_argument("--model_prefix", type=str, default="ParticleNet_dropout1", help="Which model to load")
parser.add_argument("--epoch", type=int, default=-1, help="epoch to run Rscores for (-1=best_epoch, 0=untrained)")
parser.add_argument("--quick", dest="quick", action="store_true")
parser.add_argument("--run_lrp", dest="run_lrp", action="store_true")
parser.add_argument("--make_dr_Mij_plots", dest="make_dr_Mij_plots", action="store_true")
parser.add_argument("--scaling_up", dest="scaling_up", action="store_true")

args = parser.parse_args()


if __name__ == "__main__":
    outpath = osp.join(args.outpath, args.model_prefix)

    if not os.path.exists(f"{outpath}/xai"):
        os.makedirs(f"{outpath}/xai")

    # run lrp pipeline to compute the Rscores
    if args.run_lrp:
        print(f"Runing the LRP pipeline to compute the Rscores for the model at epoch {args.epoch}")
        # load the testing data
        print("- loading datafiles for lrp studies...")
        data_test = load_data(args.dataset, "test", 4, args.quick)
        loader = DataLoader(data_test, batch_size=1, shuffle=True)

        # load a pretrained model
        with open(f"{outpath}/model_kwargs.pkl", "rb") as f:
            model_kwargs = pkl.load(f)

        if args.epoch == -1:  # load the best trained model
            state_dict = torch.load(f"{outpath}/best_epoch_weights.pth", map_location=device)
            PATH = f"{outpath}/xai/Rscores_best/"
        elif args.epoch == 0:  # load the untrained model
            state_dict = torch.load(f"{outpath}/before_training_weights.pth", map_location=device)
            PATH = f"{outpath}/xai/Rscores_untrained/"
        else:  # load a specefic epoch of the trained model
            state_dict = torch.load(f"{outpath}/epoch_weights/epoch_{args.epoch}_weights.pth", map_location=device)
            PATH = f"{outpath}/xai/Rscores_epoch_{args.epoch}/"

        # the following line will make it possible to retrieve intermediate activations
        model_kwargs["for_LRP"] = True
        # model_kwargs["depth"] = 1  # TODO: remove
        # instantiate a ParticleNet model with the loaded configuration
        model = ParticleNet(**model_kwargs)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        print(model)

        # make directory to hold the edgeRscores
        if not os.path.exists(PATH):
            os.makedirs(PATH)

        # initilize an lrp instance
        lrp = LRP_ParticleNet(device=device, model=model, epsilon=1e-8)

        # initilize placeholders to store the input, target, and p4 for each jet
        batch_x_list, batch_y_list, batch_p4_list = [], [], []
        # initilize placeholders to hold the edgeRscores and edge_index of each EdgeConv block for each jet
        R_edges_list, edge_index_list = [], []

        ti = time.time()
        for i, jet in enumerate(loader):
            if i == 5 and args.quick:
                break

            if i == 1000:
                break

            print(f"Explaining jet # {i}: {jet}")

            # explain a single jet
            try:
                R_edges, edge_index = lrp.explain(jet.to(device))
            except Exception:
                print("jet is not processed correctly so skipping it")
                continue

            batch_x_list.append(jet.x.detach().cpu())
            batch_y_list.append(jet.y.detach().cpu())

            # for fast jet, store the p4 information
            batch_p4_list.append(
                torch.cat(
                    [
                        jet.px.detach().cpu().unsqueeze(1),
                        jet.py.detach().cpu().unsqueeze(1),
                        jet.pz.detach().cpu().unsqueeze(1),
                        jet.E.detach().cpu().unsqueeze(1),
                    ],
                    axis=1,
                )
            )

            R_edges_list.append(R_edges)
            edge_index_list.append(edge_index)

            print("------------------------------------------------------")

        tf = time.time()

        with open(f"{PATH}/time.json", "w") as fp:  # dump time
            json.dump(
                {
                    "time_min": round((tf - ti) / 60, 3),
                    "time_hours": round((tf - ti) / 3600, 3),
                },
                fp,
            )

        # store the lists containing the jet information
        with open(f"{PATH}/batch_x.pkl", "wb") as handle:
            pkl.dump(batch_x_list, handle)
        with open(f"{PATH}/batch_y.pkl", "wb") as handle:
            pkl.dump(batch_y_list, handle)
        with open(f"{PATH}/batch_p4.pkl", "wb") as handle:
            pkl.dump(batch_p4_list, handle)

        # store the lists containing the edgeRscores and edge_index of each EdgeConv block for each jet
        with open(f"{PATH}/R_edges.pkl", "wb") as handle:
            pkl.dump(R_edges_list, handle)
        with open(f"{PATH}/edge_index.pkl", "wb") as handle:
            pkl.dump(edge_index_list, handle)

        print(f"Finished computing the Rscores for the model at epoch {args.epoch}")

    # produce the lrp deltaR and Mij result plots
    if args.make_dr_Mij_plots:
        print(f"Computing the deltaR and invariant mass distributions of the most relevant edges at epoch {args.epoch}")
        make_dr_Mij_plots(f"{outpath}/xai", epoch=args.epoch, Top_N=5)

    # produce the lrp scaling_up result plots
    if args.scaling_up:
        print(f"Computing the fraction of relevant edges connecting different subjets for the model at epoch {args.epoch}")
        scaling_up(
            f"{outpath}/xai",
            epoch=args.epoch,
            N_values=15,
            N_SUBJETS=3,
            JET_ALGO="CA",
            jet_radius=0.8,
        )
