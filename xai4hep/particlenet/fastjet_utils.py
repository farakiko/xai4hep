import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import torch


def get_subjets(px, py, pz, e, N_SUBJETS=3, JET_ALGO="CA", jet_radius=0.8):
    """
    Declusters a jet into exactly N_SUBJETS using the JET_ALGO and jet_radius provided.

    Args:
        px (np.ndarray): NumPy array of shape ``[num_particles]`` containing the px of each particle inside the jet.
        py (np.ndarray): NumPy array of shape ``[num_particles]`` containing the py of each particle inside the jet.
        pz (np.ndarray): NumPy array of shape ``[num_particles]`` containing the pz of each particle inside the jet.
        e (np.ndarray): NumPy array of shape ``[num_particles]`` containing the e of each particle inside the jet.
        N_SUBJETS (int): Number of subjets to decluster the jet into.
            (default is 3)
        JET_ALGO (str): The jet declustering algorithm to use. Choices are ["CA", "kt", "antikt"].
            (default is "CA")
        jet_radius (float): The jet radius to use when declustering.
            (default is 0.8)

    Returns:
        subjet_idx (np.array): NumPy array of shape ``[num_particles]`` with elements
                                representing which subjet the particle belongs to.
        subjet_vectors (list): includes bjet information (e.g. px, py, pz)

    """
    import awkward as ak
    import fastjet
    import vector

    if JET_ALGO == "kt":
        JET_ALGO = fastjet.kt_algorithm
    elif JET_ALGO == "antikt":
        JET_ALGO = fastjet.antikt_algorithm
    elif JET_ALGO == "CA":
        JET_ALGO = fastjet.cambridge_algorithm

    jetdef = fastjet.JetDefinition(JET_ALGO, jet_radius)

    # define jet directly not an array of jets
    jet = ak.zip(
        {
            "px": px,
            "py": py,
            "pz": pz,
            "E": e,
        },
        with_name="Momentum4D",
    )

    pseudojet = [
        fastjet.PseudoJet(particle.px.item(), particle.py.item(), particle.pz.item(), particle.E.item()) for particle in jet
    ]

    cluster = fastjet.ClusterSequence(pseudojet, jetdef)

    # cluster jets
    jets = cluster.inclusive_jets()
    assert len(jets) == 1

    # get the 3 exclusive jets
    subjets = cluster.exclusive_subjets(jets[0], N_SUBJETS)
    assert len(subjets) == N_SUBJETS

    # sort by pt
    subjets = sorted(subjets, key=lambda x: x.pt(), reverse=True)

    # define a subjet_idx placeholder
    subjet_idx = ak.zeros_like(px, dtype=int) - 1
    mapping = subjet_idx.to_list()

    subjet_indices = []
    for subjet_idx, subjet in enumerate(subjets):
        subjet_indices.append([])
        for subjet_const in subjet.constituents():
            for idx, jet_const in enumerate(pseudojet):
                if (
                    subjet_const.px() == jet_const.px()
                    and subjet_const.py() == jet_const.py()
                    and subjet_const.pz() == jet_const.pz()
                    and subjet_const.E() == jet_const.E()
                ):
                    subjet_indices[-1].append(idx)

    for subjet_idx, subjet in enumerate(subjets):
        local_mapping = np.array(mapping)
        local_mapping[subjet_indices[subjet_idx]] = subjet_idx
        mapping = local_mapping

    # add the jet index
    jet["subjet_idx"] = ak.Array(mapping)

    subjet_vectors = [
        vector.obj(
            px=ak.sum(jet.px[jet.subjet_idx == j], axis=-1),
            py=ak.sum(jet.py[jet.subjet_idx == j], axis=-1),
            pz=ak.sum(jet.pz[jet.subjet_idx == j], axis=-1),
            E=ak.sum(jet.E[jet.subjet_idx == j], axis=-1),
        )
        for j in range(0, N_SUBJETS)
    ]

    return jet["subjet_idx"].to_numpy(), subjet_vectors


def scaling_up(outpath, epoch, N_values=15, N_SUBJETS=3, JET_ALGO="CA", jet_radius=0.8):
    """
    Computes the distribution of edges connecting different subjets for different values of N.

    Args:
        outpath (str): Path to load the Rscores pkl files.
        epoch (int): The epoch at which to load the model.
            (if -1: best trained model, if 0: untrained model, otherwise loads the corresponding epoch)
        N_values (int): The different values of N to scan, which will be taken as range(N_values).
            (default is 15)
        N_SUBJETS (int): Number of subjets to decluster the jet into.
            (default is 3)
        JET_ALGO (str): The jet declustering algorithm to use. Choices are ["CA", "kt", "antikt"].
            (default is "CA")
        jet_radius (float): The jet radius to use when declustering.
            (default is 0.8)

    """

    # same: list of counters, for each N, of edges connecting the same subjet
    top_same, qcd_same = np.array([0] * N_values), np.array([0] * N_values)
    # diff: list of counters, for each N, of edges connecting different subjets
    top_diff, qcd_diff = np.array([0] * N_values), np.array([0] * N_values)

    if epoch == -1:
        PATH = f"{outpath}/Rscores_best"
        save_as = "best"
        legend_title = "Trained model"
    elif epoch == 0:
        PATH = f"{outpath}/Rscores_untrained"
        save_as = "untrained"
        legend_title = "Untrained model"
    else:
        PATH = f"{outpath}/Rscores_epoch_{epoch}"
        save_as = f"epoch_{epoch}"
        legend_title = f"Model at epoch {epoch}"

    # load the jet information
    # with open(f"{PATH}/batch_x.pkl", "rb") as handle:
    #     batch_x_list = pkl.load(handle)
    with open(f"{PATH}/batch_y.pkl", "rb") as handle:
        batch_y_list = pkl.load(handle)
    with open(f"{PATH}/batch_p4.pkl", "rb") as handle:
        batch_p4_list = pkl.load(handle)

    # load the edgeRscores and edge_index of each EdgeConv block
    with open(f"{PATH}/R_edges.pkl", "rb") as handle:
        R_edges_list = pkl.load(handle)
    with open(f"{PATH}/edge_index.pkl", "rb") as handle:
        edge_index_list = pkl.load(handle)

    Num_jets = len(batch_p4_list)
    print(f"Total # of jets is {Num_jets}")
    for i in range(Num_jets):
        # define the jet information
        jet_label = batch_y_list[i]

        # eta = batch_x_list[i][:, 0]
        # phi = batch_x_list[i][:, 1]

        px = batch_p4_list[i][:, 0]
        py = batch_p4_list[i][:, 1]
        pz = batch_p4_list[i][:, 2]
        e = batch_p4_list[i][:, 3]

        # define the edgeRscores and the edge_index of the last EdgeConv block
        edge_Rscores = R_edges_list[i]["edge_conv_2"]
        edge_index = edge_index_list[i]["edge_conv_2"]

        # get subjets
        # try:
        print(f"- Declustering jet # {i} using {JET_ALGO} algorithm")
        subjet_idx, _ = get_subjets(px, py, pz, e, N_SUBJETS, JET_ALGO, jet_radius)
        # except Exception:
        #     print(f"skipping jet # {i}")
        #     continue

        for N in range(N_values):
            # N=0 doesn't make sense here
            for edge in torch.topk(edge_Rscores, N + 1).indices:
                if jet_label == 1:
                    if subjet_idx[edge_index[0][edge]] != subjet_idx[edge_index[1][edge]]:
                        top_diff[N] += 1
                    else:
                        top_same[N] += 1
                else:
                    if subjet_idx[edge_index[0][edge]] != subjet_idx[edge_index[1][edge]]:
                        qcd_diff[N] += 1
                    else:
                        qcd_same[N] += 1

    top_fraction = top_diff / (top_same + top_diff)
    qcd_fraction = qcd_diff / (qcd_same + qcd_diff)

    outpath = f"{outpath}/scaling_up"
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    with open(f"{outpath}/top_fraction_{save_as}.pkl", "wb") as f:
        pkl.dump(top_fraction, f)
    with open(f"{outpath}/qcd_fraction_{save_as}.pkl", "wb") as f:
        pkl.dump(qcd_fraction, f)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(range(len(top_fraction)), top_fraction, label="Top")
    ax.plot(range(len(qcd_fraction)), qcd_fraction, label="QCD")
    ax.legend(title=legend_title)
    ax.set_xlabel(r"$N_{edges}$", fontsize=20)
    ax.set_ylabel(r"$N_{edges \ between \ subjets}$ / $N_{edges}$", fontsize=20)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    plt.savefig(f"{outpath}/scaling_up_{save_as}.pdf")
    print(f"saved the plot as {outpath}/scaling_up_{save_as}.pdf")
