[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4559587.svg)](https://doi.org/10.5281/zenodo.7266537)

# xai4hep

Code for:

[1] Farouk Mokhtar et. al., *Do graph neural networks learn traditional jet substructure?*, [ML4PS @ NeurIPS 2022](https://ml4physicalsciences.github.io/2022/) [`arXiv:2211.09912`](https://arxiv.org/abs/2211.09912) \
[2] Farouk Mokhtar et. al., *Explaining machine‑learned particle‑flow reconstruction*, [ML4PS @ NeurIPS 2021](https://ml4physicalsciences.github.io/2021/) [`arXiv:2111.12840`](https://arxiv.org/abs/2111.12840)


## Overview
XAI toolbox for interpreting state-of-the-art ML algorithms for high energy physics.

xai4hep provides necessary implementation of explainable AI (XAI) techniques for state-of-the-art graph neural networks (GNNs) developed for various tasks at the CERN LHC. Current models include: machine-learned particle flow (MLPF), and ParticleNet. The layerwise-relevance propagation (LRP) technique is implemented for such models, and additional XAI techniques are under development.

### Explaining ParticleNet using LRP will produce the following edge-R-graphs.
<figure>
<img src="https://raw.githubusercontent.com/farakiko/xai4hep/main/docs/_static/images/rgraphs.png" alt="Trulli" style="width:100%">
<figcaption align = "center">Fig.1 - The jet constituents are represented as nodes in (eta, phi) space with interconnections as edges, whose intensities correspond to the connection's edge R score. Each node's intensity corresponds to the relative p<sub>T</sub> of the corresponding particle. Constituents belonging to the three different CA subjets are shown in blue, red, and green in descending p<sub>T</sub> order. We observe that by the last EdgeConv block the model learns to rely more on edge connections between the different subjets.</figcaption>
</figure>

<br/>

### Explaining MLPF using LRP will produce the following R-maps.
<figure>
<img src="https://raw.githubusercontent.com/farakiko/xai4hep/main/docs/_static/images/rmaps.png" alt="Trulli" style="width:100%">
<figcaption align = "center">Fig.2 - This figure constitutes averaged R-maps for elements associated to charged hadrons (top), and neutral hadrons (bottom). We see that charged hadrons use more neighbor information than neutral hadrons.</figcaption>
</figure>

## Setup
We recommend using the `requirements.txt` file then installing `xai4hep` as a module by running
```
pip install .
```

Other ways to setup,

1. If you have access to the kubernetes [PRP Nautlius cluster](https://nautilus.optiputer.net/), then refer to this gitlab repo for the setup https://gitlab.nrp-nautilus.io/fmokhtar/xai4hep

2. Using docker
```bash
docker build docker/
```
