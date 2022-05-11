<p align="center">
  <img width="400" src="https://raw.githubusercontent.com/farakiko/xai4hep/master/docs/_static/images/mlpf_rscores.png" />
</p>

# xai4hep

XAI toolbox for interpreting state-of-the-art ML algorithms for high energy physics.

xai4hep provides necessary implementations of explainable AI (XAI) techniques for state-of-the-art graph neural networks (GNNs) developed for various tasks at the LHC-CERN. Models include: machine-learned particle flow (MLPF), the interaction network (IN), ParticleNet. Currently, the layerwise-relevance propagation (LRP) technique is implemented for such models, and additional XAI techniques are under development.


## Setup


## Quickstart

Running standard LRP on a toy dataset with a highly discriminatory feature:

```bash
python lrp_pipeline.py
```

Running modified LRP for a trained MLPF model:

```bash
python mlpf_pipeline.py --run_lrp --make_rmaps --load_model=$model_dir --load_epoch=$epoch --outpath=$path_to_model --loader=$dataloader
```
