# xai4hep

XAI toolbox for interpreting state-of-the-art ML algorithms for high energy physics.

xai4hep provides necessary implementation of explainable AI (XAI) techniques for state-of-the-art graph neural networks (GNNs) developed for various tasks at the CERN LHC. Current models include: machine-learned particle flow (MLPF), and ParticleNet. The layerwise-relevance propagation (LRP) technique is implemented for such models, and additional XAI techniques are under development.


## Setup
Have conda installed.
```bash
conda env create -f environment.yml
conda activate xai
```

## Explaining simple FCN
Running LRP to explain a simple fully connected network (FCN) trained on a toy dataset with one highly discriminatory feature:

```bash
python run_lrp_fcn.py
```

## Explaining MLPF

<p align="center">
  <img width="600" src="https://raw.githubusercontent.com/farakiko/xai4hep/dev/docs/_static/images/mlpf_rscores.png" />
</p>

- **Running modified LRP for a trained MLPF model**
```bash
python run_lrp_mlpf.py --run_lrp --make_rmaps --load_model=$model_dir --load_epoch=$epoch
```

## Explaining ParticleNet

<p align="center">
  <img width="500" src="https://raw.githubusercontent.com/farakiko/xai4hep/dev/docs/_static/images/particlenet_rscores.png" />
</p>


### Quickstart
- **Get and process the Top tagging dataset**
```bash
cd particlenet/
./get_data.sh
```
This will automatically create a `data/` folder under the `xai4hep/` repository, with a `toptagging/` folder that contains `train/`,`val/`,`test/` folders; each containing a respective subset of the dataset.

- **Run a quick training**

From the `xai4hep/` repository run
```bash
mkdir experiments/
cd particlenet
python run_training.py --quick --model_prefix=ParticleNet_model
```
This will run a quick training over a small sample of the dataset and store the model under `experiments/`.

- **Run a quick LRP test**

From the `xai4hep/` repository run
```bash
python run_lrp_particlenet.py --quick --model_prefix=ParticleNet_model --run_lrp --make_dr_Mij_plots --scaling_up
```
