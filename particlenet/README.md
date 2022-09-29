# Setup

Have conda installed.
```bash
conda env create -f environment.yml
conda activate mlpf
```

# Training

### DELPHES training
The dataset is available from zenodo: https://doi.org/10.5281/zenodo.4452283.

To download and process the full dataset:
```bash
./get_data.sh
```

This script will download and process the data under a directory called `data/toptagging` under `xai4hep`.


To perform a training on the dataset:
```bash
cd ../
python -u particlenet_pipeline.py --dataset=data/toptagging
```

To load a pretrained model which is stored in a directory under `particleflow/experiments` for evaluation:
```bash
cd ../
python -u pyg_pipeline.py --data delphes --load --load_model=<model_directory> --load_epoch=<epoch_to_load> --dataset=<path_to_delphes_data> --dataset_qcd=<path_to_delphes_data>
```


### XAI and LRP studies on ParticleNet

You must have a pre-trained model under `experiments`:
```bash
cd ../
python -u lrp_mlpf_pipeline.py --run_lrp --make_rmaps --load_model=<your_model> --load_epoch=<your_epoch>
```
