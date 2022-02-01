# README
The replication package of paper *Just-In-Time Obsolete Comment Detection and Update*.

## Dataset
- Download the dataset from [here](https://drive.google.com/drive/folders/1FKhZTQzkj-QpTdPE9f_L9Gn_pFP_EdBi?usp=sharing)

```bash
mkdir data
mv cup2_dataset.zip cup2_updater_dataset.zip data
cd data
unzip cup_dataset.zip
unzip cup2_updater_dataset.zip
```


## Installation
```
conda env create -f environment.yml
pip install git+https://github.com/Maluuba/nlg-eval.git@81702e
# set the data_path
nlg-eval --setup ${data_path}
```

```bash
sudo apt-get install python2
curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py
sudo python2 get-pip.py
pip2 install scipy
```

## Run
```bash
# run OCD
python -m main run_ocd configs/OCD.yml OCD

# run CUP
python -m main run_cup configs/CUP.yml CUP

# run CUP2
python -m main run_cup2 configs/OCD.yml OCD configs/CUP.yml CUP configs/CUP2.yml CUP2
```

## Infer and Eval using trained models
- Download the trained models from [here](https://zenodo.org/record/5802819).

```bash
unzip OCD.zip
unzip CUP.zip
mkdir CUP2

# infer and eval OCD
python -m infer --log-dir OCD --config configs/OCD.yml
python -m eval --log-dir OCD --config configs/OCD.yml

# infer and eval CUP
python -m infer --log-dir CUP --config configs/CUP.yml
python -m eval --log-dir CUP --config configs/CUP.yml

# infer and eval CUP^2
python -m two_stage infer configs/OCD.yml OCD configs/CUP.yml CUP configs/CUP2.yml CUP2
python -m two_stage eval configs/OCD.yml OCD configs/CUP.yml CUP configs/CUP2.yml CUP2
```

**NOTE**: In our paper, each model was trained and evaluated 10 times, and the reported results are the average performance of the 10 experiments. 
Here we only provide one trained model for OCD and CUP each.
So the outputs of the above commands would be different from those reported in our paper.

## Scripts for Matching Comment Sentences
See `utils.comment.test_javadoc_desc_preprocessor` for a usage example.
