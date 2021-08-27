# README
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
conda env create -f environment
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

