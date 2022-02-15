# SNPAD
Paper:Semi-supervised Neural Process for Anomaly Detection

![](https://img.shields.io/badge/python-3.8.10-green)![](https://img.shields.io/badge/pytorch-1.9.0-green)![](https://img.shields.io/badge/cudatoolkit-10.2-green)![](https://img.shields.io/badge/cudnn-7.6.5-green)

## Basic Usage

### Requirements

The code was tested with `python 3.8`, `pytorch 1.9.0`, `cudatookkit 10.2, and `cudnn 7.6.5`. Install the dependencies via [Anaconda](https://www.anaconda.com/):

```shell
# create virtual environment
conda create --name SNPAD python=3.8

# activate environment
conda activate SNPAD

# install pytorch & cudatoolkit
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

# install other requirements
```

### Run the code
```shell
cd SNPAD

# run the model PrEf
python main.py
