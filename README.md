## Prerequisites
- Python 3.6
- GPU Memory >= 11G
- Numpy
- Pytorch 0.4+

Preparation 1: create folder for dataset.
first, download DukeMTMC-reID dataset from the links below
google drive: https://drive.google.com/open?id=1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O
baidu disk: https://pan.baidu.com/s/1jS0XM7Var5nQGcbf9xUztw
password for baidu disk: bhbh

second,
```bash
mkdir data
unzip DukeMTMC-reID.zip
ln -s DukeMTMC-reID duke

``` 
then, get the directory structure
├── negative_transfer
    　　　　　　      ├── data
    　　　　　　　            ├── duke
    　　　　　　　            ├── DukeMTMC-reID


Preparation 2: Put the images with the same id in one folder. You may use 
```bash
python prepare.py
```

Finally, conduct training, testing and evaluating with one command
```bash
python run.py
```

This code is related to our paper _A Negative Transfer Approach to Person Re-identification via Domain Augmentation_.

If you use this code, please cite our paper as:
```

```
