# TPC-NAS: Sub-Five-Minute Neural Architecture Search for Image Classification, Object-Detection, and Super-Resolution

# Pretrained Model and Result
Download and unzip save_dir in the root of TPC-NAS from the link below:
https://drive.google.com/file/d/1YCeSXHhu3YGfbRj84iXTLXlZkAj-LEL1/view?usp=sharing

Change the data path in DataLoader

## run 
```
sh scripts/TPC_NAS_params300.sh
sh scripts/TPC_NAS_params500.sh
sh scripts/TPC_NAS_params700.sh
```

# Open Source
Some few files in this repository are modified from the following open-source implementations:
```
https://github.com/DeepVoltaire/AutoAugment/blob/master/autoaugment.py
https://github.com/VITA-Group/TENAS
https://github.com/SamsungLabs/zero-cost-nas
https://github.com/BayesWatch/nas-without-training
https://github.com/rwightman/gen-efficientnet-pytorch
https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html
```
Most of the code thanks to the contribution of Zen-NAS
```
https://github.com/idstcv/ZenNAS
```


