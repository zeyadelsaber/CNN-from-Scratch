# CNN from Scratch — TartanAir

MATLAB CNN built from scratch to classify TartanAir scenes as **Nature**, **Rural**, or **Urban**. The model uses three convolution/batch-normalization/ReLU/max-pooling blocks, Adam optimization, and simple geometric augmentation.

**Test accuracy: 69.31%.**

## Run

Arrange the provided dataset as `data/{Training,Validation,Test}/{Nature,Rural,Urban}` or set `TARTANAIR_DATASET_ROOT`, then run:

```matlab
train_scratch_cnn
```

Requires MATLAB with Deep Learning Toolbox and Computer Vision Toolbox. A GPU is recommended but not required. Generated models and the dataset are excluded from Git.

Author: [Zeyad Elsaber](https://github.com/zeyadelsaber), University of Rome Tor Vergata.
