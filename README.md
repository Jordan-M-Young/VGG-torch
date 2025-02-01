# VGG-torch
Pytorch Implementation of VGG as detailed in the 2015 paper [VERY DEEP CONVOLUTIONAL NETWORKS
FOR LARGE-SCALE IMAGE RECOGNITION](https://www.robots.ox.ac.uk/~vgg/publications/2015/Simonyan15/simonyan15.pdf) by Karen Simonyan & Andrew Zisserman


# Data

The methodology described in the paper uses the ILVSRC-2012 imagenet-1k dataset. Details on this data can be found [here](https://www.image-net.org/challenges/LSVRC/2012/). This dataset is available for download from the huggingface datasets repo under the card `ILSVRC/imagenet-1k`. However you may need to authenticate and accepts some terms of use to use this dataset. This dataset is quite large ~100 GB, for this reason I've chosen to work with a specific copy of the dataset hosted on Huggingface by the user [benjamin-paine](https://huggingface.co/benjamin-paine). This dataset is referred to as the [imagenet-1k-256x256](https://huggingface.co/datasets/benjamin-paine/imagenet-1k-256x256/tree/main/data). This is the same dataset I used in my [AlexNet Implementation](https://github.com/Jordan-M-Young/alexnet-torch).



## Quick Start

To download the whole dataset, I'd suggest using the huggingface/transformers library. To download a fraction of the dataset run the following:

```bash
poetry run download-sample
```

This will download the `train-00000-of-00040.parquet` file from the huggingface repo to this projects data directory. Feel free to extend this basic functionality.

To unpack the downloaded parquet file, run:

```bash
poetry run mk-imgs
```

This will write the byte strings found in the downloaded parquet file to .jpg and write the associated labels to a labels.txt file.


