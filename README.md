### Hyperbolic Online Time Stream Modeling

This repository contains the code for [Hyperbolic Online Time Stream Modeling](https://dl.acm.org/doi/10.1145/3404835.3463119) accepted at SIGIR 2021 and implements the hyperbolic time aware LSTM as well as extends it to a hierarchical hyperbolic RNN model.

Dependencies:

1. PyTorch
2. geoopt==0.1.2

Please follow the [FAST](https://github.com/midas-research/fast-eacl) repository to obtain the data. This code expects data in a pickle format, with separate files for training and testing (dummy data to be uploaded soon!).

To train a model for stock movement prediction or profitability, from "code/", run:

```python
python -W ignore train.py --task movement --data stock(or china) --lr 5e-4 --num_epochs 500 --decay 1e-5 --batch_size 128 --name exp_name
```


To train a model for stock volatility prediction or profitability, from "code/", run:
```python
python -W ignore train.py --task volatility --data stock(or china) --lr 5e-4 --num_epochs 500 --decay 1e-5 --batch_size 128 --name exp_name
```

To calculate the profitability, from "code/", run:
```python
python stock_trade.py --model_path /path/to/model.pth --data stock(or china)
```


Acknolwedgements:
1. Hyrnn code: [https://github.com/ferrine/hyrnn](https://github.com/ferrine/hyrnn)
2. Manifolds and RAdam optimizer: [https://github.com/HazyResearch/hgcn](https://github.com/HazyResearch/hgcn)


If you use our models, consider citing:
```
@inproceedings{10.1145/3404835.3463119,
author = {Sawhney, Ramit and Agarwal, Shivam and Thakkar, Megh and Wadhwa, Arnav and Shah, Rajiv Ratn},
title = {Hyperbolic Online Time Stream Modeling},
year = {2021},
isbn = {9781450380379},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3404835.3463119},
doi = {10.1145/3404835.3463119},
booktitle = {Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {1682–1686},
numpages = {5},
keywords = {finance, language processing, hyperbolic geometry, stock market},
location = {Virtual Event, Canada},
series = {SIGIR '21}
}
```
