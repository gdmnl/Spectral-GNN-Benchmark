# Framework Arrangement

## Covered Models

Refer to {py:class}`benchmark.trainer.ModelLoader`.

| **Category** | **Model** |
|:------------:|:----------|
| Fixed Filter | [GCN](https://arxiv.org/abs/1609.02907), [SGC](https://arxiv.org/pdf/1902.07153), [gfNN](https://arxiv.org/pdf/1905.09550), [GZoom](https://arxiv.org/pdf/1910.02370), [S$^2$GC](https://openreview.net/pdf?id=CYO5T-YjWZV),[GLP](https://arxiv.org/pdf/1901.09993), [APPNP](https://arxiv.org/pdf/1810.05997), [GCNII](https://arxiv.org/pdf/2007.02133), [GDC](https://proceedings.neurips.cc/paper_files/paper/2019/file/23c894276a2c5a16470e6a31f4618d73-Paper.pdf), [DGC](https://arxiv.org/pdf/2102.10739), [AGP](https://arxiv.org/pdf/2106.03058), [GRAND+](https://arxiv.org/pdf/2203.06389)|
|Variable Filter|[GIN](https://arxiv.org/pdf/1810.00826), [AKGNN](https://arxiv.org/pdf/2112.04575), [DAGNN](https://dl.acm.org/doi/pdf/10.1145/3394486.3403076), [GPRGNN](https://arxiv.org/pdf/2006.07988), [ARMAGNN](https://arxiv.org/pdf/1901.01343), [ChebNet](https://papers.nips.cc/paper_files/paper/2016/file/04df4d434d481c5bb723be1b6df1ee65-Paper.pdf), [ChebNetII](https://arxiv.org/pdf/2202.03580), [HornerGCN/ClenshawGCN](https://arxiv.org/pdf/2210.16508), [BernNet](https://arxiv.org/pdf/2106.10994), [LegendreNet](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10160025), [JacobiConv](https://arxiv.org/pdf/2205.11172), [FavardGNN/OptBasisGNN](https://arxiv.org/pdf/2302.12432)|
|Filter Bank|[AdaGNN](https://arxiv.org/pdf/2104.12840), [FBGNN](https://arxiv.org/pdf/2008.08844), [ACMGNN](https://arxiv.org/pdf/2210.07606), [FAGCN](https://arxiv.org/pdf/2101.00797), [G$^2$CN](https://proceedings.mlr.press/v162/li22h/li22h.pdf), [GNN-LF/HF](https://arxiv.org/pdf/2101.11859), [FiGURe](https://arxiv.org/pdf/2310.01892)|


## Covered Datasets

Refer to {py:class}`benchmark.trainer.SingleGraphLoader`.

| **Source** | **Graph** |
|:------------:|:----------|
| [PyG](https://pytorch-geometric.readthedocs.io/en/stable/modules/datasets.html) | cora, citeseer, pubmed, flickr, actor |
| [OGB](https://ogb.stanford.edu/docs/nodeprop/) | ogbn-arxiv, ogbn-mag, ogbn-products |
| [LINKX](https://github.com/CUAI/Non-Homophily-Large-Scale) | penn94, arxiv-year, genius, twitch-gamer, snap-patients, pokec, wiki |
| [Yandex](https://github.com/yandex-research/heterophilous-graphs) | chameleon, squirrel, roman-empire, minesweeper, amazon-ratings, questions, tolokers |
