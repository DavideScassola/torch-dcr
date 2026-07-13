# `torch-dcr`: efficient DCR (Distance to Closest Record) for tabular data.

[![PyPI version](https://img.shields.io/pypi/v/torch-dcr.svg)](https://pypi.org/project/torch-dcr/)
[![CI](https://github.com/DavideScassola/torch-dcr/actions/workflows/ci.yml/badge.svg)](https://github.com/DavideScassola/torch-dcr/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A library for efficient, GPU-accelerated computation of **DCR (Distance to Closest Record)** for heterogeneous tabular data.

DCR measures, for each record in a source dataset, how close the nearest record in a target dataset is. It is commonly used to evaluate **privacy risk in synthetic data**: if synthetic records are too close to the training records, synthetic data probably leaks information about individuals in the training set.
`torch-dcr` computes this efficiently on CPU or GPU using PyTorch, and handles datasets with a mix of continuous and categorical columns out of the box.

## Basic usage

```python
import pandas as pd
from torch_dcr import dcr

source_df = pd.DataFrame(
    {
        "A": [4, 0, 3],
        "B": ["a", "b", "d"],
        "C": [1.12, -1.6, 6],
    }
)
target_df = pd.DataFrame(
    {
        "A": [1, 2, 3, 4, 5],
        "B": ["f", "b", "g", "d", "d"],
        "C": [0.0, 0.0, 2.0, -12.0, -4.0],
    }
)

dcr(source_df, target_df, metric="cosine")
```

Output:

```
      dcr_1
0  0.667516
1  0.166855
2  0.449738
```

Each row corresponds to a record in `source_df`, and `dcr_1` is the distance to the closest record found in `target_df`.

## Installation

To install the library run:
```bash
pip install torch-dcr
```

## How mixed-type data is handled

Real-world tabular data usually mixes continuous columns (e.g. floats, ints) with categorical columns (e.g. strings). `torch-dcr` handles this automatically.
The standard approach consists in converting categorical columns to one-hot encoded vectors, and then computing distances in the resulting high-dimensional space.
`torch-dcr` performs this computation without instantiating the full one-hot encodings, saving a considerable amount of memory and computation time.
Categorical columns are automatically detected based on the DataFrame's dtypes.


## Advanced usage

```python
dcr_df, indexes_df = dcr(
    source_df=source_df,   # Source DataFrame; DCR is computed for each record in this DataFrame
    target_df=target_df,   # Target DataFrame where the closest records are searched
    output_indexes=True,   # If True, also return the indexes of the closest records
    k=2,                   # Number of closest records to consider for each record in source_df
    metric="l1",           # Distance metric for continuous columns: "cosine", "euclidean", or "l1"
    device="cuda",         # "cpu" or "cuda" for GPU acceleration
    standardize=True,      # Whether to standardize continuous features before computing distances
    batch_size=1000,       # Batch size for processing, useful for large DataFrames
)
```

Output:

```
DCR:
       dcr_1     dcr_2
0  2.790001  3.465423
1  1.551357  2.918901
2  2.716115  3.055198

Indexes:
   index_1  index_2
0        2        1
1        1        0
2        2        4
```

With `k=2`, `dcr_1`/`dcr_2` are the distances to the 1st and 2nd closest records in `target_df`, and `index_1`/`index_2` (from `indexes_df`) are the corresponding row indexes in `target_df`.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source_df` | `pd.DataFrame` | required | Records for which DCR is computed |
| `target_df` | `pd.DataFrame` | required | Records searched for nearest neighbors |
| `metric` | `str` | `"cosine"` | Distance metric for continuous columns: `"cosine"`, `"euclidean"`, or `"l1"` |
| `k` | `int` | `1` | Number of nearest neighbors to return per record |
| `output_indexes` | `bool` | `False` | If `True`, also return a DataFrame of nearest-neighbor indexes |
| `standardize` | `bool` | `True` | Standardize continuous columns before computing distances |
| `device` | `str` | `"cpu"` | `"cpu"` or `"cuda"` |
| `batch_size` | `int` | 1000 | Batch size to limit memory usage on large datasets |
| `progress_bar` | `bool` | `True` | If `True`, show a progress bar during computation |


## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.


## Citation

If you use `torch-dcr` in your research or project, please cite it using the following bibtex entry:

```bibtex
@misc{torch_dcr_2026,
  author       = {Davide Scassola},
  title        = {torch-dcr: A fast implementation of the DCR (Distance to Closest Record) for tabular data},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{[https://github.com/DavideScassola/torch-dcr](https://github.com/DavideScassola/torch-dcr)}}
}
