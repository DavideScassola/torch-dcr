# torch-dcr
[![CI](https://github.com/DavideScassola/torch-dcr/actions/workflows/ci.yml/badge.svg)](https://github.com/DavideScassola/torch-dcr/actions/workflows/ci.yml)

A library for efficiently computing DCR (Distance to Closest Record) of tabular data.

To install the library in editable mode, run the following command:
```
python -m pip install -e .
```


## Example usage
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
