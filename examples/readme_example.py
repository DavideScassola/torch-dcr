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

dcr_df = dcr(source_df, target_df, metric="cosine")

print(dcr_df)
