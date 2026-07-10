import pandas as pd
import torch

from torch_dcr import dcr


def test_dcr1():
    source_df = pd.DataFrame(
        {
            "A": [4, 0, 3, -3, 1],
            "B": ["a", "b", "c", "d", "e"],
            "C": [1.12, -20, 3.4, -1.6, 60],
            "D": ["1", "2", "3", "3", "4"],
        }
    )

    target_df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": ["f", "b", "g", "d", "d"],
            "C": [0.0, 0.0, 2.0, -11.0, -4.0],
            "D": ["2", "2", "9", "4", "4"],
        }
    )

    dcr_df, index_df = dcr(
        source_df, target_df, device="cpu", k=1, output_indexes=True, metric="cosine"
    )

    expected_dcr = pd.DataFrame(
        {"dcr_1": [0.7761391, 0.53872633, 0.6627745, 0.3845973, 0.4744165]}
    ).astype("float32")

    expected_index = pd.DataFrame({"index_1": [2, 3, 2, 0, 2]})

    pd.testing.assert_frame_equal(dcr_df, expected_dcr, check_exact=False, atol=1e-5)
    pd.testing.assert_frame_equal(index_df, expected_index)


def test_dcr2():
    source_df = pd.DataFrame(
        {
            "A": [0, 0, -1],
            "B": ["a", "g", "b"],
        }
    )

    target_df = pd.DataFrame(
        {
            "A": [0, 0, 0, -0.5],
            "B": ["a", "b", "c", "e"],
        }
    )

    dcr_df, index_df = dcr(
        source_df,
        target_df,
        device="cpu",
        k=1,
        output_indexes=True,
        metric="l1",
        standardize=False,
    )

    expected_dcr = pd.DataFrame({"dcr_1": [0.0, 2.0, 1.0]}).astype("float32")
    expected_index = pd.DataFrame({"index_1": [0, 1, 1]})

    pd.testing.assert_frame_equal(dcr_df, expected_dcr, check_exact=False, atol=1e-5)
    pd.testing.assert_frame_equal(index_df, expected_index)


def test_dcr3():
    source_df = pd.DataFrame(
        {
            "A": ["a", "b", "c"],
            "B": ["1", "2", "3"],
        }
    )

    target_df = pd.DataFrame(
        {
            "A": ["a", "a", "b"],
            "B": ["1", "2", "1"],
        }
    )

    expected_dcrs = {
        "cosine": [0.0, 0.5, 1.0],
        "euclidean": [0.0, 1.4142135, 2.0],
        "l1": [0.0, 2.0, 4.0],
    }

    for metric, expected_dcr in expected_dcrs.items():
        expected_dcr_df = pd.DataFrame({"dcr_1": expected_dcr}).astype("float32")

        dcr_df = dcr(
            source_df,
            target_df,
            device="cpu",
            k=1,
            metric=metric,
            standardize=False,
        )
        pd.testing.assert_frame_equal(
            dcr_df, expected_dcr_df, check_exact=False, atol=1e-5
        )


def test_dcr4():
    source_df = pd.DataFrame(
        {
            "c1": ["a"],
            "c2": ["1"],
            "c3": [1.0],
            "c4": [-1.0],
        }
    )

    target_df = pd.DataFrame(
        {
            "c1": ["a"],
            "c2": ["2"],
            "c3": [1.0],
            "c4": [4.0],
        }
    )

    v1 = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, -1.0])
    v2 = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 4.0])

    expected_dcrs = {
        "l1": torch.abs(v1 - v2).sum().item(),
        "euclidean": torch.sqrt(((v1 - v2) ** 2).sum()).item(),
        "cosine": 1 - (torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))).item(),
    }

    for metric, expected_dcr in expected_dcrs.items():
        dcr_df = dcr(
            source_df,
            target_df,
            device="cpu",
            k=1,
            metric=metric,
            standardize=False,
        )
        assert dcr_df.iloc[0, 0] == expected_dcr
