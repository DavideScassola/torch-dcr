import pandas as pd
import torch
from tqdm import tqdm


def hybrid_cosine_distance(x_real, x_cat, y_real, y_cat):
    """Computes the hybrid cosine distance between two sets of records.

    Args:
        x_real (torch.Tensor): A tensor containing the continuous features of
        the first set of records. x_cat (torch.Tensor): A tensor containing the
        categorical features of the first set of records. y_real (torch.Tensor):
        A tensor containing the continuous features of the second set of
        records. y_cat (torch.Tensor): A tensor containing the categorical
        features of the second set of records.

    Returns:
        torch.Tensor: A tensor containing the hybrid cosine distances between
        each pair of records from the two sets.
    """
    cat_dim = x_cat.shape[-1]
    real_dot_product = torch.matmul(x_real, y_real.t())
    cat_dot_product = (x_cat.unsqueeze(1) == y_cat.unsqueeze(0)).sum(-1).float()
    x_norm = torch.sqrt((x_real**2).sum(dim=-1) + cat_dim)
    y_norm = torch.sqrt((y_real**2).sum(dim=-1) + cat_dim)
    return 1 - (real_dot_product + cat_dot_product) / (
        x_norm.unsqueeze(1) * y_norm.unsqueeze(0)
    )


def hybrid_euclidean_distance(x_real, x_cat, y_real, y_cat):
    """Computes the hybrid Euclidean distance between two sets of records.

    Args:
        x_real (torch.Tensor): A tensor containing the continuous features of
        the first set of records. x_cat (torch.Tensor): A tensor containing the
        categorical features of the first set of records. y_real (torch.Tensor):
        A tensor containing the continuous features of the second set of
        records. y_cat (torch.Tensor): A tensor containing the categorical
        features of the second set of records.

    Returns:
        torch.Tensor: A tensor containing the hybrid Euclidean distances between
        each pair of records from the two sets.
    """
    real_l2_distance = ((x_real.unsqueeze(1) - y_real.unsqueeze(0)) ** 2).sum(-1)
    cat_l2_distance = (x_cat.unsqueeze(1) != y_cat.unsqueeze(0)).sum(-1).float() * 2
    return torch.sqrt(real_l2_distance + cat_l2_distance)


def hybrid_l1_distance(x_real, x_cat, y_real, y_cat):
    """Computes the hybrid L1 distance between two sets of records.

    Args:
        x_real (torch.Tensor): A tensor containing the continuous features of
        the first set of records. x_cat (torch.Tensor): A tensor containing the
        categorical features of the first set of records. y_real (torch.Tensor):
        A tensor containing the continuous features of the second set of
        records. y_cat (torch.Tensor): A tensor containing the categorical
        features of the second set of records.

    Returns:
        torch.Tensor: A tensor containing the hybrid L1 distances between each
        pair of records from the two sets.
    """
    real_l1_distance = torch.abs(x_real.unsqueeze(1) - y_real.unsqueeze(0)).sum(-1)
    cat_l1_distance = (x_cat.unsqueeze(1) != y_cat.unsqueeze(0)).sum(-1).float() * 2
    return real_l1_distance + cat_l1_distance


DISTANCES = {
    "cosine": hybrid_cosine_distance,
    "euclidean": hybrid_euclidean_distance,
    "l1": hybrid_l1_distance,
}


def dcr(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    k: int = 1,
    device: str = "cpu",
    standardize: bool = True,
    batch_size: int = 1000,
    metric: str = "cosine",
    output_indexes: bool = False,
    progress_bar: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Computes the distance to the closest record (DCR) between two dataframes.

    For each record in the source dataframe, it outputs the distances to the k
    closest records in the target dataframe. The output has the same number of
    rows as the source dataframe, and k columns.

    Args:
        source_df (pd.DataFrame): The source dataframe containing the records
        for which we want to compute the DCR. target_df (pd.DataFrame): The
        target dataframe containing the records against which we want to compute
        the DCR. k (int, optional): The number of closest records to consider.
        Defaults to 1. device (str, optional): The device on which to perform
        the computations. Defaults to "cpu". standardize (bool, optional):
        Whether to standardize the continuous features. The standardization is
        performed using the mean and standard deviation of the target dataset.
        Defaults to True. batch_size (int, optional): The size of the batches
        for processing. Defaults to 1000. metric (str, optional): The metric to
        use for computing distances. Options are "cosine", "euclidean" and "l1".
        Defaults to "cosine". output_indexes (bool, optional): Whether to output
        the indexes of the closest records. If true a tuple of (distances,
        indexes) is returned. Defaults to False. progress_bar (bool, optional):
        Whether to display a progress bar. Defaults to True.

    Returns:
        pd.DataFrame: A dataframe containing the distances to the k closest
        records in the target dataframe. If output_indexes is True, a tuple of
        (distances, indexes) is returned, where distances is a dataframe
        containing the distances and indexes is a dataframe containing the
        indexes of the closest records in the target dataframe.
    """
    if metric not in DISTANCES:
        raise ValueError(
            f"Invalid metric '{metric}'. Valid options are: {list(DISTANCES.keys())}"
        )
    hybrid_distance = DISTANCES[metric]

    source_delimiter = len(source_df)
    common_df = pd.concat([source_df.copy(), target_df.copy()], ignore_index=True)

    # Detect categorical columns
    cat_columns = common_df.select_dtypes(
        include=["category", "object", str]
    ).columns.tolist()

    # Convert object, str and categorical columns to categorical
    for col in cat_columns:
        common_df[col] = common_df[col].astype("category")

    # Convert categorical columns to numerical codes
    for col in cat_columns:
        common_df[col + "_number"] = common_df[col].cat.codes
        common_df = common_df.drop(columns=[col])
        common_df = common_df.rename(columns={col + "_number": col})

    # Convert to tensors
    source_continuous = torch.tensor(
        common_df.iloc[:source_delimiter].drop(columns=cat_columns).values,
        dtype=torch.float32,
    ).to(device)
    target_continuous = torch.tensor(
        common_df.iloc[source_delimiter:].drop(columns=cat_columns).values,
        dtype=torch.float32,
    ).to(device)
    source_categorical = torch.tensor(
        common_df.iloc[:source_delimiter][cat_columns].values, dtype=torch.int32
    ).to(device)
    target_categorical = torch.tensor(
        common_df.iloc[source_delimiter:][cat_columns].values, dtype=torch.int32
    ).to(device)

    del common_df  # Free up memory

    if standardize:
        # using the mean and std of the target dataset
        mean = target_continuous.mean(dim=0)
        std = target_continuous.std(dim=0)
        source_continuous = (source_continuous - mean) / std
        target_continuous = (target_continuous - mean) / std

    distance_matrix = torch.zeros((len(source_df), k)).to(device)

    indices_matrix = None
    if output_indexes:
        indices_matrix = torch.zeros((len(source_df), k), dtype=torch.int64).to(device)

    for i in tqdm(range(0, len(source_df), batch_size), disable=not progress_bar):
        start = i
        end = min((i + 1) + batch_size, len(source_df))

        distance = hybrid_distance(
            source_continuous[start:end],
            source_categorical[start:end],
            target_continuous,
            target_categorical,
        )

        distance_matrix[start:end] = torch.topk(distance, k=k, largest=False)[0]
        if indices_matrix is not None:
            indices_matrix[start:end] = torch.topk(distance, k=k, largest=False)[1]

    distances = distance_matrix.cpu().numpy()
    dcr_df = pd.DataFrame(distances, columns=[f"dcr_{i + 1}" for i in range(k)])
    if indices_matrix is not None:
        index_df = pd.DataFrame(
            indices_matrix.cpu().numpy(), columns=[f"index_{i + 1}" for i in range(k)]
        )
        return dcr_df, index_df
    return dcr_df
