import os
import pandas as pd
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE


def load_week_data(base_dir: str, week_folder: str) -> pd.DataFrame:
    """
    Load and merge all .csv/.csv.gz files for a given week folder of CESNET-TLS22.

    Expected structure:
        base_dir/
            W-2021-40/
                2021-10-04/
                    file1.csv.gz
                    file2.csv.gz
                2021-10-05/
                    ...

    Returns
    -------
    merged_df : pd.DataFrame
    """
    week_path = os.path.join(base_dir, week_folder)
    if not os.path.isdir(week_path):
        raise FileNotFoundError(f"Week path not found: {week_path}")

    df_list = []

    for day_folder in sorted(os.listdir(week_path)):
        day_path = os.path.join(week_path, day_folder)
        if not os.path.isdir(day_path):
            continue

        for file_name in sorted(os.listdir(day_path)):
            if not (file_name.endswith(".csv") or file_name.endswith(".csv.gz")):
                continue

            file_path = os.path.join(day_path, file_name)
            try:
                print(f"[INFO] Loading {file_path}")
                df_day = pd.read_csv(file_path)
                df_list.append(df_day)
            except MemoryError:
                print(f"[WARN] MemoryError while loading {file_path}, skipping.")
            except Exception as e:
                print(f"[WARN] Failed to load {file_path}: {e}")

    if not df_list:
        raise RuntimeError(f"No CSV data was loaded from week folder: {week_path}")

    merged_df = pd.concat(df_list, ignore_index=True)
    print(f"[INFO] Merged DataFrame shape for {week_folder}: {merged_df.shape}")
    return merged_df


def clean_and_encode(
    df: pd.DataFrame,
    label_col: str = "CATEGORY",
    drop_cols=None,
) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder, MinMaxScaler]:
    """
    Clean raw CESNET data and return scaled numeric features + encoded labels.

    Steps:
    - Drop non-useful/object columns.
    - Remove duplicates.
    - Encode the label column.
    - Keep only numeric feature columns.
    - Apply MinMax scaling.

    Returns
    -------
    X : pd.DataFrame
        Scaled numeric feature matrix.
    y : pd.Series
        Encoded labels.
    le : LabelEncoder
        Fitted label encoder.
    scaler : MinMaxScaler
        Fitted scaler.
    """
    df_clean = df.copy()

    if drop_cols is None:
        # adjust this list based on what you dropped in the notebook
        drop_cols = ["source_file", "PPI", "APP"]

    for col in drop_cols:
        if col in df_clean.columns:
            df_clean = df_clean.drop(columns=[col])

    if label_col not in df_clean.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame.")

    # Remove duplicates
    df_clean = df_clean.drop_duplicates()

    # Encode label
    le = LabelEncoder()
    df_clean[label_col] = le.fit_transform(df_clean[label_col])
    y = df_clean[label_col]

    # Separate features
    X_raw = df_clean.drop(columns=[label_col])

    # Keep numeric columns only
    numeric_df = X_raw.select_dtypes(include=["int64", "int32", "float64", "float32"])
    print(f"[INFO] Numeric feature columns: {len(numeric_df.columns)}")

    # Scale features
    scaler = MinMaxScaler()
    X_scaled_array = scaler.fit_transform(numeric_df)
    X = pd.DataFrame(X_scaled_array, columns=numeric_df.columns)

    print(f"[INFO] Preprocessing complete. X shape = {X.shape}, y shape = {y.shape}")
    return X, y, le, scaler


def split_and_resample(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Train/test split with stratification and SMOTE resampling on the training set.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    print(f"[INFO] Train shape before SMOTE: {X_train.shape}")
    print(f"[INFO] Test  shape: {X_test.shape}")

    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"[INFO] Train shape after SMOTE: {X_train_res.shape}")
    return X_train_res, X_test, y_train_res, y_test
