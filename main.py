# main.py

from src.data_preparation import (
    load_week_data,
    clean_and_encode,
    split_and_resample,
)
from src.modeling import build_models, train_and_evaluate


def main():
    # Adjust these paths/names to match your local CESNET folder
    base_dir = "data/cesnet-tls22"   # root where W-2021-40 etc. live
    week_folder = "W-2021-40"        # change or loop over multiple weeks if you want

    print("[STEP 1] Loading raw CESNET-TLS22 data...")
    raw_df = load_week_data(base_dir, week_folder)

    print("[STEP 2] Cleaning & encoding...")
    X, y, label_encoder, scaler = clean_and_encode(
        raw_df,
        label_col="CATEGORY",   # change if your label is named differently
    )

    print("[STEP 3] Train/Test split + SMOTE...")
    X_train, X_test, y_train, y_test = split_and_resample(X, y)

    print("[STEP 4] Building models...")
    models = build_models()

    print("[STEP 5] Training & evaluating...")
    results = train_and_evaluate(models, X_train, y_train, X_test, y_test)

    print("\n[SUMMARY] AUC by model:")
    for name, info in results.items():
        print(f"  {name}: {info['auc']:.3f}" if info["auc"] is not None else f"  {name}: N/A")


if __name__ == "__main__":
    main()
