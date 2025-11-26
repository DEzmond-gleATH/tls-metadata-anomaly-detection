# Encrypted Traffic Anomaly Detection via TLS Metadata

This project classifies **encrypted TLS flows** into application categories using only
**flow-level metadata**, without decrypting payloads. It was developed as part of  
**CP-614: Applied Cryptography** at Wilfrid Laurier University.

We use the **CESNET-TLS22** dataset and train **Random Forest** and **XGBoost** models
on a highly imbalanced, real-world dataset.

---

## Goals

- Detect anomalies / classify encrypted TLS flows using only metadata.
- Preserve **privacy** by avoiding payload inspection.
- Handle **severe class imbalance** using **SMOTE**.
- Compare **Random Forest** and **XGBoost** performance on multi-class traffic.

---

## Project Structure

```text
tls-metadata-anomaly-detection/
├─ src/
│  ├─ __init__.py
│  ├─ data_preparation.py      # loading, cleaning, encoding, scaling, SMOTE
│  ├─ modeling.py              # model build / train / evaluate
├─ data/
│  └─ cesnet-tls22/            # raw CESNET-TLS22 week folders (not in repo)
├─ reports/
│  └─ Group_7_Report.pdf       # course report (optional)
├─ main.py                     # entry point script
├─ requirements.txt
├─ README.md
└─ .gitignore
````

---

## Methods (Overview)

**Dataset**

* **Source:** CESNET-TLS22 (public encrypted traffic dataset).
* Each row = one **TLS flow** with 27 numerical metadata features, including:

  * Bytes and packets in each direction
  * Flow duration
  * PPI timing/length features
  * TCP flag indicators (e.g., ACK, SYN, PSH, and reverse flags)
* **Label:** `CATEGORY` – 21 application / service classes.

**Preprocessing**

Implemented in `src/data_preparation.py`:

* Load and merge multiple `.csv` / `.csv.gz` files for a selected week.
* Drop non-informative columns (e.g., `source_file`, `PPI`, `APP`).
* Remove duplicates.
* Encode `CATEGORY` using `LabelEncoder`.
* Keep only numeric feature columns and scale them with `MinMaxScaler`.
* Train–test split with **stratification** on the label.
* Apply **SMOTE** to balance the training set.

**Models**

Implemented in `src/modeling.py`:

* `RandomForestClassifier`
* `XGBClassifier` (multi-class softmax)

Both are trained on the SMOTE-balanced training data and evaluated on the held-out test set.

**Evaluation**

* Text **classification report** (precision, recall, F1-score).
* **Confusion matrix**.
* **Multi-class ROC-AUC (OvR)** when `predict_proba` is available.

---

## Results

Headline metrics from our experiments:

| Model         | ROC AUC | Precision | Recall | F1-score |
| ------------- | :-----: | :-------: | :----: | :------: |
| Random Forest |   0.82  |    0.78   |  0.83  |   0.79   |
| XGBoost       |   0.70  |    0.60   |  0.79  |   0.63   |

**Summary:**

* **Random Forest** achieved the most balanced performance across all metrics.
* **XGBoost** reached high recall but lower precision/F1, producing more false positives.
* Strong results are possible using **metadata only**, which is promising for
  privacy-preserving monitoring.

(You can update these numbers if you rerun with different settings.)

---

## Installation

```bash
git clone https://github.com/DEzmond-gleATH/tls-metadata-anomaly-detection.git
cd tls-metadata-anomaly-detection

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` contains:

```text
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
matplotlib
seaborn
```

---

## Data Setup

1. Download the **CESNET-TLS22** dataset from the official source.
2. Place one or more week folders under:

   ```text
   data/cesnet-tls22/W-2021-40/...
   ```

   (You can change the week name in `main.py`.)

> The raw dataset is **not** included in this repository due to size and licensing.
> Only the code and project report are version-controlled.

---

## How to Run

From the project root, with the virtual environment activated:

```bash
python main.py
```

The script will:git initgit status

1. Load and merge the configured CESNET-TLS22 week.
2. Clean, encode, scale, and rebalance the data.
3. Train Random Forest and XGBoost.
4. Print classification reports, confusion matrices and ROC-AUC for each model.
---
