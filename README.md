# AI-Powered Customer Intelligence and Business Analytics

This repository contains an end-to-end Jupyter notebook for Amazon e-commerce customer and business analytics.

## Contents

- `amazon_customer_intelligence.ipynb`: complete analysis and model-training workflow
- `amazon_ecommerce_1M.csv`: transaction dataset used by the notebook, Follow the link to download the dataset: https://www.kaggle.com/datasets/sharmajicoder/amazon-e-commerce
- `customer_intelligence_models.joblib`: generated model bundle, created after running the export cell

## Analytics included

1. Descriptive analytics for categories, brands, orders, revenue, and customer buying behavior
2. K-Means customer segmentation
3. Random Forest classification for a purchase-category proxy
4. Linear Regression and Random Forest Regression for customer revenue modeling
5. Optional TF-IDF and Naive Bayes sentiment analysis when free-text reviews exist

## Data usage

The notebook reads the complete CSV into memory with `pandas.read_csv`. It does not intentionally sample the one-million-row file for the main descriptive analysis or customer aggregation. Cleaning can reduce the final usable transaction count because duplicate, invalid, or missing records are removed.

The models do not all fit on raw transaction rows:

- K-Means uses one engineered row per customer.
- Purchase classification uses one row per repeat customer and excludes the final target transaction from the history features.
- Revenue models use one engineered row per customer.
- Descriptive analytics uses the complete cleaned transaction table.

The notebook prints row counts so the exact data used can be audited.

## Setup

Create a Python environment and install the main dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

Open the notebook in Jupyter or VS Code:

```bash
jupyter notebook amazon_customer_intelligence.ipynb
```

Run the cells from top to bottom. The data-loading cell searches the current notebook folder first and also includes a Kaggle path fallback.

## Generate the Joblib model file

Run all training cells, then run **Step 10: Export reusable ML artifacts with Joblib**. This creates:

```text
customer_intelligence_models.joblib
```

The bundle contains the fitted segmentation scaler and K-Means model, segmentation feature names and clipping bounds, the revenue model when training succeeds, and the purchase classifier when training succeeds. Saving preprocessing with the model prevents feature-order and scaling mismatches during reuse.

Do not load untrusted `.joblib` files. Joblib uses Python serialization and can execute code during loading.

## Reuse a model from GitHub

Commit `customer_intelligence_models.joblib` to a GitHub repository. In the notebook's **Step 11: Reuse the exported model from GitHub**, set `JOBLIB_URL` to the file's raw GitHub URL:

```python
JOBLIB_URL = (
    "https://raw.githubusercontent.com/OWNER/REPOSITORY/main/"
    "customer_intelligence_models.joblib"
)
```

The reuse cell downloads and loads the bundle without loading the one-million-row CSV or retraining the models. It demonstrates:

- Customer revenue prediction from an already-engineered customer feature record
- Segment assignment using the saved clipping bounds, scaler, and K-Means model

The input columns must use the same definitions and units as the training notebook. For example, `total_revenue` is not an input to the revenue prediction example because it is the value being predicted.

## Important modeling limitations

- The current data has ratings but no written review text. Sentiment modeling is skipped until a review-text column is supplied.
- The revenue model predicts historical total revenue as a CLV proxy. True future CLV requires a defined future observation window.
- Purchase prediction uses the last observed category as a proxy target. A production recommender should use a clear future time window and richer product/customer features.
- Before production use, add time-based validation, monitoring, privacy controls, fairness checks, and model versioning.
