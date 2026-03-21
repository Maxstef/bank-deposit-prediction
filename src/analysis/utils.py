import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_model_and_subset(
    pipeline, X, subset_indices=None, sample_size=100, random_state=42
):
    """
    Extract model and transformed subset of data from a pipeline.

    Returns:
    - model
    - transformed subset as DataFrame (with feature names)
    - transformed full DataFrame (with feature names)
    - subset indices of provided X dataset
    """

    model = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessing"]

    if subset_indices is None:
        sample_size = min(sample_size, len(X))
        subset_indices = np.random.default_rng(random_state).choice(
            X.index, size=sample_size, replace=False
        )

    X_subset_raw = X.loc[subset_indices]

    # transform subset
    X_subset_transformed = pipeline[:-1].transform(X_subset_raw)
    feature_names = preprocessor.get_feature_names_out()

    # transform full
    X_full_transformed = pipeline[:-1].transform(X)

    X_subset_transformed_df = pd.DataFrame(
        X_subset_transformed, columns=feature_names, index=subset_indices
    ).astype(np.float64)
    X_full_transformed_df = pd.DataFrame(X_full_transformed, columns=feature_names)

    return model, X_subset_transformed_df, X_full_transformed_df, subset_indices


def log_odds_to_proba(log_odds):
    """
    Convert log-odds (logits) to probability.

    Parameters:
    - log_odds: float or array-like

    Returns:
    - probability in range [0, 1]
    """
    return 1 / (1 + np.exp(-log_odds))


def get_mean_shap(shap_values, class_idx=None):
    if class_idx is not None:
        values = shap_values.values[:, :, class_idx]
    else:
        values = shap_values.values

    return np.abs(values).mean(axis=0), shap_values.feature_names


def plot_shap_comparison(shap_rf, shap_xgb, shap_lr, max_display=15):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- RF ---
    rf_vals, rf_names = get_mean_shap(shap_rf, class_idx=1)
    rf_idx = np.argsort(rf_vals)[-max_display:]

    # print(np.array(rf_names)[rf_idx], rf_vals[rf_idx])

    axes[0].barh(np.array(rf_names)[rf_idx], rf_vals[rf_idx])
    axes[0].set_title("Random Forest (SHAP)")

    # --- XGB ---
    xgb_vals, xgb_names = get_mean_shap(shap_xgb, class_idx=1)
    xgb_idx = np.argsort(xgb_vals)[-max_display:]

    # print(np.array(xgb_names)[xgb_idx], xgb_vals[xgb_idx])

    axes[1].barh(np.array(xgb_names)[xgb_idx], xgb_vals[xgb_idx])
    axes[1].set_title("XGBoost (SHAP)")

    # --- LR ---
    lr_vals, lr_names = get_mean_shap(shap_lr)
    lr_idx = np.argsort(lr_vals)[-max_display:]

    # print(np.array(lr_names)[lr_idx], lr_vals[lr_idx])

    axes[2].barh(np.array(lr_names)[lr_idx], lr_vals[lr_idx])
    axes[2].set_title("Logistic Regression (SHAP)")

    plt.tight_layout()
    plt.show()
