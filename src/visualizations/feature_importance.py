import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_importance(pipeline, top_n=20, name="", figsize=(10, 6)):
    """
    Plot top N feature importances for a sklearn Pipeline.

    Supports:
    - Tree-based models (feature_importances_)
    - Linear models (coef_)

    Parameters:
    - pipeline: fitted sklearn Pipeline
    - top_n: number of top features to display
    - name: model name
    - figsize: matplotlib figure size
    """

    # --- Extract preprocessor and model ---
    preprocessor = pipeline.named_steps["preprocessing"]
    model = pipeline.named_steps["classifier"]

    # --- Get feature names after preprocessing ---
    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        raise ValueError("Preprocessor does not support get_feature_names_out()")

    # --- Get feature importance ---
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()

    else:
        raise ValueError("Model does not support feature importance extraction")

    # --- Create DataFrame ---
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})

    # --- Sort and select top N ---
    fi_df = fi_df.sort_values(by="importance", ascending=False).head(top_n)

    # display(fi_df)

    # --- Plot ---
    plt.figure(figsize=figsize)
    plt.barh(fi_df["feature"][::-1], fi_df["importance"][::-1])
    plt.xlabel("Importance")
    plt.title(f"{name} - Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.show()
