import logging
from typing import Annotated, Tuple

import pandas as pd
import xgboost as xgb
from zenml import step

# Set up logger
logger = logging.getLogger(__name__)


@step
def train_xgb_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.1,
) -> Annotated[xgb.XGBClassifier, "model"]:

    """Trains an XGBoost classifier.

    Args:
        X_train: Training features.
        y_train: Training target.
        n_estimators: Number of trees in the model.
        max_depth: Maximum depth of each tree.
        learning_rate: Learning rate for the model.

    Returns:
        Tuple containing the trained model
    """
    # # Check for class imbalance
    # class_counts = y_train.value_counts()

    # Initialize and train XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        # use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )

    # Train the model on the original features
    model.fit(X_train, y_train)

    return model