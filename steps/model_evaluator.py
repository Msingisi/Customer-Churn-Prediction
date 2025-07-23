from typing import Dict, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from typing_extensions import Annotated
from zenml import step
from zenml.types import HTMLString


def generate_evaluation_html(
    y_test: pd.Series,
    y_pred: np.ndarray,
    metrics: Dict[str, float],
    feature_importance_fig: go.Figure,
) -> str:
    """Generate HTML visualization of model evaluation results."""

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual"),
        x=["Non Churn", "Churn"],
        y=["Non Churn", "Churn"],
        color_continuous_scale="Blues",
        title="Confusion Matrix",
    )
    cm_fig.update_layout(width=600, height=500)
    cm_html = cm_fig.to_html(full_html=False, include_plotlyjs="cdn")

    # Feature Importance HTML
    feature_importance_html = feature_importance_fig.to_html(full_html=False, include_plotlyjs="cdn")

    # HTML Page
    html_parts = ["""
    <html><head><style>
        body { font-family: Arial; background-color: #f9f9f9; padding: 20px; }
        .dashboard { background: white; border-radius: 8px; padding: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .metrics { display: flex; flex-wrap: wrap; justify-content: space-between; }
        .metric-card {
            flex: 1; min-width: 200px; margin: 10px; padding: 20px;
            border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center;
        }
        .metric-value { font-size: 28px; font-weight: bold; margin: 10px 0; }
        .row { display: flex; flex-wrap: wrap; margin: 0 -15px; }
        .column { flex: 50%; padding: 0 15px; }
        @media (max-width: 800px) { .column { flex: 100%; } }
        .plot-container { margin-bottom: 30px; }
    </style></head><body>
    <div class="dashboard">
        <h1>Customer Churn Model Evaluation</h1>
        <div class="metrics">
    """]

    for name, color in zip(
        ["accuracy", "precision", "recall", "f1"],
        ["#f0f8ff", "#fff8f0", "#f0fff8", "#e8f0fe"],
    ):
        html_parts.append(f"""
        <div class="metric-card" style="background-color: {color};">
            <h3>{name.replace('_', ' ').upper()}</h3>
            <div class="metric-value">{metrics[name]:.3f}</div>
        </div>
        """)

    html_parts.append("</div><div class='row'>")

    # Plots
    html_parts.append(f"<div class='column'><div class='plot-container'><h3>Confusion Matrix</h3>{cm_html}</div></div>")
    html_parts.append(f"<div class='column'><div class='plot-container'><h3>Feature Importances</h3>{feature_importance_html}</div></div>")

    html_parts.append("</div></div></body></html>")
    return "".join(html_parts)


@step
def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[
    Annotated[Dict[str, float], "metrics"],
    Annotated[HTMLString, "evaluation_visualization"],
]:
    """Evaluates XGBoost classifier and returns metrics + visual report."""

    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_val = precision_score(y_test, y_pred)
    recall_val = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision": precision_val,
        "recall": recall_val,
        "f1": f1,
    }

    # Feature Importance: Top 10
    booster = model.get_booster()
    importance = booster.get_score(importance_type='gain')
    importance_df = pd.DataFrame(
        list(importance.items()), columns=["feature", "importance"]
    ).sort_values(by="importance", ascending=True).head(10)

    feature_fig = px.bar(
        importance_df,
        x="importance",
        y="feature",
        orientation="h",
        title="Feature Importances (by Gain)",
        labels={"importance": "Importance", "feature": "Feature"},
        color_discrete_sequence=["royalblue"],
    )
    feature_fig.update_layout(height=400)

    html_report = generate_evaluation_html(y_test, y_pred, metrics, feature_fig)
    return metrics, HTMLString(html_report)