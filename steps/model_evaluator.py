from typing import Dict, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from typing_extensions import Annotated
from zenml import step
from zenml.types import HTMLString


def generate_evaluation_html(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    metrics: Dict[str, float],
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

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {metrics['roc_auc']:.3f})"))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="gray")))
    roc_fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=600,
        height=500,
        template="plotly_white",
    )
    roc_html = roc_fig.to_html(full_html=False, include_plotlyjs="cdn")

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_fig = go.Figure()
    pr_fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"PR (AUC = {metrics['pr_auc']:.3f})"))
    pr_fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        width=600,
        height=500,
        template="plotly_white",
    )
    pr_html = pr_fig.to_html(full_html=False, include_plotlyjs="cdn")

    # Prediction Probability Distribution
    pred_df = pd.DataFrame({"actual": y_test, "predicted_proba": y_pred_proba})
    dist_fig = px.histogram(
        pred_df,
        x="predicted_proba",
        color="actual",
        nbins=30,
        opacity=0.7,
        barmode="overlay",
        title="Distribution of Prediction Probabilities",
        labels={"predicted_proba": "Predicted Probability", "actual": "Actual Churn"},
        color_discrete_map={0: "#636EFA", 1: "#E64E33"},
    )
    dist_fig.update_layout(width=700, height=500, template="plotly_white")
    dist_html = dist_fig.to_html(full_html=False, include_plotlyjs="cdn")

    # Build HTML
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

    # Metric cards
    for name, color in zip(
        ["accuracy", "precision", "recall", "roc_auc", "pr_auc"],
        ["#f0f8ff", "#fff8f0", "#f0fff8", "#e8f0fe", "#ffe4e1"],
    ):
        html_parts.append(f"""
        <div class="metric-card" style="background-color: {color};">
            <h3>{name.replace('_', ' ').upper()}</h3>
            <div class="metric-value">{metrics[name]:.3f}</div>
        </div>
        """)

    html_parts.append("</div><div class='row'>")

    # Plots
    html_parts.append(f"<div class='column'><div class='plot-container'><h3>Confusion Matrix</h3>{cm_html}</div>")
    html_parts.append(f"<div class='plot-container'><h3>ROC Curve</h3>{roc_html}</div></div>")

    html_parts.append(f"<div class='column'><div class='plot-container'><h3>Precision-Recall Curve</h3>{pr_html}</div>")
    html_parts.append(f"<div class='plot-container'><h3>Prediction Distribution</h3>{dist_html}</div></div>")

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
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_val = precision_score(y_test, y_pred)
    recall_val = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)

    metrics = {
        "accuracy": accuracy,
        "precision": precision_val,
        "recall": recall_val,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }

    html_report = generate_evaluation_html(y_test, y_pred, y_pred_proba, metrics)
    return metrics, HTMLString(html_report)