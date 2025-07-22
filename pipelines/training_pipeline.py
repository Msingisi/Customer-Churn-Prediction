import logging

from steps.data_cleaner import clean_data_step
from steps.ingest_data import ingest_df
from steps.data_preprocessor import preprocess_data_step
from steps.data_splitter import split_data_step
from steps.model_evaluator import evaluate_model
from steps.model_trainer import train_xgb_model
from zenml import pipeline

# Set up logger
logger = logging.getLogger(__name__)


@pipeline(enable_cache=False)
def churn_training_pipeline(data_path: str):
    """
    Pipeline to train a customer churn prediction model.
    """
    raw_data = ingest_df(data_path=data_path)
    cleaned_data = clean_data_step(df=raw_data)
    preprocessed_data = preprocess_data_step(df=cleaned_data)
    X_train, X_test, y_train, y_test = split_data_step(df=preprocessed_data)
    model = train_xgb_model(
        X_train=X_train, y_train=y_train
    )
    evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    logger.info("Customer Churn training pipeline completed.")