import logging
from typing import List
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
from zenml import step

# Set up logger
logger = logging.getLogger(__name__)


@step
def preprocess_data_step(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for model training.

    Args:
        df: Cleaned Pandas DataFrame.

    Returns:
        Preprocessed Pandas DataFrame.
    """
    data_processed = df.copy()

    # Convert categorical variables to dummy variables
    categorical_cols = ['Complains', 'Charge  Amount', 'Tariff Plan', 'Status']
    data_processed = pd.get_dummies(data_processed, columns=[col for col in categorical_cols if col in data_processed.columns])


    # Handle skewness using cube root transformation
    for col in ['Call  Failure', 'Subscription  Length', 'Seconds of Use', 'Frequency of use', 'Frequency of SMS', 'Distinct Called Numbers',
                'Customer Value']:
         if col in data_processed.columns:
             data_processed[col] = np.cbrt(data_processed[col])


    # Min-Max Scaling
    scale_cols = ['Call  Failure', 'Subscription  Length', 'Seconds of Use', 'Frequency of use',
                  'Frequency of SMS', 'Distinct Called Numbers', 'Customer Value', 'Age']
    scaler = MinMaxScaler()
    for col in scale_cols:
         if col in data_processed.columns:
             data_processed[[col]] = scaler.fit_transform(data_processed[[col]])

    logging.info("Data preprocessing complete.")
    return data_processed