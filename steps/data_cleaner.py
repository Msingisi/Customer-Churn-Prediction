import logging
import pandas as pd
from zenml import step


@step
def clean_data_step(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the input DataFrame.

    Args:
        df: Pandas DataFrame to clean.

    Returns:
        Cleaned Pandas DataFrame.
    """
    data = df.copy()

    # Drop irrelevant columns
    drop_cols = ["Age Group", "Charge  Amount"]
    data.drop(columns=[col for col in drop_cols if col in data.columns], inplace=True)

    # Map binary/ordinal to categorical values
    if 'Complains' in data.columns:
        data['Complains'] = data['Complains'].map({0: 'No complaint', 1: 'Complaint'})

    if 'Tariff Plan' in data.columns:
        data['Tariff Plan'] = data['Tariff Plan'].map({1: 'Pay as you go', 2: 'Contractual'})

    if 'Status' in data.columns:
        data['Status'] = data['Status'].map({1: 'Active', 2: 'Non-active'})

    # if 'Charge Amount' in data.columns:
    #      # Remove rows where Charge Amount is 10
    #     data = [data['Charge  Amount'] != 10]

    # # Categorize Charge Amount into Low, Medium, High
    # if 'Charge  Amount' in data.columns:
    #     data['Charge  Amount'] = pd.cut(
    #         data['Charge  Amount'],
    #         bins=[0, 3, 6, 9],
    #         labels=['Low', 'Medium', 'High']
    #         )

    logging.info("Data cleaning complete.")
    return data