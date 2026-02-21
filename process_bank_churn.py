import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Dict, Any, Tuple, Optional, List


def split_train_val(df: pd.DataFrame, target_col: str, test_size: float = 0.2) -> Dict[str, pd.DataFrame]:
    """
    Split the dataframe into training and validation sets.
    """
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df[target_col])
    return {'train': train_df, 'val': val_df}


def create_inputs_targets(df_dict: Dict[str, pd.DataFrame], drop_cols: List[str], target_col: str) -> Dict[str, Any]:
    """
    Create inputs and targets for training and validation sets.
    """
    data = {}
    for split, df in df_dict.items():
        data[f'{split}_inputs'] = df.drop(columns=drop_cols + [target_col]).copy()
        data[f'{split}_targets'] = df[target_col].copy()
    return data


def scale_numeric_features(data: Dict[str, Any], numeric_cols: List[str]) -> MinMaxScaler:
    """
    Scale numeric features using MinMaxScaler and return the fitted scaler.
    """
    scaler = MinMaxScaler().fit(data['train_inputs'][numeric_cols])
    for split in ['train', 'val']:
        data[f'{split}_inputs'][numeric_cols] = scaler.transform(data[f'{split}_inputs'][numeric_cols])
    return scaler


def encode_categorical_features(data: Dict[str, Any], categorical_cols: List[str]) -> OneHotEncoder:
    """
    One-hot encode categorical features and return the fitted encoder.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(data['train_inputs'][categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

    for split in ['train', 'val']:
        encoded = encoder.transform(data[f'{split}_inputs'][categorical_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=data[f'{split}_inputs'].index)
        data[f'{split}_inputs'] = pd.concat([data[f'{split}_inputs'], encoded_df], axis=1)
        data[f'{split}_inputs'].drop(columns=categorical_cols, inplace=True)

    data['encoded_cols'] = encoded_cols
    return encoder


def preprocess_data(raw_df: pd.DataFrame, scaler_numeric: bool = True) -> Tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], Optional[MinMaxScaler], OneHotEncoder]:
    """
    Main pipeline to preprocess the raw dataframe.
    Includes optional numeric scaling.
    """
    df_dict = split_train_val(raw_df, target_col='Exited')

    drop_cols = ['id', 'CustomerId', 'Surname']
    data = create_inputs_targets(df_dict, drop_cols=drop_cols, target_col='Exited')

    numeric_cols = data['train_inputs'].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data['train_inputs'].select_dtypes(include='object').columns.tolist()

    scaler = None
    if scaler_numeric:
        scaler = scale_numeric_features(data, numeric_cols)

    encoder = encode_categorical_features(data, categorical_cols)

    input_cols = numeric_cols + data['encoded_cols']

    X_train = data['train_inputs'][input_cols]
    X_val = data['val_inputs'][input_cols]

    return X_train, data['train_targets'], X_val, data['val_targets'], input_cols, scaler, encoder


def preprocess_new_data(raw_df: pd.DataFrame, input_cols: List[str], encoder: OneHotEncoder,
                        scaler: Optional[MinMaxScaler] = None) -> pd.DataFrame:
    """
    Preprocess new (test) data using already fitted scaler and encoder.
    """
    df_processed = raw_df.copy()

    categorical_cols = list(encoder.feature_names_in_)
    numeric_cols = [col for col in df_processed.columns if
                    col not in categorical_cols and col not in ['id', 'CustomerId', 'Surname', 'Exited']]

    if scaler is not None:
        df_processed[numeric_cols] = scaler.transform(df_processed[numeric_cols])

    encoded = encoder.transform(df_processed[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df_processed.index)

    df_processed = pd.concat([df_processed, encoded_df], axis=1)

    return df_processed[input_cols]