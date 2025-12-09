from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import pandas as pd # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import torch # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from sklearn.ensemble import RandomForestRegressor # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from sklearn.preprocessing import StandardScaler # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from torch.utils.data import Dataset # pyright: ignore[reportMissingModuleSource, reportMissingImports]


# ========== Global Configurations ==========
CLASS_NAMES = ['Did Not Survive', 'Survived']

CUR_MODEL = 'xgboost'  # Options: 'RandomForest', 'gradientboosting', 'logisticregression', 'svm', 'knn', 'xgboost', 'MLP'
KAGGLE_SUBMIT = False
EARLY_STOPPING = True
NUM_WORKERS = 0
IMMFEATURE = True  # When True, use notebook-style feature engineering


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 128
    num_epochs: int = 250
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_ratio: float = 0.2
    patience: int = 30
    seed: int = 45
    seeds: tuple[int, ...] = (45, 2025, 777)

TITLE_REPLACEMENTS = {
    'Mlle': 'Miss',
    'Ms': 'Miss',
    'Mme': 'Mrs',
    'Lady': 'Royalty',
    'Countess': 'Royalty',
    'Sir': 'Royalty',
    'Jonkheer': 'Royalty',
    'Don': 'Royalty',
    'Dona': 'Royalty',
    'Col': 'Officer',
    'Major': 'Officer',
    'Capt': 'Officer',
    'Dr': 'Officer',
    'Rev': 'Officer'
}

COMMON_TITLES = {'Mr', 'Mrs', 'Miss', 'Master'}


def set_global_seed(seed: int) -> None:
    """Seed numpy and torch for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_title(name: pd.Series) -> pd.Series:
    titles = name.str.extract(' ([A-Za-z]+)\.', expand=False)
    titles = titles.replace(TITLE_REPLACEMENTS)
    titles = titles.fillna('Rare')
    titles = titles.where(titles.isin(COMMON_TITLES), 'Rare')
    return titles


def _engineer_features_default(df: pd.DataFrame) -> pd.DataFrame:
    processed = df.copy()

    processed['Title'] = _extract_title(processed['Name'])
    processed['Sex'] = (processed['Sex'] == 'female').astype(int)

    embarked_mode = processed['Embarked'].dropna().mode()
    processed['Embarked'] = processed['Embarked'].fillna(embarked_mode.iloc[0] if not embarked_mode.empty else 'S')
    processed['Fare'] = processed['Fare'].fillna(processed['Fare'].median())

    processed['FamilySize'] = processed['SibSp'] + processed['Parch'] + 1
    processed['IsAlone'] = (processed['FamilySize'] == 1).astype(int)

    processed['CabinDeck'] = processed['Cabin'].str[0].fillna('N')
    processed['HasCabin'] = (~processed['Cabin'].isna()).astype(int)

    processed['TicketGroup'] = processed.groupby('Ticket')['Ticket'].transform('count').clip(0, 4)

    processed['Age'] = processed.groupby(['Title', 'Pclass'])['Age'].transform(lambda col: col.fillna(col.median()))
    processed['Age'] = processed['Age'].fillna(processed['Age'].median())
    processed['AgeBin'] = pd.cut(
        processed['Age'],
        bins=[0, 12, 20, 40, 60, 80],
        labels=False,
        include_lowest=True
    ).astype('Int64').fillna(2)

    fare_bins = [-1, 8, 15, 31, 100, np.inf]
    processed['FareBin'] = pd.cut(
        processed['Fare'],
        bins=fare_bins,
        labels=False,
        include_lowest=True
    ).astype('Int64').fillna(2)

    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    features = processed.drop(columns=drop_cols, errors='ignore')

    categorical_cols = ['Embarked', 'Title', 'CabinDeck', 'Pclass']
    features[categorical_cols] = features[categorical_cols].astype('category')
    features = pd.get_dummies(features, columns=categorical_cols, drop_first=True)
    features = features.fillna(0.0)
    return features


def _engineer_features_notebook(df: pd.DataFrame) -> pd.DataFrame:
    processed = df.copy()

    processed['Embarked'] = processed['Embarked'].fillna('S')
    processed['Fare'] = processed['Fare'].fillna(processed['Fare'].mean())

    title1 = processed['Name'].str.split(', ', expand=True)[1]
    processed['Title1'] = title1.str.split('.', expand=True)[0]
    processed['Title2'] = processed['Title1'].replace(
        ['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'the Countess', 'Jonkheer', 'Col', 'Rev', 'Capt', 'Sir', 'Don', 'Dona'],
        ['Miss', 'Mrs', 'Miss', 'Mr', 'Mr', 'Mrs', 'Mrs', 'Mr', 'Mr', 'Mr', 'Mr', 'Mr', 'Mr', 'Mrs']
    )

    processed['Ticket_info'] = processed['Ticket'].apply(
        lambda x: x.replace('.', '').replace('/', '').strip().split(' ')[0] if not str(x).isdigit() else 'X'
    )

    processed['Cabin'] = processed['Cabin'].apply(lambda x: str(x)[0] if not pd.isnull(x) else 'NoCabin')
    processed['Family_Size'] = processed['Parch'] + processed['SibSp']

    categorical_cols = ['Sex', 'Embarked', 'Pclass', 'Title1', 'Title2', 'Cabin', 'Ticket_info']
    for col in categorical_cols:
        processed[col] = processed[col].astype('category').cat.codes

    age_columns = ['Embarked', 'Fare', 'Pclass', 'Sex', 'Family_Size', 'Title1', 'Title2', 'Cabin', 'Ticket_info']
    data_age_null = processed[processed['Age'].isnull()]
    data_age_not_null = processed[processed['Age'].notnull()]
    if not data_age_null.empty and not data_age_not_null.empty:
        outlier_mask = (
            (np.abs(data_age_not_null['Fare'] - data_age_not_null['Fare'].mean()) > (4 * data_age_not_null['Fare'].std())) |
            (np.abs(data_age_not_null['Family_Size'] - data_age_not_null['Family_Size'].mean()) > (4 * data_age_not_null['Family_Size'].std()))
        )
        age_train = data_age_not_null.loc[~outlier_mask]
        rf_model_age = RandomForestRegressor(n_estimators=2000, random_state=42)
        rf_model_age.fit(age_train[age_columns], age_train['Age'])
        processed.loc[data_age_null.index, 'Age'] = rf_model_age.predict(data_age_null[age_columns])

    features = processed[
        ['Age', 'Embarked', 'Fare', 'Pclass', 'Sex', 'Family_Size', 'Title2', 'Ticket_info', 'Cabin']
    ].copy()
    features = features.fillna(0.0)
    return features


def _engineer_features(df: pd.DataFrame, immfeature: bool = False) -> pd.DataFrame:
    if immfeature:
        return _engineer_features_notebook(df)
    return _engineer_features_default(df)


def prepare_titanic_data(
    train_csv: str,
    test_csv: str,
    immfeature: bool | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load Titanic CSVs, engineer features, and return numpy arrays."""
    if immfeature is None:
        immfeature = IMMFEATURE

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    y = train_df['Survived'].values.astype(np.int64)

    combined = pd.concat(
        [train_df.drop(columns=['Survived']), test_df],
        axis=0,
        ignore_index=True
    )
    combined_features = _engineer_features(combined, immfeature=immfeature)

    train_features = combined_features.iloc[: len(train_df)].reset_index(drop=True)
    test_features = combined_features.iloc[len(train_df):].reset_index(drop=True)

    feature_names = combined_features.columns.tolist()
    passenger_ids = test_df['PassengerId'].values.astype(np.int64)

    if immfeature:
        train_scaled = train_features.to_numpy(dtype=np.float32)
        test_scaled = test_features.to_numpy(dtype=np.float32)
    else:
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_features).astype(np.float32)
        test_scaled = scaler.transform(test_features).astype(np.float32)

    return train_scaled, y, test_scaled, passenger_ids, feature_names


class TitanicDataset(Dataset):
    """Torch Dataset for the Titanic tabular task."""

    def __init__(self, features: np.ndarray, labels: np.ndarray | None = None, ids: np.ndarray | None = None):
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64)) if labels is not None else None
        self.ids = torch.from_numpy(ids.astype(np.int64)) if ids is not None else None

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        x = self.features[idx]
        if self.labels is not None:
            return x, self.labels[idx]
        if self.ids is not None:
            return x, self.ids[idx]
        return x
        