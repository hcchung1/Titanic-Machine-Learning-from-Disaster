from __future__ import annotations

import os # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import subprocess # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import sys # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import warnings # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from datetime import datetime
from dataclasses import replace
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource, reportMissingImports]
matplotlib.use('Agg')
import numpy as np # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import pandas as pd # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import seaborn as sns # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import torch # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import torch.nn as nn # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import torch.optim as optim # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from loguru import logger # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from sklearn.linear_model import LogisticRegression # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from sklearn.neighbors import KNeighborsClassifier # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from sklearn.svm import SVC # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from sklearn.model_selection import GridSearchCV, train_test_split # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from tqdm.auto import tqdm # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import xgboost
from xgboost import XGBClassifier # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from xgboost.callback import EarlyStopping
import csv
from typing import Iterable, List, Tuple
from network import TitanicMLP, evaluate, test, train_epoch
from utils import (
    CLASS_NAMES,
    CUR_MODEL,
    EARLY_STOPPING,
    KAGGLE_SUBMIT,
    NUM_WORKERS,
    FEATURE,
    TrainingConfig,
    TitanicDataset,
    prepare_titanic_data,
    set_global_seed,
)


warnings.filterwarnings('ignore', category=FutureWarning)

def load_labels(path: Path) -> List[Tuple[int, int]]:
	"""Load (PassengerId, Survived) rows from a CSV file."""
	if not path.exists():
		raise FileNotFoundError(f"CSV not found: {path}")

	rows: List[Tuple[int, int]] = []
	with path.open(newline="", encoding="utf-8-sig") as handle:
		reader = csv.DictReader(handle)
		required = {"PassengerId", "Survived"}
		if not required.issubset(reader.fieldnames or set()):
			raise ValueError(
				f"CSV must contain headers {required}, found: {reader.fieldnames}"
			)

		for line_number, row in enumerate(reader, start=2):
			try:
				pid = int(row["PassengerId"])
				survived = int(row["Survived"])
			except (TypeError, ValueError) as exc:
				raise ValueError(
					f"Invalid data at line {line_number} in {path}: {row}"
				) from exc
			rows.append((pid, survived))

	if not rows:
		raise ValueError(f"No data rows found in {path}")

	return rows


def compute_accuracy(
	ground_truth: Iterable[Tuple[int, int]], predictions: Iterable[Tuple[int, int]]
) -> Tuple[float, int, int]:
	"""Return (accuracy, matches, total) after strict ID alignment."""

	gt_list = list(ground_truth)
	pred_list = list(predictions)

	if len(gt_list) != len(pred_list):
		raise ValueError(
			f"Row count mismatch: ground truth has {len(gt_list)}, "
			f"predictions have {len(pred_list)}"
		)

	mismatched_ids = [
		(idx + 1, gt_id, pred_id)
		for idx, ((gt_id, _), (pred_id, _)) in enumerate(zip(gt_list, pred_list))
		if gt_id != pred_id
	]
	if mismatched_ids:
		first = mismatched_ids[0]
		raise ValueError(
			"PassengerId mismatch at aligned rows; first difference "
			f"at row {first[0]}: ground truth {first[1]} vs predictions {first[2]}"
		)

	matches = sum(
		1 for (_, gt_survived), (_, pred_survived) in zip(gt_list, pred_list)
		if gt_survived == pred_survived
	)
	total = len(gt_list)
	accuracy = matches / total if total else 0.0
	return accuracy, matches, total


def score_submission(submission_path: Path, ground_truth_path: Path):
    """Compute accuracy of a submission CSV against ground truth."""

    ground_truth = load_labels(ground_truth_path)
    predictions = load_labels(submission_path)
    accuracy, matches, total = compute_accuracy(ground_truth, predictions)
    return accuracy, matches, total


def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curves')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f'Training history plot saved to {save_path}')


def plot_confusion_matrix(labels, preds, class_names, save_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Validation Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f'Confusion matrix saved to {save_path}')


def submit_to_kaggle(file_path: str, competition: str, message: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Submission file not found: {file_path}')

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi # pyright: ignore[reportMissingModuleSource, reportMissingImports]

        api = KaggleApi()
        api.authenticate()
        logger.info(f"Submitting '{file_path}' to Kaggle competition '{competition}'")
        api.competition_submit(file_path, message, competition)
    except ImportError:
        logger.info('Falling back to Kaggle CLI for submission')
        subprocess.run(
            ['kaggle', 'competitions', 'submit', '-c', competition, '-f', file_path, '-m', message],
            check=True
        )


def train_random_forest(
    train_features,
    train_labels,
    test_features,
    passenger_ids,
    cfg,
    output_dir,
    log_name,
    today,
    cm_name,
    submission_path
):
    X_train, X_val, y_train, y_val = train_test_split(
        train_features,
        train_labels,
        test_size=cfg.val_ratio,
        random_state=cfg.seed,
        stratify=train_labels
    )

    # rf = RandomForestClassifier(
    #     n_estimators=1000,
    #     max_depth=None,
    #     min_samples_split=2,
    #     min_samples_leaf=1,
    #     max_features='sqrt',
    #     class_weight='balanced_subsample',
    #     random_state=cfg.seed,
    #     n_jobs=-1
    # )

    rf = RandomForestClassifier(criterion='gini', 
                             n_estimators=1000,
                             min_samples_split=12,
                             min_samples_leaf=1,
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1) 
    logger.info('Training RandomForestClassifier...')
    rf.fit(X_train, y_train)

    val_preds = rf.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds) * 100.0
    report = classification_report(y_val, val_preds, target_names=CLASS_NAMES, digits=4)
    logger.info(f'RandomForest validation accuracy: {val_acc:.2f}%')
    logger.info('Validation report:\n' + report)
    plot_confusion_matrix(y_val, val_preds, CLASS_NAMES, cm_name)

    test_preds = rf.predict(test_features)
    submission_df = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': test_preds})
    submission_df = submission_df.sort_values('PassengerId')
    submission_df.to_csv(submission_path, index=False)
    logger.info(f'RandomForest submission saved to {submission_path}')

    model_path = os.path.join(output_dir, f'{log_name}_random_forest_{today}.pth')
    torch.save(rf, model_path)
    logger.info(f'RandomForest model serialized to {model_path}')

    return val_acc, submission_path, model_path, rf


def train_gradient_boosting(
    train_features,
    train_labels,
    test_features,
    passenger_ids,
    cfg,
    output_dir,
    log_name,
    today,
    cm_name,
    submission_path
):
    X_train, X_val, y_train, y_val = train_test_split(
        train_features,
        train_labels,
        test_size=cfg.val_ratio,
        random_state=cfg.seed,
        stratify=train_labels
    )

    gb_base = GradientBoostingClassifier(random_state=cfg.seed)
    param_grid = {
        'n_estimators': [300, 500, 800],
        'learning_rate': [0.01, 0.02, 0.05],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [1, 2],
        'subsample': [0.85, 0.9, 1.0]
    }


    """
    n_estimators: 400,600,800

    learning_rate: 0.01,0.02,0.05

    max_depth: 2,3,4

    min_samples_split: 2,4,6

    min_samples_leaf: 1,2,4

    subsample: 0.8,0.9,1.0
    """

    logger.info('Running GridSearchCV for GradientBoostingClassifier...')
    grid = GridSearchCV(
        gb_base,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)

    best_model: GradientBoostingClassifier = grid.best_estimator_
    logger.info(
        f'Best GradientBoosting params: {grid.best_params_} '
        f'(cv accuracy {grid.best_score_:.4f})'
    )

    val_preds = best_model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds) * 100.0
    report = classification_report(y_val, val_preds, target_names=CLASS_NAMES, digits=4)
    logger.info(f'GradientBoosting validation accuracy: {val_acc:.2f}%')
    logger.info('Validation report:\n' + report)
    plot_confusion_matrix(y_val, val_preds, CLASS_NAMES, cm_name)

    test_preds = best_model.predict(test_features)
    submission_df = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': test_preds})
    submission_df = submission_df.sort_values('PassengerId')
    submission_df.to_csv(submission_path, index=False)
    logger.info(f'GradientBoosting submission saved to {submission_path}')

    model_path = os.path.join(output_dir, f'{log_name}_gradient_boosting_{today}.pth')
    torch.save(best_model, model_path)
    logger.info(f'GradientBoosting model serialized to {model_path}')

    return val_acc, submission_path, model_path, best_model


def train_xgboost(
    train_features,
    train_labels,
    test_features,
    passenger_ids,
    cfg,
    output_dir,
    log_name,
    today,
    cm_name,
    submission_path
):
    X_train, X_val, y_train, y_val = train_test_split(
        train_features,
        train_labels,
        test_size=cfg.val_ratio,
        random_state=cfg.seed,
        stratify=train_labels
    )

    cpu_count = os.cpu_count() or 1
    
    xgb = XGBClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=1.0,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=cfg.seed,
        n_jobs=max(1, cpu_count - 1),
        early_stopping_rounds=50,  # 放在這裡
    )
    logger.info('Training XGBoostClassifier...')
    xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    # xgb.fit(X_train, y_train)
    val_preds = xgb.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds) * 100.0
    report = classification_report(y_val, val_preds, target_names=CLASS_NAMES, digits=4)
    logger.info(f'XGBoost validation accuracy: {val_acc:.2f}%')
    logger.info('Validation report:\n' + report)
    plot_confusion_matrix(y_val, val_preds, CLASS_NAMES, cm_name)

    test_preds = xgb.predict(test_features)
    submission_df = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': test_preds})
    submission_df = submission_df.sort_values('PassengerId')
    submission_df.to_csv(submission_path, index=False)
    logger.info(f'XGBoost submission saved to {submission_path}')

    model_path = os.path.join(output_dir, f'{log_name}_xgboost_{today}.pth')
    torch.save(xgb, model_path)
    logger.info(f'XGBoost model serialized to {model_path}')

    return val_acc, submission_path, model_path, xgb


def train_rfxgb(
    train_features,
    train_labels,
    test_features,
    passenger_ids,
    cfg,
    output_dir,
    log_name,
    today,
    cm_name,
    submission_path
):
    """Train RF and XGB on the same split and soft-vote their probabilities."""

    X_train, X_val, y_train, y_val = train_test_split(
        train_features,
        train_labels,
        test_size=cfg.val_ratio,
        random_state=cfg.seed,
        stratify=train_labels
    )

    rf = RandomForestClassifier(
        criterion='gini',
        n_estimators=1000,
        min_samples_split=12,
        min_samples_leaf=1,
        oob_score=True,
        random_state=cfg.seed,
        n_jobs=-1
    )
    logger.info('Training RandomForestClassifier (RFXGB ensemble part)...')
    rf.fit(X_train, y_train)

    cpu_count = os.cpu_count() or 1
    xgb = XGBClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=1.0,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=cfg.seed,
        n_jobs=max(1, cpu_count - 1),
        early_stopping_rounds=50,
    )
    logger.info('Training XGBoostClassifier (RFXGB ensemble part)...')
    xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    rf_val_proba = rf.predict_proba(X_val)
    xgb_val_proba = xgb.predict_proba(X_val)
    avg_val_proba = (rf_val_proba + xgb_val_proba) / 2.0
    val_preds = np.argmax(avg_val_proba, axis=1)

    val_acc = accuracy_score(y_val, val_preds) * 100.0
    report = classification_report(y_val, val_preds, target_names=CLASS_NAMES, digits=4)
    logger.info(f'RFXGB validation accuracy: {val_acc:.2f}%')
    logger.info('Validation report:\n' + report)
    plot_confusion_matrix(y_val, val_preds, CLASS_NAMES, cm_name)

    rf_test_proba = rf.predict_proba(test_features)
    xgb_test_proba = xgb.predict_proba(test_features)
    avg_test_proba = (rf_test_proba + xgb_test_proba) / 2.0
    test_preds = np.argmax(avg_test_proba, axis=1)

    submission_df = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': test_preds})
    submission_df = submission_df.sort_values('PassengerId')
    submission_df.to_csv(submission_path, index=False)
    logger.info(f'RFXGB submission saved to {submission_path}')

    model_path = os.path.join(output_dir, f'{log_name}_rfxgb_{today}.pth')
    torch.save({'rf': rf, 'xgb': xgb}, model_path)
    logger.info(f'RFXGB models serialized to {model_path}')

    return val_acc, submission_path, model_path, {'rf': rf, 'xgb': xgb}


def train_rfxgb_seed_only(
    train_features,
    train_labels,
    cfg,
    log_name,
    today
):
    """Train RF and XGB for a given seed, return individual models and accuracies."""

    X_train, X_val, y_train, y_val = train_test_split(
        train_features,
        train_labels,
        test_size=cfg.val_ratio,
        random_state=cfg.seed,
        stratify=train_labels
    )

    rf = RandomForestClassifier(
        criterion='gini',
        n_estimators=1000,
        min_samples_split=12,
        min_samples_leaf=1,
        oob_score=True,
        random_state=cfg.seed,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_val_preds = rf.predict(X_val)
    rf_val_acc = accuracy_score(y_val, rf_val_preds) * 100.0

    cpu_count = os.cpu_count() or 1
    xgb = XGBClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=1.0,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=cfg.seed,
        n_jobs=max(1, cpu_count - 1),
        early_stopping_rounds=50,
    )
    xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    xgb_val_preds = xgb.predict(X_val)
    xgb_val_acc = accuracy_score(y_val, xgb_val_preds) * 100.0

    logger.info(
        f'RFXGB seed {cfg.seed}: RF val {rf_val_acc:.2f}%, XGB val {xgb_val_acc:.2f}%'
    )

    return {
        'seed': cfg.seed,
        'rf_acc': rf_val_acc,
        'xgb_acc': xgb_val_acc,
        'rf_model': rf,
        'xgb_model': xgb,
        'log_name': log_name,
        'today': today,
    }


def train_rfxgb_best_over_seeds(
    train_features,
    train_labels,
    test_features,
    passenger_ids,
    base_cfg,
    seeds,
    output_dir,
    log_name,
    today,
    ground_truth_path
):
    """Train RF/XGB across seeds separately, pick best RF and best XGB, then soft-vote."""

    if not seeds:
        raise ValueError('No seeds provided for RFXGB')

    best_rf = None
    best_xgb = None
    best_rf_acc = -1.0
    best_xgb_acc = -1.0

    for seed in seeds:
        cfg = replace(base_cfg, seed=seed)
        set_global_seed(seed)
        seed_log_name = f'{log_name}_seed{seed}'
        result = train_rfxgb_seed_only(
            train_features,
            train_labels,
            cfg,
            seed_log_name,
            today
        )
        if result['rf_acc'] > best_rf_acc:
            best_rf_acc = result['rf_acc']
            best_rf = result['rf_model']
            best_rf_seed = seed
        if result['xgb_acc'] > best_xgb_acc:
            best_xgb_acc = result['xgb_acc']
            best_xgb = result['xgb_model']
            best_xgb_seed = seed

    if best_rf is None or best_xgb is None:
        raise RuntimeError('Failed to obtain best RF or XGB model for RFXGB')

    rf_test_proba = best_rf.predict_proba(test_features)
    xgb_test_proba = best_xgb.predict_proba(test_features)
    avg_test_proba = (rf_test_proba + xgb_test_proba) / 2.0
    test_preds = np.argmax(avg_test_proba, axis=1)

    submission_path = os.path.join(output_dir, f'{log_name}_best_rfxgb_submission.csv')
    submission_df = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': test_preds})
    submission_df = submission_df.sort_values('PassengerId')
    submission_df.to_csv(submission_path, index=False)
    logger.info(
        f'RFXGB (best seeds) submission saved to {submission_path}; '
        f'best RF seed {best_rf_seed} ({best_rf_acc:.2f}%), '
        f'best XGB seed {best_xgb_seed} ({best_xgb_acc:.2f}%)'
    )

    model_path = os.path.join(output_dir, f'{log_name}_best_rfxgb_{today}.pth')
    torch.save({'rf': best_rf, 'xgb': best_xgb, 'rf_seed': best_rf_seed, 'xgb_seed': best_xgb_seed}, model_path)
    logger.info(f'Best RFXGB models serialized to {model_path}')

    summary_path = os.path.join(output_dir, f'{log_name}_best_rfxgb_summary_{today}.txt')
    with open(summary_path, 'w', encoding='utf-8') as handle:
        handle.write(f'Best RF seed: {best_rf_seed}, val_acc={best_rf_acc:.4f}\n')
        handle.write(f'Best XGB seed: {best_xgb_seed}, val_acc={best_xgb_acc:.4f}\n')

    submission_accuracy, matches, total = score_submission(Path(submission_path), ground_truth_path)
    logger.info(
        f'RFXGB best-seed submission accuracy vs corrected ground truth: '
        f'{submission_accuracy:.4f} ({matches}/{total})'
    )

    ensemble_val_acc = float((best_rf_acc + best_xgb_acc) / 2.0)
    return ensemble_val_acc, submission_path, model_path, {'rf': best_rf, 'xgb': best_xgb}


def train_logistic_regression(
    train_features,
    train_labels,
    test_features,
    passenger_ids,
    cfg,
    output_dir,
    log_name,
    today,
    cm_name,
    submission_path
):
    X_train, X_val, y_train, y_val = train_test_split(
        train_features,
        train_labels,
        test_size=cfg.val_ratio,
        random_state=cfg.seed,
        stratify=train_labels
    )

    lr = LogisticRegression(
        max_iter=5000,
        class_weight='balanced',
        solver='liblinear',
        n_jobs=-1,
        random_state=cfg.seed
    )
    logger.info('Training LogisticRegression...')
    lr.fit(X_train, y_train)

    val_preds = lr.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds) * 100.0
    report = classification_report(y_val, val_preds, target_names=CLASS_NAMES, digits=4)
    logger.info(f'LogisticRegression validation accuracy: {val_acc:.2f}%')
    logger.info('Validation report:\n' + report)
    plot_confusion_matrix(y_val, val_preds, CLASS_NAMES, cm_name)

    test_preds = lr.predict(test_features)
    submission_df = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': test_preds})
    submission_df = submission_df.sort_values('PassengerId')
    submission_df.to_csv(submission_path, index=False)
    logger.info(f'LogisticRegression submission saved to {submission_path}')

    model_path = os.path.join(output_dir, f'{log_name}_logistic_regression_{today}.pth')
    torch.save(lr, model_path)
    logger.info(f'LogisticRegression model serialized to {model_path}')

    return val_acc, submission_path, model_path, lr


def train_svm(
    train_features,
    train_labels,
    test_features,
    passenger_ids,
    cfg,
    output_dir,
    log_name,
    today,
    cm_name,
    submission_path
):
    X_train, X_val, y_train, y_val = train_test_split(
        train_features,
        train_labels,
        test_size=cfg.val_ratio,
        random_state=cfg.seed,
        stratify=train_labels
    )

    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', class_weight='balanced', probability=True)),
    ])
    svm_param_grid = {
        'svc__C': [0.5, 1.0, 2.0, 5.0],
        'svc__gamma': ['scale', 0.05, 0.1],
    }
    logger.info('Running GridSearchCV for SVM (RBF kernel) with scaling...')
    svm_grid = GridSearchCV(
        svm_pipeline,
        param_grid=svm_param_grid,
        scoring='accuracy',
        cv=3,
        n_jobs=-1,
        verbose=0,
    )
    svm_grid.fit(X_train, y_train)

    best_model: Pipeline = svm_grid.best_estimator_
    logger.info(
        f'Best SVM params: {svm_grid.best_params_} '
        f'(cv accuracy {svm_grid.best_score_:.4f})'
    )

    val_preds = best_model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds) * 100.0
    report = classification_report(y_val, val_preds, target_names=CLASS_NAMES, digits=4)
    logger.info(f'SVM validation accuracy: {val_acc:.2f}%')
    logger.info('Validation report:\n' + report)
    plot_confusion_matrix(y_val, val_preds, CLASS_NAMES, cm_name)

    test_preds = best_model.predict(test_features)
    submission_df = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': test_preds})
    submission_df = submission_df.sort_values('PassengerId')
    submission_df.to_csv(submission_path, index=False)
    logger.info(f'SVM submission saved to {submission_path}')

    model_path = os.path.join(output_dir, f'{log_name}_svm_{today}.pth')
    torch.save(best_model, model_path)
    logger.info(f'SVM model serialized to {model_path}')

    return val_acc, submission_path, model_path, best_model


def train_knn(
    train_features,
    train_labels,
    test_features,
    passenger_ids,
    cfg,
    output_dir,
    log_name,
    today,
    cm_name,
    submission_path
):
    X_train, X_val, y_train, y_val = train_test_split(
        train_features,
        train_labels,
        test_size=cfg.val_ratio,
        random_state=cfg.seed,
        stratify=train_labels
    )

    knn_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(metric='minkowski', n_jobs=-1)),
    ])
    knn_param_grid = {
        'knn__n_neighbors': [7, 11, 15, 21],
        'knn__weights': ['uniform', 'distance'],
        'knn__p': [1, 2],
    }
    logger.info('Running GridSearchCV for KNeighborsClassifier with scaling...')
    knn_grid = GridSearchCV(
        knn_pipeline,
        param_grid=knn_param_grid,
        scoring='accuracy',
        cv=3,
        n_jobs=-1,
        verbose=0,
    )
    knn_grid.fit(X_train, y_train)

    best_model: Pipeline = knn_grid.best_estimator_
    logger.info(
        f'Best KNN params: {knn_grid.best_params_} '
        f'(cv accuracy {knn_grid.best_score_:.4f})'
    )

    val_preds = best_model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds) * 100.0
    report = classification_report(y_val, val_preds, target_names=CLASS_NAMES, digits=4)
    logger.info(f'KNN validation accuracy: {val_acc:.2f}%')
    logger.info('Validation report:\n' + report)
    plot_confusion_matrix(y_val, val_preds, CLASS_NAMES, cm_name)

    test_preds = best_model.predict(test_features)
    submission_df = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': test_preds})
    submission_df = submission_df.sort_values('PassengerId')
    submission_df.to_csv(submission_path, index=False)
    logger.info(f'KNN submission saved to {submission_path}')

    model_path = os.path.join(output_dir, f'{log_name}_knn_{today}.pth')
    torch.save(best_model, model_path)
    logger.info(f'KNN model serialized to {model_path}')

    return val_acc, submission_path, model_path, best_model


def train_ensemble(
    train_features,
    train_labels,
    test_features,
    passenger_ids,
    cfg,
    output_dir,
    log_name,
    today,
    cm_name,
    submission_path
):
    """Train all base models, pick top-3 by validation accuracy, then soft-vote."""

    base_trainers = [
        ('randomforest', train_random_forest),
        ('gradientboosting', train_gradient_boosting),
        ('logisticregression', train_logistic_regression),
        ('svm', train_svm),
        ('knn', train_knn),
        ('xgboost', train_xgboost),
    ]

    results = []
    for name, trainer in base_trainers:
        model_log_name = f'{log_name}_{name}'
        model_submission_path = os.path.join(output_dir, f'{model_log_name}_submission.csv')
        model_cm_name = os.path.join(output_dir, f'{model_log_name}_cm_{today}.png')
        val_acc, sub_path, model_path, model_obj = trainer(
            train_features,
            train_labels,
            test_features,
            passenger_ids,
            cfg,
            output_dir,
            model_log_name,
            today,
            model_cm_name,
            model_submission_path
        )
        results.append({
            'name': name,
            'val_acc': val_acc,
            'submission': sub_path,
            'model_path': model_path,
            'model': model_obj,
        })

    if not results:
        raise RuntimeError('No base models were trained for ensemble.')

    results_sorted = sorted(results, key=lambda x: x['val_acc'], reverse=True)
    top3 = results_sorted[:3]
    logger.info('Top-3 models for ensemble (val acc): ' + ', '.join(
        f"{r['name']}={r['val_acc']:.2f}%" for r in top3
    ))

    probas = []
    for item in top3:
        model = item['model']
        if hasattr(model, 'predict_proba'):
            probas.append(model.predict_proba(test_features))
        else:
            logger.warning(f"Model {item['name']} lacks predict_proba; skipping in soft vote")

    if not probas:
        raise RuntimeError('No models provided probabilities for ensemble voting.')

    avg_proba = np.mean(np.stack(probas, axis=0), axis=0)
    final_preds = np.argmax(avg_proba, axis=1)

    ensemble_submission_path = os.path.join(output_dir, f'{log_name}_ensemble_submission.csv')
    submission_df = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': final_preds})
    submission_df = submission_df.sort_values('PassengerId')
    submission_df.to_csv(ensemble_submission_path, index=False)
    logger.info(f'Ensemble submission saved to {ensemble_submission_path}')

    ensemble_summary_path = os.path.join(output_dir, f'{log_name}_ensemble_top3_{today}.txt')
    with open(ensemble_summary_path, 'w', encoding='utf-8') as handle:
        handle.write('Top-3 models by validation accuracy\n')
        for item in top3:
            handle.write(f"{item['name']}: {item['val_acc']:.4f}, model={item['model_path']}\n")

    ensemble_val_acc = float(np.mean([item['val_acc'] for item in top3]))
    return ensemble_val_acc, ensemble_submission_path, ensemble_summary_path, top3


def main():
    base_cfg = TrainingConfig()
    seed_list = getattr(base_cfg, 'seeds', (base_cfg.seed,)) or (base_cfg.seed,)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    output_dir = os.path.join(current_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    ground_truth_path = Path(data_dir) / 'submission_corrected.csv'

    log_name = f'{CUR_MODEL}'
    if EARLY_STOPPING:
        log_name += '_ES'
    today = datetime.now().strftime('%Y-%m-%d')

    logger.remove()
    logger.add(sys.stderr, format='<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>')
    logger.add(
        os.path.join(output_dir, f'{log_name}_all_seeds_{today}.log'),
        format='{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}',
        colorize=False
    )

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{torch.cuda.device_count() - 1}')
    else:
        device = torch.device('cpu')
    logger.info(f'Using device: {device}')

    train_csv = os.path.join(data_dir, 'train.csv')
    test_csv = os.path.join(data_dir, 'test.csv')
    train_features, train_labels, test_features, passenger_ids, feature_names = prepare_titanic_data(
        train_csv,
        test_csv,
        feature=FEATURE
    )
    logger.info(f'Loaded {train_features.shape[0]} training samples with {train_features.shape[1]} features')

    model_name = CUR_MODEL.lower()
    best_result = {
        'val_acc': -1.0,
        'seed': None,
        'submission': None,
        'model_path': None
    }

    if model_name == 'rfxgb':
        val_acc, submission, model_path, _model_obj = train_rfxgb_best_over_seeds(
            train_features,
            train_labels,
            test_features,
            passenger_ids,
            base_cfg,
            seed_list,
            output_dir,
            log_name,
            today,
            ground_truth_path
        )
        best_result = {
            'val_acc': val_acc,
            'seed': 'best_rf_xgb',
            'submission': submission,
            'model_path': model_path
        }
    else:
        for seed in seed_list:
            cfg = replace(base_cfg, seed=seed)
            set_global_seed(seed)

            seed_log_name = f'{log_name}_seed{seed}'
            plot_name = os.path.join(output_dir, f'{seed_log_name}_training_{today}.png')
            cm_name = os.path.join(output_dir, f'{seed_log_name}_cm_{today}.png')
            best_model_path = os.path.join(output_dir, f'best_{seed_log_name}_{today}.pth')
            submission_path = os.path.join(output_dir, f'{seed_log_name}_submission.csv')
            seed_log_path = os.path.join(output_dir, f'{seed_log_name}_{today}.log')

            file_sink = logger.add(
                seed_log_path,
                format='{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}',
                colorize=False
            )

            try:
                if model_name == 'randomforest':
                    val_acc, submission, model_path, _model_obj = train_random_forest(
                        train_features,
                        train_labels,
                        test_features,
                        passenger_ids,
                        cfg,
                        output_dir,
                        seed_log_name,
                        today,
                        cm_name,
                        submission_path
                    )
                elif model_name == 'gradientboosting':
                    val_acc, submission, model_path, _model_obj = train_gradient_boosting(
                        train_features,
                        train_labels,
                        test_features,
                        passenger_ids,
                        cfg,
                        output_dir,
                        seed_log_name,
                        today,
                        cm_name,
                        submission_path
                    )
                elif model_name == 'logisticregression':
                    val_acc, submission, model_path, _model_obj = train_logistic_regression(
                        train_features,
                        train_labels,
                        test_features,
                        passenger_ids,
                        cfg,
                        output_dir,
                        seed_log_name,
                        today,
                        cm_name,
                        submission_path
                    )
                elif model_name == 'svm':
                    val_acc, submission, model_path, _model_obj = train_svm(
                        train_features,
                        train_labels,
                        test_features,
                        passenger_ids,
                        cfg,
                        output_dir,
                        seed_log_name,
                        today,
                        cm_name,
                        submission_path
                    )
                elif model_name == 'knn':
                    val_acc, submission, model_path, _model_obj = train_knn(
                        train_features,
                        train_labels,
                        test_features,
                        passenger_ids,
                        cfg,
                        output_dir,
                        seed_log_name,
                        today,
                        cm_name,
                        submission_path
                    )
                elif model_name == 'xgboost':
                    val_acc, submission, model_path, _model_obj = train_xgboost(
                        train_features,
                        train_labels,
                        test_features,
                        passenger_ids,
                        cfg,
                        output_dir,
                        seed_log_name,
                        today,
                        cm_name,
                        submission_path
                    )
                elif model_name == 'ensemble':
                    val_acc, submission, model_path, _model_obj = train_ensemble(
                        train_features,
                        train_labels,
                        test_features,
                        passenger_ids,
                        cfg,
                        output_dir,
                        seed_log_name,
                        today,
                        cm_name,
                        submission_path
                    )
                else: # MLP Training
                    X_train, X_val, y_train, y_val = train_test_split(
                        train_features,
                        train_labels,
                        test_size=cfg.val_ratio,
                        random_state=cfg.seed,
                        stratify=train_labels
                    )

                    train_loader = DataLoader(
                        TitanicDataset(X_train, y_train),
                        batch_size=cfg.batch_size,
                        shuffle=True,
                        num_workers=NUM_WORKERS
                    )
                    val_loader = DataLoader(
                        TitanicDataset(X_val, y_val),
                        batch_size=cfg.batch_size,
                        shuffle=False,
                        num_workers=NUM_WORKERS
                    )
                    test_loader = DataLoader(
                        TitanicDataset(test_features, ids=passenger_ids),
                        batch_size=cfg.batch_size,
                        shuffle=False,
                        num_workers=NUM_WORKERS
                    )

                    model = TitanicMLP(input_dim=train_features.shape[1], hidden_dims=(256, 128, 64), dropout=0.35).to(device)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=6)
                    logger.info(f'Model: {model.__class__.__name__}')
                    logger.info(f'Optimizer: {optimizer.__class__.__name__}')

                    train_losses, val_losses, train_accs, val_accs = [], [], [], []
                    best_val = 0.0
                    epochs_without_improve = 0

                    for epoch in range(cfg.num_epochs):
                        train_loss, train_acc = train_epoch(
                            model,
                            tqdm(train_loader, desc=f'Train {epoch+1}/{cfg.num_epochs}', leave=False),
                            criterion,
                            optimizer,
                            device
                        )
                        val_loss, epoch_val_acc = evaluate(
                            model,
                            tqdm(val_loader, desc=f'Val {epoch+1}/{cfg.num_epochs}', leave=False),
                            criterion,
                            device
                        )

                        scheduler.step(epoch_val_acc)

                        train_losses.append(train_loss)
                        val_losses.append(val_loss)
                        train_accs.append(train_acc)
                        val_accs.append(epoch_val_acc)

                        logger.info(
                            f'Epoch {epoch+1}/{cfg.num_epochs} | '
                            f'Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}% | '
                            f'Val Loss {val_loss:.4f}, Val Acc {epoch_val_acc:.2f}%'
                        )

                        if epoch_val_acc > best_val:
                            best_val = epoch_val_acc
                            epochs_without_improve = 0
                            torch.save(model.state_dict(), best_model_path)
                            logger.info(f'New best model saved with validation accuracy {epoch_val_acc:.2f}%')
                        else:
                            epochs_without_improve += 1
                            if EARLY_STOPPING and epochs_without_improve >= cfg.patience:
                                logger.info('Early stopping triggered')
                                break

                    plot_training_history(train_losses, train_accs, val_losses, val_accs, plot_name)

                    if os.path.exists(best_model_path):
                        state_dict = torch.load(best_model_path, map_location=device)
                        model.load_state_dict(state_dict)

                    val_loss, final_val_acc, val_labels, val_preds = evaluate(
                        model,
                        tqdm(val_loader, desc='Val (final)', leave=False),
                        criterion,
                        device,
                        return_predictions=True
                    )
                    logger.info(f'Final validation accuracy: {final_val_acc:.2f}% (loss {val_loss:.4f})')
                    report = classification_report(val_labels, val_preds, target_names=CLASS_NAMES, digits=4)
                    logger.info('Validation report:\n' + report)
                    plot_confusion_matrix(val_labels, val_preds, CLASS_NAMES, cm_name)

                    test_predictions = test(model, tqdm(test_loader, desc='Test', leave=False), device)
                    submission_df = pd.DataFrame(test_predictions, columns=['PassengerId', 'Survived'])
                    submission_df = submission_df.sort_values('PassengerId')
                    submission_df.to_csv(submission_path, index=False)
                    logger.info(f'Submission saved to {submission_path}')

                    val_acc = max(best_val, final_val_acc)
                    submission = submission_path
                    model_path = best_model_path
                    _model_obj = model

                submission_accuracy, matches, total = score_submission(
                    Path(submission), ground_truth_path
                )
                logger.info(
                    f'Submission accuracy vs corrected ground truth: '
                    f'{submission_accuracy:.4f} ({matches}/{total})'
                )
            finally:
                logger.remove(file_sink)

            if val_acc > best_result['val_acc']:
                best_result = {
                    'val_acc': val_acc,
                    'seed': seed,
                    'submission': submission,
                    'model_path': model_path
                }

    if best_result['submission']:
        logger.info(
            f'Best seed {best_result["seed"]} achieved validation accuracy {best_result["val_acc"]:.2f}% '
            f'with submission {best_result["submission"]}'
        )
        if KAGGLE_SUBMIT:
            message = f'{CUR_MODEL} best seed {best_result["seed"]} on {today} (val {best_result["val_acc"]:.2f}%)'
            submit_to_kaggle(best_result['submission'], 'titanic', message)


if __name__ == '__main__':
    main()
