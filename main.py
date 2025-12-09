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
from torch.utils.data import DataLoader # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from tqdm.auto import tqdm # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import csv
from typing import Iterable, List, Tuple
from network import TitanicMLP, evaluate, test, train_epoch
from utils import (
    CLASS_NAMES,
    CUR_MODEL,
    EARLY_STOPPING,
    IMMFEATURE,
    KAGGLE_SUBMIT,
    NUM_WORKERS,
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

    return val_acc, submission_path, model_path


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

    return val_acc, submission_path, model_path


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

    return val_acc, submission_path, model_path


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

    svm = SVC(
        C=2.0,
        kernel='rbf',
        gamma='scale',
        class_weight='balanced',
        probability=False
    )
    logger.info('Training SVM (RBF kernel)...')
    svm.fit(X_train, y_train)

    val_preds = svm.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds) * 100.0
    report = classification_report(y_val, val_preds, target_names=CLASS_NAMES, digits=4)
    logger.info(f'SVM validation accuracy: {val_acc:.2f}%')
    logger.info('Validation report:\n' + report)
    plot_confusion_matrix(y_val, val_preds, CLASS_NAMES, cm_name)

    test_preds = svm.predict(test_features)
    submission_df = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': test_preds})
    submission_df = submission_df.sort_values('PassengerId')
    submission_df.to_csv(submission_path, index=False)
    logger.info(f'SVM submission saved to {submission_path}')

    model_path = os.path.join(output_dir, f'{log_name}_svm_{today}.pth')
    torch.save(svm, model_path)
    logger.info(f'SVM model serialized to {model_path}')

    return val_acc, submission_path, model_path


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

    knn = KNeighborsClassifier(
        n_neighbors=15,
        weights='distance',
        metric='minkowski',
        p=2,
        n_jobs=-1
    )
    logger.info('Training KNeighborsClassifier...')
    knn.fit(X_train, y_train)

    val_preds = knn.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds) * 100.0
    report = classification_report(y_val, val_preds, target_names=CLASS_NAMES, digits=4)
    logger.info(f'KNN validation accuracy: {val_acc:.2f}%')
    logger.info('Validation report:\n' + report)
    plot_confusion_matrix(y_val, val_preds, CLASS_NAMES, cm_name)

    test_preds = knn.predict(test_features)
    submission_df = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': test_preds})
    submission_df = submission_df.sort_values('PassengerId')
    submission_df.to_csv(submission_path, index=False)
    logger.info(f'KNN submission saved to {submission_path}')

    model_path = os.path.join(output_dir, f'{log_name}_knn_{today}.pth')
    torch.save(knn, model_path)
    logger.info(f'KNN model serialized to {model_path}')

    return val_acc, submission_path, model_path


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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    train_csv = os.path.join(data_dir, 'train.csv')
    test_csv = os.path.join(data_dir, 'test.csv')
    train_features, train_labels, test_features, passenger_ids, feature_names = prepare_titanic_data(
        train_csv,
        test_csv,
        immfeature=IMMFEATURE
    )
    logger.info(f'Loaded {train_features.shape[0]} training samples with {train_features.shape[1]} features')

    model_name = CUR_MODEL.lower()
    best_result = {
        'val_acc': -1.0,
        'seed': None,
        'submission': None,
        'model_path': None
    }

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
                val_acc, submission, model_path = train_random_forest(
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
                val_acc, submission, model_path = train_gradient_boosting(
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
                val_acc, submission, model_path = train_logistic_regression(
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
                val_acc, submission, model_path = train_svm(
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
                val_acc, submission, model_path = train_knn(
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
            else:
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
