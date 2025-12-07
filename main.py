from __future__ import annotations

import os # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import subprocess # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import sys # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import warnings # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from datetime import datetime

import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import numpy as np # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import pandas as pd # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import seaborn as sns # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import torch # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import torch.nn as nn # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import torch.optim as optim # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from loguru import logger # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from sklearn.model_selection import train_test_split # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from torch.utils.data import DataLoader # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from tqdm.auto import tqdm # pyright: ignore[reportMissingModuleSource, reportMissingImports]

from network import TitanicMLP, evaluate, test, train_epoch
from utils import (
    CLASS_NAMES,
    CUR_MODEL,
    EARLY_STOPPING,
    KAGGLE_SUBMIT,
    NUM_WORKERS,
    TrainingConfig,
    TitanicDataset,
    prepare_titanic_data,
    set_global_seed,
)


warnings.filterwarnings('ignore', category=FutureWarning)


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

    rf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced_subsample',
        random_state=cfg.seed,
        n_jobs=-1
    )
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

    model_path = os.path.join(output_dir, f'{log_name}_random_forest_{today}.joblib')
    try:
        import joblib # pyright: ignore[reportMissingModuleSource, reportMissingImports]
        joblib.dump(rf, model_path)
        logger.info(f'RandomForest model serialized to {model_path}')
    except ImportError:
        logger.warning('joblib not installed; skipping model serialization')

    return val_acc


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

    gb = GradientBoostingClassifier(
        n_estimators=600,
        learning_rate=0.02,
        max_depth=3,
        min_samples_split=4,
        min_samples_leaf=2,
        subsample=0.9,
        random_state=cfg.seed
    )
    logger.info('Training GradientBoostingClassifier...')
    gb.fit(X_train, y_train)

    val_preds = gb.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds) * 100.0
    report = classification_report(y_val, val_preds, target_names=CLASS_NAMES, digits=4)
    logger.info(f'GradientBoosting validation accuracy: {val_acc:.2f}%')
    logger.info('Validation report:\n' + report)
    plot_confusion_matrix(y_val, val_preds, CLASS_NAMES, cm_name)

    test_preds = gb.predict(test_features)
    submission_df = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': test_preds})
    submission_df = submission_df.sort_values('PassengerId')
    submission_df.to_csv(submission_path, index=False)
    logger.info(f'GradientBoosting submission saved to {submission_path}')

    model_path = os.path.join(output_dir, f'{log_name}_gradient_boosting_{today}.joblib')
    try:
        import joblib # pyright: ignore[reportMissingModuleSource, reportMissingImports]
        joblib.dump(gb, model_path)
        logger.info(f'GradientBoosting model serialized to {model_path}')
    except ImportError:
        logger.warning('joblib not installed; skipping model serialization')

    return val_acc


def evaluate_submission(submission_path, gender_csv, raw_test_csv, output_dir):
    if not os.path.exists(submission_path):
        logger.warning(f'Submission file not found at {submission_path}; skipping evaluation')
        return
    if not os.path.exists(gender_csv):
        logger.warning('Ground-truth file gender_submission.csv not found; skipping evaluation')
        return
    if not os.path.exists(raw_test_csv):
        logger.warning('Raw test CSV not found; skipping FN/TN exports')
        return

    submission_df = pd.read_csv(submission_path)
    pred_df = submission_df.rename(columns={'Survived': 'Pred'})
    actual_df = pd.read_csv(gender_csv).rename(columns={'Survived': 'Actual'})
    merged = pred_df.merge(actual_df, on='PassengerId', how='left')
    missing_actual = merged['Actual'].isna().sum()
    if missing_actual:
        logger.warning(f'Missing ground-truth labels for {missing_actual} passengers; excluding them from metrics')
        merged = merged.dropna(subset=['Actual'])

    merged['Actual'] = merged['Actual'].astype(int)
    merged['Pred'] = merged['Pred'].astype(int)

    accuracy = accuracy_score(merged['Actual'], merged['Pred']) * 100.0
    f1 = f1_score(merged['Actual'], merged['Pred'])
    cm = confusion_matrix(merged['Actual'], merged['Pred'], labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
        logger.warning('Confusion matrix shape unexpected; defaulting counts to zero')

    logger.info(
        f"Submission metrics — Accuracy: {accuracy:.2f}%, F1: {f1:.4f}, TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}"
    )

    test_df = pd.read_csv(raw_test_csv)
    detail_cols = ['PassengerId', 'Actual', 'Pred'] + [c for c in test_df.columns if c != 'PassengerId']
    detailed = merged.merge(test_df, on='PassengerId', how='left')
    available_cols = [c for c in detail_cols if c in detailed.columns]
    detailed = detailed[available_cols]

    base_name = os.path.splitext(os.path.basename(submission_path))[0]
    fn_path = os.path.join(output_dir, f'{base_name}_fn.csv')
    tn_path = os.path.join(output_dir, f'{base_name}_tn.csv')
    metrics_path = os.path.join(output_dir, f'{base_name}_metrics.txt')

    fn_detail = detailed[(detailed['Actual'] == 1) & (detailed['Pred'] == 0)]
    tn_detail = detailed[(detailed['Actual'] == 0) & (detailed['Pred'] == 0)]
    fn_detail.to_csv(fn_path, index=False)
    tn_detail.to_csv(tn_path, index=False)

    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f'Accuracy: {accuracy:.4f}\n')
        f.write(f'F1 Score: {f1:.4f}\n')
        f.write(f'TN: {tn}\nFP: {fp}\nFN: {fn}\nTP: {tp}\n')

    logger.info(f'False negatives exported to {fn_path}')
    logger.info(f'True negatives exported to {tn_path}')
    logger.info(f'Metric summary written to {metrics_path}')


def main():
    cfg = TrainingConfig()
    set_global_seed(cfg.seed)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    output_dir = os.path.join(current_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    log_name = f'{CUR_MODEL}'
    if EARLY_STOPPING:
        log_name += '_ES'
    today = datetime.now().strftime('%Y-%m-%d')
    plot_name = os.path.join(output_dir, f'{log_name}_training_{today}.png')
    cm_name = os.path.join(output_dir, f'{log_name}_cm_{today}.png')
    best_model_path = os.path.join(output_dir, f'best_{log_name}_{today}.pth')
    submission_path = os.path.join(output_dir, f'{log_name}_submission.csv')

    logger.remove()
    logger.add(sys.stderr, format='<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>')
    logger.add(
        os.path.join(output_dir, f'{log_name}_{today}.log'),
        format='{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}',
        colorize=False
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    train_csv = os.path.join(data_dir, 'train.csv')
    test_csv = os.path.join(data_dir, 'test.csv')
    gender_csv = os.path.join(data_dir, 'gender_submission.csv')
    train_features, train_labels, test_features, passenger_ids, feature_names = prepare_titanic_data(train_csv, test_csv)
    logger.info(f'Loaded {train_features.shape[0]} training samples with {train_features.shape[1]} features')

    model_name = CUR_MODEL.lower()
    if model_name == 'randomforest':
        val_acc = train_random_forest(
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
        )

        evaluate_submission(submission_path, gender_csv, test_csv, output_dir)

        if KAGGLE_SUBMIT:
            message = f'RandomForest on {today} (val {val_acc:.2f}%)'
            submit_to_kaggle(submission_path, 'titanic', message)
        return

    if model_name == 'gradientboosting':
        val_acc = train_gradient_boosting(
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
        )

        evaluate_submission(submission_path, gender_csv, test_csv, output_dir)

        if KAGGLE_SUBMIT:
            message = f'GradientBoosting on {today} (val {val_acc:.2f}%)'
            submit_to_kaggle(submission_path, 'titanic', message)
        return

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
        val_loss, val_acc = evaluate(
            model,
            tqdm(val_loader, desc=f'Val {epoch+1}/{cfg.num_epochs}', leave=False),
            criterion,
            device
        )

        scheduler.step(val_acc)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        logger.info(
            f'Epoch {epoch+1}/{cfg.num_epochs} | '
            f'Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}% | '
            f'Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%'
        )

        if val_acc > best_val:
            best_val = val_acc
            epochs_without_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f'New best model saved with validation accuracy {val_acc:.2f}%')
        else:
            epochs_without_improve += 1
            if EARLY_STOPPING and epochs_without_improve >= cfg.patience:
                logger.info('Early stopping triggered')
                break

    plot_training_history(train_losses, train_accs, val_losses, val_accs, plot_name)

    if os.path.exists(best_model_path):
        state_dict = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state_dict)

    val_loss, val_acc, val_labels, val_preds = evaluate(
        model,
        tqdm(val_loader, desc='Val (final)', leave=False),
        criterion,
        device,
        return_predictions=True
    )
    logger.info(f'Final validation accuracy: {val_acc:.2f}% (loss {val_loss:.4f})')
    report = classification_report(val_labels, val_preds, target_names=CLASS_NAMES, digits=4)
    logger.info('Validation report:\n' + report)
    plot_confusion_matrix(val_labels, val_preds, CLASS_NAMES, cm_name)

    test_predictions = test(model, tqdm(test_loader, desc='Test', leave=False), device)
    submission_df = pd.DataFrame(test_predictions, columns=['PassengerId', 'Survived'])
    submission_df = submission_df.sort_values('PassengerId')
    submission_df.to_csv(submission_path, index=False)
    logger.info(f'Submission saved to {submission_path}')

    evaluate_submission(submission_path, gender_csv, test_csv, output_dir)

    if KAGGLE_SUBMIT:
        message = f'{CUR_MODEL} on {today} (val {best_val:.2f}%)'
        submit_to_kaggle(submission_path, 'titanic', message)


if __name__ == '__main__':
    main()
