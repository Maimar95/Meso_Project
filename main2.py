# main2.py – Evaluation mit exklusiven, stratifizierten Subsets pro Run

import os
import json
import numpy as np
import random
import logging
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import torch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Konfiguration
NUM_RUNS = 10
RESULT_DIR = "evaluation_results"
CONFUSION_DIR = "confusion_matrices"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CONFUSION_DIR, exist_ok=True)

# Import aus main.py
from main import (
    prepare_data,
    create_hf_dataset,
    label_names,
    tokenizer
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, preds)
    return {"eval_accuracy": acc}


def plot_confusion_matrix_from_predictions(y_true, y_pred, label_names, save_path):
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(matrix, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.xlabel("Vorhergesagte Klasse")
    plt.ylabel("Tatsächliche Klasse")
    plt.xticks(np.arange(len(label_names)), label_names, rotation=45)
    plt.yticks(np.arange(len(label_names)), label_names)

    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, f'{val}', ha='center', va='center', color='black')

    plt.colorbar(cax)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def stratified_deduplicated_splits(data, num_splits=10, seed=42):
    """Teilt die Daten stratifiziert nach Label in num_splits exklusive Subsets."""
    random.seed(seed)
    buckets = defaultdict(list)
    for item in data:
        buckets[item["label"]].append(item)

    splits = [[] for _ in range(num_splits)]

    for label, items in buckets.items():
        random.shuffle(items)
        chunk_size = len(items) // num_splits
        for i in range(num_splits):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_splits - 1 else len(items)
            splits[i].extend(items[start:end])

    return splits


def split_train_val(test_subset, all_data, seed=42):
    """Teilt alle Daten außer Testdaten in Train/Val."""
    test_ids = set(id(x) for x in test_subset)
    rest_data = [x for x in all_data if id(x) not in test_ids]

    labels = [d["label"] for d in rest_data]
    train, val = train_test_split(rest_data, test_size=0.15, random_state=seed, stratify=labels)
    return train, val


def run_single_evaluation(test_subset, all_data, label2id, id2label, run_number=1, epochs=3):
    from main import label_names as imported_label_names
    label_names[:] = imported_label_names

    logging.info(f"🚀 Starte Run {run_number} mit {len(test_subset)} exklusiven Testbeispielen")

    # Exklusive Train+Val
    train_data, val_data = split_train_val(test_subset, all_data, seed=100 + run_number)

    hf_train = create_hf_dataset(train_data)
    hf_val = create_hf_dataset(val_data)
    hf_test = create_hf_dataset(test_subset)

    # Modell
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-german-cased',
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training_args = TrainingArguments(
        output_dir=f"stadt_klassifikator_run{run_number}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_train,
        eval_dataset=hf_val,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Evaluation
    predictions = trainer.predict(hf_test)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    # Fehlklassifikationen
    misclassified = []
    for idx, (pred, true) in enumerate(zip(y_pred, y_true)):
        if pred != true:
            misclassified.append({
                "text": tokenizer.decode(hf_test[idx]['input_ids'], skip_special_tokens=True),
                "true_city": id2label[true],
                "predicted_city": id2label[pred]
            })

    # Confusion Matrix
    plot_confusion_matrix_from_predictions(
        y_true, y_pred, label_names,
        save_path=os.path.join(CONFUSION_DIR, f"confusion_matrix_run_{run_number}.png")
    )

    result = {
        "run": run_number,
        "accuracy": accuracy_score(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, target_names=label_names, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "misclassified_samples": misclassified
    }

    with open(os.path.join(RESULT_DIR, f"run_{run_number}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logging.info(f"✅ Run {run_number} abgeschlossen mit Accuracy: {result['accuracy']:.4f}")
    return result


if __name__ == '__main__':
    from main import split_dataset
    from sklearn.model_selection import train_test_split

    all_data, label2id, id2label = prepare_data("Trinningsdata.JSON")
    run_splits = stratified_deduplicated_splits(all_data, num_splits=NUM_RUNS, seed=42)

    all_results = []
    for run_num in range(1, NUM_RUNS + 1):
        test_subset = run_splits[run_num - 1]
        result = run_single_evaluation(test_subset, all_data, label2id, id2label, run_number=run_num)
        all_results.append(result)

    # Zusammenfassung
    accuracies = [r["accuracy"] for r in all_results]
    mean_acc = round(np.mean(accuracies), 4)
    std_acc = round(np.std(accuracies), 4)

    mean_matrix = np.mean([np.array(r["confusion_matrix"]) for r in all_results], axis=0).tolist()
    std_matrix = np.std([np.array(r["confusion_matrix"]) for r in all_results], axis=0).tolist()

    all_misclassified = [m for r in all_results for m in r["misclassified_samples"]]

    summary = {
        "total_runs": NUM_RUNS,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "all_accuracies": accuracies,
        "mean_confusion_matrix": mean_matrix,
        "std_confusion_matrix": std_matrix,
        "total_misclassifications": len(all_misclassified),
        "misclassified_samples": all_misclassified
    }

    with open(os.path.join(RESULT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info(f"\n📊 Mittelwert Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    logging.info(f"📁 Ergebnisse gespeichert unter: {RESULT_DIR}")
    logging.info(f"🖼️ Konfusionsmatrizen gespeichert unter: {CONFUSION_DIR}")
