"""
KI-basierte Pipeline zur automatischen Stadtzuordnung
Im Rahmen der Masterarbeit:
„Einsatz von KI zur automatisierten Analyse medizinischer Texte 
und zur Klassifikation passender OPS-Codes“

Entwickelt in Zusammenarbeit mit Meso International GmbH
"""

import json
import random
import logging
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Logging einrichten
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialisierung
tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
label_names = []

def lade_json(pfad):
    """Lädt den Datensatz aus einer JSON-Datei."""
    try:
        with open(pfad, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"⚠️ Datei nicht gefunden: {pfad}")
        raise
    except json.JSONDecodeError:
        logging.error(f"⚠️ Ungültiges JSON in Datei: {pfad}")
        raise

def speichere_json(pfad, daten):
    """Speichert Daten im JSON-Format mit UTF-8-Kodierung."""
    with open(pfad, 'w', encoding='utf-8') as f:
        json.dump(daten, f, ensure_ascii=False, indent=2)
    logging.info(f"Daten gespeichert unter {pfad}")

def tokenize_and_chunk(text, max_length=512, stride=50):
    """
    Tokenisiert einen Text und teilt ihn in Chunks auf,
    um Transformer-basierte Modelle mit begrenzter Länge zu bedienen.
    """
    encoded = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_tensors='pt'
    )
    return list(zip(encoded['input_ids'].tolist(), encoded['attention_mask'].tolist()))

def split_dataset(data, seed=42):
    """Teilt den Datensatz stratifiziert in Train/Val/Test auf (70–15–15)."""
    labels = [d['label'] for d in data]
    train_data, temp_data = train_test_split(data, test_size=0.30, random_state=seed, shuffle=True, stratify=labels)
    val_data, test_data = train_test_split(temp_data, test_size=0.50, random_state=seed, shuffle=True, stratify=[d['label'] for d in temp_data])
    return train_data, val_data, test_data

def create_hf_dataset(data_list):
    """Konvertiert in Hugging Face Dataset-Format."""
    return Dataset.from_dict({
        'input_ids': [d['input_ids'] for d in data_list],
        'attention_mask': [d['attention_mask'] for d in data_list],
        'label': [d['label'] for d in data_list]
    })

def prepare_model(labels):
    """Bereitet ein vortrainiertes BERT-Modell für die Stadt-Klassifikation vor."""
    num_labels = len(labels)
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-german-cased',
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    return model, label2id, id2label

def compute_metrics(eval_pred):
    """Berechnet Accuracy und Classification Report."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    class_report = classification_report(labels, predictions, target_names=label_names, zero_division=0)
    return {
        "accuracy": acc,
        "classification_report": class_report
    }

def classify_city(model, tokenizer, id2label, text):
    """Macht eine Vorhersage für einen neuen Text mit Chunking und Aggregation."""
    chunks = tokenize_and_chunk(text, max_length=512, stride=50)
    chunk_probs = []

    for input_ids, attention_mask in chunks:
        inputs = {
            'input_ids': torch.tensor([input_ids]),
            'attention_mask': torch.tensor([attention_mask])
        }
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).detach().numpy()[0]
        chunk_probs.append(probs)

    # Durchschnittliche Konfidenz berechnen
    final_probabilities = np.mean(chunk_probs, axis=0)
    pred_idx = np.argmax(final_probabilities)
    predicted_city = id2label[pred_idx]
    confidence = float(final_probabilities[pred_idx])

    return {
        "predicted_city": predicted_city,
        "confidence": confidence,
        "probabilities": final_probabilities.tolist()
    }

def manual_feedback(predicted_city):
    """Fragt Benutzer, ob die Vorhersage korrekt ist."""
    print(f"\nKI sagt: '{predicted_city}'")
    korrekt = input("Ist das richtig? (j/n): ").strip().lower()
    if korrekt == 'j':
        return True, predicted_city
    else:
        neue_stadt = input("Welche Stadt ist es wirklich? (Berlin/Chemnitz/Hamburg): ")
        return False, neue_stadt

def plot_confusion_matrix(true_labels, preds, label_names):
    """Erstellt und speichert eine Confusion Matrix."""
    matrix = confusion_matrix(true_labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(matrix, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.xlabel("Vorhergesagte Klasse")
    plt.ylabel("Tatsächliche Klasse")
    plt.xticks(np.arange(len(label_names)), label_names, rotation=45)
    plt.yticks(np.arange(len(label_names)), label_names)

    # Werte in die Matrix schreiben
    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, f'{val}', ha='center', va='center', color='black')

    plt.colorbar(cax)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

# Teil 1: Daten laden und verarbeiten
def prepare_data(json_pfad, use_clean=True):
    logging.info("🔄 Lade Daten...")
    daten = lade_json(json_pfad)

    logging.info("🏷️ Erstelle Label-Mappings...")
    global label_names
    labels = sorted({eintrag.get('City') or eintrag.get('city') or eintrag.get('stadt', 'Unbekannt') for eintrag in daten})
    label_names = labels
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    speichere_json('label2id.json', label2id)
    speichere_json('id2label.json', id2label)

    logging.info("🧱 Verarbeite Daten (Tokenisierung & Chunking)...")
    verarbeitete_daten = []
    for eintrag in daten:
        text = eintrag['clean_text'] if use_clean else eintrag.get('raw_text', '')
        if not text:
            logging.warning(f"⚠️ Leerer Text in Eintrag: {eintrag.get('quelle', 'Unbekannt')}")
            continue

        city = eintrag.get('City') or eintrag.get('city') or eintrag.get('stadt', 'Unbekannt')
        if city not in label2id:
            logging.warning(f"⚠️ Unbekannte Stadt '{city}' in Eintrag – übersprungen.")
            continue

        chunks = tokenize_and_chunk(text)
        for input_ids, attention_mask in chunks:
            verarbeitete_daten.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': label2id[city]
            })
    return verarbeitete_daten, label2id, id2label

# Teil 2: Modelltraining und Evaluation
def train_and_evaluate(verarbeitete_daten, label2id, id2label, epochs=3, output_dir="stadt_klassifikator"):
    logging.info("✂️ Teile Datensatz in Train/Val/Test...")
    train_data, val_data, test_data = split_dataset(verarbeitete_daten)
    hf_train = create_hf_dataset(train_data)
    hf_val = create_hf_dataset(val_data)
    hf_test = create_hf_dataset(test_data)

    logging.info("🧠 Bereite Modell vor...")
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-german-cased',
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_train,
        eval_dataset=hf_val,
        compute_metrics=compute_metrics
    )

    logging.info("🏋️ Beginne mit dem Training...")
    trainer.train()

    logging.info("🔍 Evaluationsbericht:")
    results = trainer.evaluate(hf_test)
    logging.info(f"Evaluations-Ergebnisse: {results}")

    predictions = trainer.predict(hf_test)
    preds = np.argmax(predictions.predictions, axis=-1)

    print(classification_report(predictions.label_ids, preds, target_names=label_names))

    logging.info("🖼️ Speichere Confusion Matrix...")
    plot_confusion_matrix(predictions.label_ids, preds, label_names)

    logging.info("💾 Speichere trainiertes Modell...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, preds, predictions.label_ids

# Teil 3: Interaktive Inferenz mit manuellem Feedback
def interactive_inference(model, tokenizer, id2label):
    logging.info("\n🧪 Manuelles Testen gestartet – gib Texte ein oder beende mit 'exit'")
    history = []

    while True:
        user_input = input("\n📝 Gib einen Text ein:\n")
        if user_input.lower() in ['exit', 'quit']:
            break
        result = classify_city(model, tokenizer, id2label, user_input)
        is_correct, actual_city = manual_feedback(result['predicted_city'])
        history.append({
            "input_text": user_input,
            "predicted_city": result["predicted_city"],
            "confidence": result["confidence"],
            "is_correct": is_correct,
            "actual_city": actual_city
        })
        print("-" * 50)

    correct_count = sum(1 for entry in history if entry['is_correct'])
    total_count = len(history)
    real_accuracy = correct_count / total_count if total_count > 0 else 0
    logging.info(f"📈 Genauigkeit mit manuellem Feedback: {real_accuracy:.2%} ({correct_count}/{total_count})")

    logging.info("💾 Speichere alle Ergebnisse...")
    speichere_json("classification_history.json", history)

    return history

# Haupt-Pipeline
def pipeline(json_pfad, use_clean=True, epochs=3, output_dir="stadt_klassifikator"):
    # Schritt 1: Daten laden und vorbereiten
    verarbeitete_daten, label2id, id2label = prepare_data(json_pfad, use_clean=use_clean)

    # Schritt 2: Modell trainieren und evaluieren
    model, _, _ = train_and_evaluate(verarbeitete_daten, label2id, id2label, epochs, output_dir)

    # Schritt 3: Interaktives Feedback aktivieren
    interactive_inference(model, tokenizer, id2label)

    logging.info("🎉 Fertig! Modell erfolgreich trainiert und gespeichert.")

if __name__ == '__main__':
    pipeline('Trinningsdata.JSON')