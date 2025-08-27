# 🏙️ Stadt-Klassifikator – Einfache Anleitung

Dieses Projekt analysiert deutsche Texte und klassifiziert sie einer von drei Städten zu: **Berlin**, **Hamburg** oder **Chemnitz**.
Das Modell basiert auf **BERT** und ist bereits vortrainiert. Du kannst es direkt nutzen, um Texte zu klassifizieren – inklusive einer interaktiven Oberfläche und automatischer Evaluation.

---

## 🚀 So funktioniert es – Schritt für Schritt

### 1. 🔽 Git installieren (falls noch nicht vorhanden)

Lade und installiere Git:
👉 [https://git-scm.com/downloads](https://git-scm.com/downloads)

### 2. 📂 Projekt herunterladen

Öffne die **Eingabeaufforderung (CMD)** oder **PowerShell** und führe aus:

```bash
git clone https://github.com/Maimar95/Letzte-Anpassung.git
cd Letzte-Anpassung
```

### 3. 🐍 Python installieren

Du benötigst **Python 3.9 oder höher**.
📥 Download: [https://www.python.org/downloads/](https://www.python.org/downloads/)
Prüfe nach der Installation:

```bash
python --version
```

→ Sollte `Python 3.9.x` oder höher anzeigen.

### 4. ⚙️ Abhängigkeiten installieren

Alle benötigten Bibliotheken sind in der `requirements.txt`-Datei enthalten. Installiere sie mit:

```bash
pip install -r requirements.txt
```

---

## ▶️ Ausführen des Projekts

### 🔹 mit Interaktive Klassifizierung (Single Run)

Starte das Hauptprogramm:

```bash
python main.py
```

* Evaluation der Testdaten wird direkt im Terminal angezeigt.
* Eine **Confusion Matrix** wird visualisiert und im selben Ordner wie main.py als `confusion_matrix.png` gespeichert.
* Anschließend startet eine **interaktive Eingabeschnittstelle**, in der du beliebige deutsche Texte eingeben kannst. Das Modell klassifiziert sie direkt und fragt nach Feedback, ob die Vorhersage korrekt ist.

➡️ **Tipp:** Um die interaktive Sitzung zu beenden, gib `exit` oder `quit` ein und drücke Enter.

Alle eingegebenen Texte, Vorhersagen (inkl. Konfidenz) und Feedback werden automatisch in der Datei **`classification_History.json`** gespeichert – zum Nachschauen oder Analysieren.

---

### 🔹 Wiederholte Ausführung (10-fache Evaluation)

Um die Stabilität und Genauigkeit des Modells statistisch zu testen, kannst du eine 10-malige Durchführung starten:

```bash
python main2.py
```

* Speichert alle **Confusion Matrices** im Ordner `Confusion_Matrices/` – jede Matrix in einer separaten PNG-Datei mit fortlaufender Nummer.
* Evaluationsergebnisse und Metriken werden im Ordner `Evaluation_Results/` in separaten JSON-Dateien gespeichert, jeweils nummeriert pro Durchlauf.

---

## 🛠️ Wichtige Dateien & Ordner

| Datei / Ordner                | Funktion                                                                      |
| ----------------------------- | ----------------------------------------------------------------------------- |
| `main.py`                     | Lädt das vortrainierte Modell und startet die interaktive Klassifizierung     |
| `main2.py`                    | Führt 10 Durchläufe von `main.py` durch und sammelt die Evaluationsergebnisse |
| `requirements.txt`            | Liste aller benötigten Python-Pakete                                          |
| `stadt_klassifikator/`        | Enthält das vortrainierte BERT-Modell und den Tokenizer                       |
| `classification_History.json` | Speichert alle interaktiven Eingaben, Vorhersagen und Feedback                |
| `confusion_matrix.png`        | Confusion Matrix des Einzeldurchlaufs                                         |
| `Confusion_Matrices/`         | Alle Confusion Matrices der 10-fachen Evaluation                              |
| `Evaluation_Results/`         | Alle Metriken der 10-fachen Evaluation in JSON-Dateien                        |



---

## ❗ Häufige Probleme & Lösungen

Wenn du Fehler wie diese siehst:

* `ModuleNotFoundError`
* `No module named 'transformers'`
* `No model found in stadt_klassifikator/`

Dann führe folgende Schritte aus:

1. Stelle sicher, dass Python und pip aktuell sind.
2. Installiere die Abhängigkeiten neu:

   ```bash
   pip install -r requirements.txt
   ```
3. Überprüfe, ob der Ordner `stadt_klassifikator/` vollständig heruntergeladen wurde.

---
