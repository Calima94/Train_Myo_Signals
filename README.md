# Train_Myo_Signals

Desktop application to process labeled surface-EMG (sEMG) data from the **Myo armband** and
train/compare machine-learning classifiers for gesture classification. The app loads the raw
recordings, extracts features, trains several scikit-learn models, serializes them with `joblib`,
and logs each classifier's score for comparison.

> Aplicação desktop para processar dados rotulados de sEMG do bracelete **Myo** e treinar/comparar
> classificadores de machine learning. O app carrega as gravações cruas, extrai features, treina
> vários modelos do scikit-learn, salva-os com `joblib` e registra o score de cada classificador
> para comparação.

This is the **training stage** of a two-part pipeline. Data is first recorded with
[**Capture_EMG_Data**](https://github.com/Calima94/Capture_EMG_Data), then trained here.

---

## Suggested repository metadata

Set these on GitHub (the "About" gear, top-right):

- **Description:** `PyQt5 app to process Myo armband sEMG data and train/compare scikit-learn classifiers for gesture recognition. Saves models with joblib.`
- **Topics:** `emg` `semg` `myo-armband` `gesture-recognition` `machine-learning` `scikit-learn` `classification` `pyqt5` `joblib` `python` `biosignals`

---

## Pipeline

```
Capture_EMG_Data  ──►  Raw_EMG_Data/  ──►  feature extraction  ──►  train & compare  ──►  files_joblib/
   (recording)          (this repo)         (mod_sig_emg.py)        classifiers           (saved models)
                                                                          │
                                                                          └──►  scores_of_classifiers.csv
```

## Features

- PyQt5 graphical interface for configuring and running training (`train_emg_data.ui`).
- Loads raw sEMG recordings produced by the capture tool.
- Signal processing and feature extraction (`mod_sig_emg.py`).
- Trains and compares **multiple scikit-learn classifiers** in one run.
- Serializes trained models to disk with `joblib` for later reuse/inference.
- Exports a side-by-side score comparison to `scores_of_classifiers.csv`.

## Example output

Each run automatically trains all classifiers and writes a side-by-side comparison to
[`scores_of_classifiers.csv`](scores_of_classifiers.csv) — the point of the tool is this
one-click benchmark, not any single result. A sample run on a small dataset:

| Classifier | Model | Score |
|------------|-------|-------|
| `lda` | Linear Discriminant Analysis | 0.9444 |
| `gnb` | Gaussian Naive Bayes | 0.9444 |
| `lin_svm` | Linear SVM | 0.9444 |
| `knn` | k-Nearest Neighbors | 0.9444 |
| `tree` | Decision Tree | 0.9444 |

> Scores depend entirely on the dataset you load; the values above are illustrative (a small sample).
> Os valores acima são ilustrativos (amostra pequena) e dependem do conjunto de dados carregado.

## Requirements

- Python 3 (tested on Linux / Ubuntu).
- Dependencies in [`requirements.txt`](requirements.txt).

```bash
pip3 install -r requirements.txt
```

## Usage

```bash
python3 main.py
```

In the interface, point the app at a raw EMG dataset, configure the run, and start training.
Trained models are written to `files_joblib/`, and the comparison of classifier scores is saved to
`scores_of_classifiers.csv`.

## Repository layout

| Path | Purpose |
|------|---------|
| `main.py` | Application entry point |
| `train_emg_data.py`, `train_emg_data.ui` | PyQt5 training interface |
| `train_signals_emg.py` | Training routine for the classifiers |
| `mod_sig_emg.py` | sEMG signal processing / feature extraction |
| `param_to_use.py` | Parameter selection / configuration |
| `Raw_EMG_Data/` | Input recordings (from Capture_EMG_Data) |
| `M_Class_Data/` | Processed multi-class datasets |
| `Parameters/` | Saved run parameters |
| `files_joblib/` | Serialized trained models |
| `scores_of_classifiers.csv` | Classifier score comparison |

## Related projects

- [**Capture_EMG_Data**](https://github.com/Calima94/Capture_EMG_Data) — records the labeled sEMG data this app trains on.

## License

Released under the **Apache-2.0 License**. See [`LICENSE`](LICENSE).

## Author

**Caio Lima** — UFABC
[GitHub](https://github.com/Calima94) · [LinkedIn](https://www.linkedin.com/in/caio-lima-9035a022a/) · [ResearchGate](https://www.researchgate.net/profile/Caio-Lima-15) · [Lattes](http://lattes.cnpq.br/0127370029893676)
