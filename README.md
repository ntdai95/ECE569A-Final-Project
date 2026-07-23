# ECE-569A

This repository contains our ECE 569A final project on customer support ticket triage.  
We study three prediction tasks from ticket text:

- ticket `type`
- ticket `priority`
- destination `queue`

The repo includes several model families:

- `TF_IDF_Kernel_SVM.ipynb`: TF-IDF + Linear / RBF SVM
- `retrieval_knn_ticketmaster_colab.ipynb`: retrieval-based kNN with cosine similarity
- `BERT_NN.ipynb`: DistilBERT embeddings + neural network
- `ticketmaster.ipynb`: dataset preparation and baseline experiments
- `demo_ticket_triage_app.py`: Streamlit demo app

## Demo

The Streamlit demo lets you:

- enter a ticket `subject` and `body`
- choose a model
- predict `type`, `priority`, and `queue`
- inspect retrieved historical tickets for the kNN models

### Included Demo Models

- `TF-IDF + Linear SVM`
- `Retrieval + kNN (k=1)`
- `Retrieval + kNN (k=3)`
- `DistilBERT + NN`

The demo also includes cached artifacts in `artifacts/` so that:

- the test-set examples load quickly
- the DistilBERT + NN model can reuse saved classifiers instead of rebuilding everything each time

## How To Run The Demo

### 1. Go to the repo folder

```bash
cd ECE569A-Final-Project
```

### 2. Install dependencies

At minimum, install:

```bash
pip install streamlit scikit-learn pandas numpy joblib torch transformers
```

### 3. Start the app

```bash
streamlit run demo_ticket_triage_app.py
```

### 4. Open the local URL

Streamlit will print a local URL such as:

```text
http://localhost:8501
```

## Data

The demo expects the processed English dataset to be available at one of these paths:

- `data/ticketmaster_processed_english.csv`
- `ticketmaster_processed_english.csv`

## Notes

- The kNN models use retrieved historical tickets for prediction and explanation.
- The DistilBERT + NN model uses DistilBERT embeddings together with the neural network settings selected in our project experiments.
- The demo is intended for presentation and comparison purposes rather than production deployment.
