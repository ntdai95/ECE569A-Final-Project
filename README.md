# Support Ticket Triage

This project was completed by a team of four for Selected Topics in Computer Engineering: AI (ECE 569A) at the University of Victoria. It compares customer support ticket triage models on the same three prediction tasks. Retrieval kNN at k=5 achieved the strongest accuracy across ticket type, priority, and destination queue. For queue routing, it reached **0.702 accuracy versus 0.597 for DistilBERT**.

## Headline Results

The held-out test set contained 3,268 English-language tickets. The table reports accuracy from the saved notebook outputs.

| Model | Ticket type | Priority | Queue |
|---|---:|---:|---:|
| **Retrieval kNN (k=5)** | **0.892** | **0.759** | **0.702** |
| DistilBERT + neural network | 0.880 | 0.692 | 0.597 |

The Streamlit demo exposes the three predictions and shows the historical tickets retrieved by the kNN models, so users can inspect the examples behind a result.

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
- `Retrieval + kNN (k=5)`
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
- The demo is built for presentation and comparison. It is not a production deployment.
