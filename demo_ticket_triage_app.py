from pathlib import Path

import joblib
import numpy as np
import torch
import pandas as pd
import streamlit as st
from genetic_algorithm import build_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from transformers import AutoModel, AutoTokenizer


st.set_page_config(
    page_title="Ticket Buddy Demo",
    page_icon="🎀",
    layout="wide",
)

PRIMARY = "#ff7aa2"
SECONDARY = "#7cc7ff"
CREAM = "#fff8f3"
MINT = "#effcf6"
LAVENDER = "#f6f0ff"
INK = "#24324a"
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)
TESTSET_EXAMPLES_PATH = ARTIFACT_DIR / "demo_testset_examples_v2.joblib"
BERT_CLASSIFIERS_PATH = ARTIFACT_DIR / "bert_demo_classifiers.joblib"

st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(180deg, {CREAM} 0%, #ffffff 100%);
        color: {INK};
    }}
    .hero-card {{
        background: linear-gradient(135deg, #fff1f6 0%, #eef8ff 100%);
        border: 1px solid #ffd4e2;
        border-radius: 24px;
        padding: 1.2rem 1.3rem;
        box-shadow: 0 10px 24px rgba(120, 120, 160, 0.12);
        margin-bottom: 1rem;
    }}
    .hero-title {{
        font-size: 2.1rem;
        font-weight: 800;
        color: {INK};
        margin-bottom: 0.25rem;
    }}
    .hero-subtitle {{
        font-size: 1rem;
        color: #50627d;
        line-height: 1.5;
    }}
    .pill-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.85rem;
    }}
    .pill {{
        background: white;
        border: 1px solid #e8d8ff;
        border-radius: 999px;
        padding: 0.4rem 0.8rem;
        font-size: 0.88rem;
        color: {INK};
    }}
    .section-card {{
        background: white;
        border-radius: 22px;
        padding: 1rem 1rem 0.8rem 1rem;
        border: 1px solid #ececf6;
        box-shadow: 0 8px 18px rgba(50, 70, 120, 0.08);
        margin-bottom: 1rem;
    }}
    .prediction-card {{
        background: linear-gradient(180deg, #ffffff 0%, #fbfcff 100%);
        border: 1px solid #edf1fb;
        border-radius: 20px;
        padding: 1rem;
        min-height: 140px;
        box-shadow: 0 8px 14px rgba(50, 70, 120, 0.06);
    }}
    .prediction-label {{
        font-size: 0.9rem;
        color: #60708a;
        margin-bottom: 0.4rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    .prediction-value {{
        font-size: 1.45rem;
        font-weight: 800;
        color: {INK};
        margin-bottom: 0.35rem;
    }}
    .prediction-note {{
        font-size: 0.92rem;
        color: #596b83;
        line-height: 1.45;
    }}
    .tiny-note {{
        font-size: 0.9rem;
        color: #62738a;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data():
    candidates = [
        Path("data/ticketmaster_processed_english.csv"),
        Path("ticketmaster_processed_english.csv"),
    ]
    data_path = next((p for p in candidates if p.exists()), None)
    if data_path is None:
        raise FileNotFoundError(
            "Could not find ticketmaster_processed_english.csv. Put it in the repo root or data/ folder."
        )

    df = pd.read_csv(data_path)
    df["combined_text"] = df["combined_text"].fillna("").astype(str).str.strip()
    df = df[df["combined_text"] != ""].reset_index(drop=True)
    return df, data_path


@st.cache_data(show_spinner=False)
def get_demo_split():
    df, _ = load_data()
    idx_train, idx_test = train_test_split(
        df.index,
        test_size=0.2,
        random_state=42,
        stratify=df["queue"],
    )
    train_df = df.loc[idx_train].reset_index(drop=True)
    test_df = df.loc[idx_test].reset_index(drop=True)
    return train_df, test_df


@st.cache_resource(show_spinner=True)
def train_demo_assets():
    df, _ = load_data()
    train_df, test_df = get_demo_split()

    tfidf_params = {
        "max_features": 15000,
        "ngram_range": (1, 2),
        "min_df": 2,
    }

    linear_svm_type = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf", LinearSVC(dual="auto", max_iter=10000, random_state=42)),
    ])
    linear_svm_priority = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf", LinearSVC(dual="auto", max_iter=10000, random_state=42)),
    ])
    linear_svm_queue = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf", LinearSVC(dual="auto", max_iter=10000, random_state=42)),
    ])

    knn1_type = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf", KNeighborsClassifier(n_neighbors=1, metric="cosine", weights="distance", algorithm="brute")),
    ])
    knn1_priority = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf", KNeighborsClassifier(n_neighbors=1, metric="cosine", weights="distance", algorithm="brute")),
    ])
    knn1_queue = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf", KNeighborsClassifier(n_neighbors=1, metric="cosine", weights="distance", algorithm="brute")),
    ])

    knn3_type = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf", KNeighborsClassifier(n_neighbors=3, metric="cosine", weights="distance", algorithm="brute")),
    ])
    knn3_priority = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf", KNeighborsClassifier(n_neighbors=3, metric="cosine", weights="distance", algorithm="brute")),
    ])
    knn3_queue = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf", KNeighborsClassifier(n_neighbors=3, metric="cosine", weights="distance", algorithm="brute")),
    ])

    models = {
        "TF-IDF + Linear SVM": {
            "type": linear_svm_type,
            "priority": linear_svm_priority,
            "queue": linear_svm_queue,
            "blurb": "Our main classical classifier from the repo.",
        },
        "Retrieval + kNN (k=3)": {
            "type": knn3_type,
            "priority": knn3_priority,
            "queue": knn3_queue,
            "blurb": "A small-neighborhood retrieval model that uses three similar tickets.",
        },
    }

    for model_group in models.values():
        model_group["type"].fit(train_df["combined_text"], train_df["type"])
        model_group["priority"].fit(train_df["combined_text"], train_df["priority"])
        model_group["queue"].fit(train_df["combined_text"], train_df["queue"])

    retrieval_vectorizer = TfidfVectorizer(**tfidf_params)
    retrieval_matrix = retrieval_vectorizer.fit_transform(train_df["combined_text"])

    return {
        "df": df,
        "train_df": train_df,
        "test_df": test_df,
        "models": models,
        "retrieval_vectorizer": retrieval_vectorizer,
        "retrieval_matrix": retrieval_matrix,
    }


@st.cache_resource(show_spinner=True)
def load_bert_demo_assets():
    df, _ = load_data()
    train_df, test_df = get_demo_split()

    bert_model_name = "distilbert-base-uncased"
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    bert_model = AutoModel.from_pretrained(bert_model_name).to(device)
    bert_model.eval()

    def embed_texts(texts, batch_size=32, max_length=128):
        texts = list(texts)
        embeddings = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)
                outputs = bert_model(**encoded)
                token_embeddings = outputs.last_hidden_state
                attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
                summed = (token_embeddings * attention_mask).sum(dim=1)
                counts = attention_mask.sum(dim=1).clamp(min=1e-9)
                mean_pooled = summed / counts
                embeddings.append(mean_pooled.cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    best_params = {
        "type": {
            "hidden_layer_sizes": (256,),
            "activation": "relu",
            "alpha": 0.0014360105306355281,
            "learning_rate_init": 0.0015086957066607878,
        },
        "priority": {
            "hidden_layer_sizes": (256, 128),
            "activation": "relu",
            "alpha": 0.04243419132772881,
            "learning_rate_init": 0.0015263405149206766,
        },
        "queue": {
            "hidden_layer_sizes": (256, 128),
            "activation": "tanh",
            "alpha": 0.00028654875792426055,
            "learning_rate_init": 0.00042054216105093673,
        },
    }

    if BERT_CLASSIFIERS_PATH.exists():
        saved = joblib.load(BERT_CLASSIFIERS_PATH)
        models = saved["models"]
    else:
        X_bert = embed_texts(train_df["combined_text"])
        models = {
            task: build_pipeline(params)
            for task, params in best_params.items()
        }

        for task, model in models.items():
            model.named_steps["nn"].early_stopping = False
            model.fit(X_bert, train_df[task])

        joblib.dump({
            "models": models,
            "best_params": best_params,
        }, BERT_CLASSIFIERS_PATH)

    return {
        "df": df,
        "train_df": train_df,
        "test_df": test_df,
        "tokenizer": tokenizer,
        "bert_model": bert_model,
        "device": device,
        "models": models,
    }


def embed_single_text_for_demo(text: str, bert_assets):
    tokenizer = bert_assets["tokenizer"]
    bert_model = bert_assets["bert_model"]
    device = bert_assets["device"]
    with torch.no_grad():
        encoded = tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)
        outputs = bert_model(**encoded)
        token_embeddings = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
        summed = (token_embeddings * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled = summed / counts
    return mean_pooled.cpu().numpy()


def retrieve_neighbors(query_text: str, assets, top_k: int = 5):
    query_vec = assets["retrieval_vectorizer"].transform([query_text])
    sims = cosine_similarity(query_vec, assets["retrieval_matrix"]).ravel()

    candidate_df = assets["train_df"].copy()
    candidate_df = candidate_df.assign(similarity=sims)

    normalized_query = query_text.strip().lower()
    candidate_df = candidate_df[
        candidate_df["combined_text"].fillna("").astype(str).str.strip().str.lower() != normalized_query
    ].copy()

    if candidate_df.empty:
        candidate_df = assets["train_df"].copy()
        candidate_df = candidate_df.assign(similarity=sims)

    neighbors = candidate_df.sort_values("similarity", ascending=False).head(top_k).copy()
    return neighbors


def suggest_priority_reason(priority: str) -> str:
    mapping = {
        "high": "This looks urgent, so the system recommends quick attention.",
        "medium": "This looks important but not immediately critical.",
        "low": "This looks routine and can likely be handled with lower urgency.",
    }
    return mapping.get(str(priority).lower(), "This priority was inferred from similar language patterns in the training data.")


def sample_examples(df: pd.DataFrame):
    examples = {}
    for idx, queue_name in enumerate(["Technical Support", "Billing and Payments", "Customer Service", "Product Support"], start=1):
        sample = df[df["queue"] == queue_name].head(1)
        if not sample.empty:
            row = sample.iloc[0]
            examples[f"Demo Example {idx}"] = {
                "subject": row["subject"],
                "body": row["body"],
            }
    return examples


@st.cache_data(show_spinner=False)
def build_testset_examples():
    if TESTSET_EXAMPLES_PATH.exists():
        return joblib.load(TESTSET_EXAMPLES_PATH)

    train_df, test_df = get_demo_split()

    tfidf_params = {
        "max_features": 15000,
        "ngram_range": (1, 2),
        "min_df": 2,
    }

    def make_linear_svm():
        return Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", LinearSVC(dual="auto", max_iter=10000, random_state=42)),
        ])

    def make_knn(k):
        return Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance", algorithm="brute")),
        ])

    model_builders = {
        "TF-IDF + Linear SVM": make_linear_svm,
        "Retrieval + kNN (k=3)": lambda: make_knn(3),
    }

    all_predictions = {}
    for model_name, builder in model_builders.items():
        preds = {}
        for target in ["type", "priority", "queue"]:
            model = builder()
            model.fit(train_df["combined_text"], train_df[target])
            preds[target] = model.predict(test_df["combined_text"])
        all_predictions[model_name] = preds

    example_bank = {}
    for model_name in model_builders:
        preds = all_predictions[model_name]
        correct_mask = (
            (preds["type"] == test_df["type"].to_numpy()) &
            (preds["priority"] == test_df["priority"].to_numpy()) &
            (preds["queue"] == test_df["queue"].to_numpy())
        )
        candidate_rows = []
        for idx, is_correct in enumerate(correct_mask):
            if not is_correct:
                continue
            selected_triplet = (
                preds["type"][idx],
                preds["priority"][idx],
                preds["queue"][idx],
            )
            disagreement = False
            disagreement_score = 0
            for other_name, other_preds in all_predictions.items():
                if other_name == model_name:
                    continue
                other_triplet = (
                    other_preds["type"][idx],
                    other_preds["priority"][idx],
                    other_preds["queue"][idx],
                )
                if other_triplet != selected_triplet:
                    disagreement = True
                disagreement_score += sum(
                    int(a != b) for a, b in zip(selected_triplet, other_triplet)
                )
            candidate_rows.append((idx, disagreement, disagreement_score))

        candidate_rows.sort(key=lambda x: (x[1], x[2]), reverse=True)
        candidate_indices = [idx for idx, _, _ in candidate_rows]

        if not candidate_indices:
            candidate_indices = [i for i, ok in enumerate(correct_mask) if ok]

        correct_df = test_df.iloc[candidate_indices].copy()
        if correct_df.empty:
            example_bank[model_name] = []
            continue

        selected_rows = []
        seen_queues = set()
        for _, row in correct_df.iterrows():
            queue_name = row["queue"]
            if queue_name not in seen_queues:
                selected_rows.append(row)
                seen_queues.add(queue_name)
            if len(selected_rows) >= 5:
                break

        if len(selected_rows) < 5:
            used_subjects = {row["subject"] for row in selected_rows}
            for _, row in correct_df.iterrows():
                if row["subject"] not in used_subjects:
                    selected_rows.append(row)
                    used_subjects.add(row["subject"])
                if len(selected_rows) >= 5:
                    break

        examples = []
        for example_idx, row in enumerate(selected_rows, start=1):
            examples.append({
                "label": f"Test Example {example_idx}",
                "subject": row["subject"],
                "body": row["body"],
                "combined_text": row["combined_text"],
            })
        example_bank[model_name] = examples

    joblib.dump(example_bank, TESTSET_EXAMPLES_PATH)
    return example_bank


def compare_all_demo_models(subject: str, body: str, assets):
    combined_text = f"{subject.strip()} {body.strip()}".strip()
    rows = []
    for model_name in ["TF-IDF + Linear SVM", "Retrieval + kNN (k=1)", "Retrieval + kNN (k=3)"]:
        selected = assets["models"][model_name]
        rows.append({
            "model": model_name,
            "type": selected["type"].predict([combined_text])[0],
            "priority": selected["priority"].predict([combined_text])[0],
            "queue": selected["queue"].predict([combined_text])[0],
        })
    return pd.DataFrame(rows)


assets = train_demo_assets()
df, data_path = load_data()
examples = sample_examples(df)
testset_examples = build_testset_examples()
MODEL_OPTIONS = [
    "TF-IDF + Linear SVM",
    "Retrieval + kNN (k=3)",
    "DistilBERT + NN",
]

st.markdown(
    f"""
    <div class="hero-card">
      <div class="hero-title">🎀 Ticket Buddy Demo</div>
      <div class="hero-subtitle">
        It predicts <b>ticket type</b>, <b>priority</b>, and <b>destination queue</b> from support ticket text.
      </div>
        <div class="pill-row">
        <div class="pill">Dataset: {len(df):,} English tickets</div>
        <div class="pill">Models: Linear SVM, kNN (k=3), DistilBERT + NN</div>
        <div class="pill">Explainer: Retrieval + cosine similarity</div>
        <div class="pill">Source: {data_path}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Try a Ticket")
    model_name = st.selectbox(
        "Choose a prediction model",
        MODEL_OPTIONS,
        help="These model families come directly from the approaches we used in the GitHub repo.",
    )
    if model_name == "DistilBERT + NN":
        st.caption("Our neural pipeline from the repo: DistilBERT embeddings followed by a tuned MLP classifier. First load may take a little longer.")
    else:
        st.caption(assets["models"][model_name]["blurb"])
    if model_name == "DistilBERT + NN":
        choice_options = ["Custom input"] + list(examples.keys())
    else:
        model_examples = testset_examples.get(model_name, [])
        choice_options = ["Custom input"] + [ex["label"] for ex in model_examples]

    choice = st.selectbox("Load an example or write your own", choice_options)

    default_subject = ""
    default_body = ""
    if choice != "Custom input":
        if model_name == "DistilBERT + NN":
            default_subject = examples[choice]["subject"]
            default_body = examples[choice]["body"]
        else:
            chosen = next(ex for ex in testset_examples[model_name] if ex["label"] == choice)
            default_subject = chosen["subject"]
            default_body = chosen["body"]

    subject = st.text_input("Subject", value=default_subject, placeholder="Example: Account access issue")
    body = st.text_area(
        "Body",
        value=default_body,
        height=220,
        placeholder="Paste or type the support request here...",
    )

    predict_clicked = st.button("✨ Predict Ticket Route", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("What the Demo Shows")
    st.markdown(
        """
        - **Type**: Incident / Request / Problem / Change  
        - **Priority**: low / medium / high  
        - **Queue**: which support team should receive the ticket  
        - **Model selector**: switch between the main repo methods  
        - **Retrieved neighbors**: similar historical tickets for explainability
        """
    )
    st.caption("This demo is optimized for presentation and interpretability rather than production deployment.")
    st.markdown('</div>', unsafe_allow_html=True)

combined_text = f"{subject.strip()} {body.strip()}".strip()

if predict_clicked:
    if not combined_text:
        st.warning("Please enter a subject or body before running the demo.")
    else:
        if model_name == "DistilBERT + NN":
            bert_assets = load_bert_demo_assets()
            embedded = embed_single_text_for_demo(combined_text, bert_assets)
            pred_type = bert_assets["models"]["type"].predict(embedded)[0]
            pred_priority = bert_assets["models"]["priority"].predict(embedded)[0]
            pred_queue = bert_assets["models"]["queue"].predict(embedded)[0]
        else:
            selected = assets["models"][model_name]
            pred_type = selected["type"].predict([combined_text])[0]
            pred_priority = selected["priority"].predict([combined_text])[0]
            pred_queue = selected["queue"].predict([combined_text])[0]

        st.markdown(f"### Using Model: `{model_name}`")

        c1, c2, c3 = st.columns(3, gap="medium")
        for col, label, value, note in [
            (c1, "Predicted Type", pred_type, "The issue category inferred from ticket text."),
            (c2, "Predicted Priority", pred_priority, suggest_priority_reason(pred_priority)),
            (c3, "Predicted Queue", pred_queue, "The support team that should most likely handle this ticket."),
        ]:
            with col:
                st.markdown(
                    f"""
                    <div class="prediction-card">
                      <div class="prediction-label">{label}</div>
                      <div class="prediction-value">{value}</div>
                      <div class="prediction-note">{note}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if "kNN" in model_name:
            neighbors = retrieve_neighbors(combined_text, assets, top_k=5)
            st.markdown("### 🔎 Most Similar Historical Tickets")
            st.markdown("<div class='tiny-note'>These examples come from the historical ticket pool used by the retrieval model.</div>", unsafe_allow_html=True)

            display_neighbors = neighbors[[
                "similarity", "subject", "type", "priority", "queue", "tag_1", "tag_2"
            ]].copy()
            display_neighbors["similarity"] = display_neighbors["similarity"].map(lambda x: f"{x:.3f}")
            st.dataframe(display_neighbors, use_container_width=True, hide_index=True)

            preview_options = [
                f"Neighbor {i+1} | {row['queue']} | sim {row['similarity']:.3f}"
                for i, (_, row) in enumerate(neighbors.reset_index(drop=True).iterrows())
            ]
            selected_preview = st.selectbox(
                "Pick a retrieved ticket to inspect",
                preview_options,
                key=f"preview_select_{model_name}",
            )

            if st.button("👀 Show Selected Ticket Body", key=f"preview_button_{model_name}"):
                preview_index = preview_options.index(selected_preview)
                row = neighbors.reset_index(drop=True).iloc[preview_index]
                st.markdown(f"**Subject:** {row['subject']}")
                st.markdown(f"**Labels:** {row['type']} | {row['priority']} | {row['queue']}")
                st.markdown("**Body:**")
                st.info(str(row["body"]))

st.markdown("---")
st.caption(
    "Built for the ECE 569A project demo. Prediction models use TF-IDF + Linear SVM, while explanations use retrieval with cosine similarity over historical tickets."
)
