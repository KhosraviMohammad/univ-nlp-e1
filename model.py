from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. مدل Word2Vec
# استفاده از یک مدل از پیش‌آموزش‌داده‌شده (Google's Word2Vec) یا آموزش مدل روی داده‌های خودتان
# مدل از قبل آموزش‌داده‌شده:
from gensim.models import KeyedVectors

from utils import text_tokenizer

model_path = "path_to_pretrained_word2vec_model.bin"  # مسیر مدل
word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# 2. تبدیل سند به بردار
def document_to_vector(document, model, tfidf_weights=None):
    """
    سند را به بردار تبدیل می‌کند.
    :param document: لیست کلمات در سند
    :param model: مدل Word2Vec
    :param tfidf_weights: وزن TF-IDF کلمات (اختیاری)
    :return: بردار سند
    """
    vectors = []
    for word in document:
        if word in model.key_to_index:  # بررسی وجود کلمه در مدل
            if tfidf_weights:
                vectors.append(model[word] * tfidf_weights.get(word, 1.0))
            else:
                vectors.append(model[word])
    if vectors:
        return np.mean(vectors, axis=0)  # میانگین بردارها
    else:
        return np.zeros(model.vector_size)

# 3. پیش‌پردازش کوئری و اسناد
def preprocess_text(text):
    # همان تابع پیش‌پردازش قبلی
    tokens = text_tokenizer(text.lower())
    return [word for word in tokens if word.isalnum()]

# فرض کنیم `documents` شامل متن‌های اسناد و `queries` شامل کوئری‌ها باشد.
documents = [
    "The first document text here.",
    "Another document about libraries and research.",
    "This is a different document about Word2Vec models."
]
queries = ["libraries and Word2Vec", "research on models"]

# پیش‌پردازش اسناد
preprocessed_documents = [preprocess_text(doc) for doc in documents]

# تبدیل اسناد به بردار
document_vectors = [
    document_to_vector(doc, word2vec_model) for doc in preprocessed_documents
]

# پیش‌پردازش کوئری‌ها و تبدیل آنها به بردار
preprocessed_queries = [preprocess_text(query) for query in queries]
query_vectors = [document_to_vector(query, word2vec_model) for query in preprocessed_queries]

# 4. محاسبه شباهت کوسینوسی و پیدا کردن بهترین اسناد
for query_idx, query_vec in enumerate(query_vectors):
    similarities = cosine_similarity([query_vec], document_vectors)[0]  # شباهت کوئری با تمام اسناد
    top_documents = np.argsort(similarities)[::-1][:10]  # ۱۰ سند برتر
    print(f"Query {query_idx + 1}: {queries[query_idx]}")
    print("Top 10 Documents:")
    for doc_idx in top_documents:
        print(f"Document {doc_idx + 1} (Similarity: {similarities[doc_idx]:.4f}): {documents[doc_idx]}")
    print("-" * 50)
