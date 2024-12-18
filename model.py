from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. مدل Word2Vec
# استفاده از یک مدل از پیش‌آموزش‌داده‌شده (Google's Word2Vec) یا آموزش مدل روی داده‌های خودتان
# مدل از قبل آموزش‌داده‌شده:
from gensim.models import KeyedVectors

from utils import text_tokenizer

model_path = "./assets/GoogleNews-vectors-negative300.bin.gz"  # مسیر مدل
word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True, limit=10000)


# 2. تبدیل سند به بردار
def vectorize_text_set(document, model, tfidf_weights=None):
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
    tokens = text_tokenizer(text, stemming=False, lemmatizing=False, remove_stop_words=False)
    return tokens


def compare_document_query(doc, query):
    doc_tokens = preprocess_text(doc)
    query_tokens = preprocess_text(query)
    document_vectors = vectorize_text_set(doc_tokens, word2vec_model)
    query_vectors = vectorize_text_set(query_tokens, word2vec_model)
    similarities = cosine_similarity([query_vectors], [document_vectors])[0]
    return similarities


def compare_document_query_as_vector(query_vectors, document_vectors_set, top=10):
    similarities = cosine_similarity([query_vectors], document_vectors_set)[0]
    top_documents = np.argsort(similarities)[::-1][:top]
    return top_documents
