import numpy as np
from joblib import Parallel, delayed
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import os
from sklearn.metrics.pairwise import cosine_similarity

# بارگذاری مدل و توکنایزر
tokenizer = DistilBertTokenizer.from_pretrained('./assets/distilbert-base-uncased')
model = DistilBertModel.from_pretrained('./assets/distilbert-base-uncased')

# تابع برای توکنیز کردن و استخراج بردارهای تعبیه‌شده
def encode_texts(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # محاسبه میانگین برای استخراج embedding
    return embeddings

# ذخیره‌سازی بردارهای تعبیه‌شده در کش
def save_embeddings_to_cache(embeddings, cache_filename="embeddings_cache.pt"):
    torch.save(embeddings, cache_filename)
    print(f"Embeddings saved to {cache_filename}")

# بارگذاری بردارهای تعبیه‌شده از کش
def load_embeddings_from_cache(cache_filename="embeddings_cache.pt"):
    # return torch.load(cache_filename, weights_only=True)
    return torch.load(cache_filename)

# بررسی وجود کش
def embeddings_in_cache(cache_filename="embeddings_cache.pt"):
    return os.path.exists(cache_filename)

# پردازش یک دسته از متون و استخراج بردارهای تعبیه‌شده
def encode_batch(batch):
    inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# پردازش دسته‌ای و موازی متون
def batch_encode_texts_parallel(texts, batch_size=32, n_jobs=-1, cache_filename="embeddings_cache.pt"):
    """
    پردازش دسته‌ای و موازی متون.
    """
    # اگر کش وجود داشته باشد، از آن بارگذاری می‌کنیم
    if embeddings_in_cache(cache_filename):
        print("Loading embeddings from cache...")
        return load_embeddings_from_cache(cache_filename)

    # تقسیم متون به دسته‌ها
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    # استفاده از پردازش موازی برای هر دسته
    results = Parallel(n_jobs=n_jobs)(delayed(encode_batch)(batch) for batch in batches)

    # ترکیب تمام بردارهای دسته‌ها به یک Tensor
    embeddings = torch.cat(results, dim=0)

    # ذخیره‌سازی بردارهای تعبیه‌شده در کش
    save_embeddings_to_cache(embeddings, cache_filename)

    return embeddings


def bert_cosine_similarity(query_embedding, docks_embedding, top=10):
    similarities = cosine_similarity(query_embedding, docks_embedding)[0]
    top_documents = np.argsort(similarities)[::-1][:top]
    return top_documents


# داده‌ها
# texts = [
#     "The cat is on the mat.",
#     "Dogs are great pets.",
#     "I love programming and solving problems.",
#     "The sun is shining bright today.",
#     "Artificial intelligence is fascinating."
# ]
#
# # استفاده از پردازش موازی و کش
# embeddings = batch_encode_texts_parallel(texts, batch_size=2, n_jobs=2)

# print(embeddings)



# document_bert_vector_set = load_embeddings_from_cache()
# query = "What is artificial intelligence?"
# query_vectors = encode_texts([query])
# result = bert_cosine_similarity(query_vectors, document_bert_vector_set)
# print(result)
