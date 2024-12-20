from transformers import DistilBertTokenizer, DistilBertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# 1. بارگذاری مدل و توکنایزر
tokenizer = DistilBertTokenizer.from_pretrained('./assets/distilbert-base-uncased')
model = DistilBertModel.from_pretrained('./assets/distilbert-base-uncased')

# 2. داده‌ها
documents = [
    "The cat is on the mat.",
    "Dogs are great pets.",
    "I love programming and solving problems.",
    "The sun is shining bright today.",
    "Artificial intelligence is fascinating."
]
query = "What is artificial intelligence?"

# 3. پردازش کوئری و اسناد
def encode_texts(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # محاسبه میانگین برای استخراج embedding
    return embeddings

query_embedding = encode_texts([query])
document_embeddings = encode_texts(documents)

# 4. محاسبه شباهت
similarities = cosine_similarity(query_embedding, document_embeddings)[0]
results = list(zip(documents, similarities))
sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

# 5. نمایش بهترین نتایج
print("Query:", query)
print("\nTop matching documents:")
for doc, score in sorted_results[:3]:
    print(f"Score: {score:.4f} | Document: {doc}")
