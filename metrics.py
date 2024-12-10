# وارد کردن کتابخانه‌ها
from sklearn.metrics import precision_score, recall_score, f1_score


# محاسبه معیارها
def evaluate_system(retrieved_docs, true_docs):
    y_true = []
    y_predication = []

    for doc_id in true_docs.union(retrieved_docs):
        y_true.append(doc_id in true_docs)
        y_predication.append(doc_id in retrieved_docs)

    precision = precision_score(y_true, y_predication)
    recall = recall_score(y_true, y_predication)
    f1 = f1_score(y_true, y_predication)

    return precision, recall, f1
