def average_precision(retrieved_docs):

    relevant_count = 0
    precision_sum = 0.0

    for index, doc_data in enumerate(retrieved_docs, start=1):
        if doc_data.get("is_relevant", False):
            relevant_count += 1
            precision_sum += relevant_count / index  # Precision@k

    if relevant_count == 0:
        return 0.0  # اگر سند مرتبطی وجود نداشت

    return precision_sum / relevant_count  # میانگین دقت‌ها


def mean_average_precision(documents_data_set):

    ap_sum = 0.0

    for documents_data in documents_data_set:
        ap_sum += average_precision(documents_data)

    return ap_sum / len(documents_data_set)  # میانگین AP



data = [
    [
        {
            "is_relevant": True,
        },
        {
            "is_relevant": False,
        },
        {
            "is_relevant": True,
        }
    ]

]

result = mean_average_precision(data)
print(result)