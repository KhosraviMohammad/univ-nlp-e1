import utils as self_utils

import re

from binary_query import process_query
from model import compare_document_query, compare_document_query_as_vector, preprocess_text, vectorize_text_set, \
    word2vec_model
from utils import text_tokenizer


def split_documents(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()  # خواندن کل محتوای فایل

    # تقسیم کردن داکیومنت‌ها بر اساس شروع هر سند
    documents = re.split(r"\.I \d+\n", content)[1:]  # حذف بخش قبل از اولین `.I`

    return documents


if __name__ == "__main__":
    print("loading...")
    # مسیر فایل‌های داده
    source_file_path = "assets/CISI.ALL"  # فایل اسناد مرتبط
    comparison_metric_file_path = "assets/CISI.REL"  # فایل اسناد مرتبط
    bln_query_file_path = "assets/CISI.BLN"  # فایل اسناد مرتبط
    text_query_file_path = "assets/CISI.QRY"  # فایل اسناد مرتبط

    separated_doc_list_text = split_documents(source_file_path)
    docs_tokens = {}
    for index, document in enumerate(separated_doc_list_text):
        docs_tokens.update({index + 1: self_utils.text_tokenizer(document)})
    revert_index = self_utils.generate_inverted_index(docs_tokens)
    whole_doc_id_set = {doc_id for doc_id in range(1, len(separated_doc_list_text) + 1)}

    document_vector_set = []
    for doc in separated_doc_list_text:
        doc_tokens = preprocess_text(doc)
        document_vector_set.append(vectorize_text_set(doc_tokens, word2vec_model))


    def model_type():

        def word_2_vec():
            query = input("enter your query: \n")
            query_tokens = preprocess_text(query)
            query_vectors = vectorize_text_set(query_tokens, word2vec_model)
            result = compare_document_query_as_vector(query_vectors, document_vector_set)
            output = []
            for doc_id in result:
                output.append(doc_id + 1)
            print(f"top 10 similar documents is:\n{output}")

        while True:
            print("select the type of model which is followed")
            print("1: Word2Vec")
            print("2: exit")
            search_type = input("enter a number: \n")
            match search_type:
                case "1":
                    word_2_vec()
                case "2":
                    break
                case _:
                    print("invalid input. allowed type (1, 2,)")


    def binary_type():
        query = input("enter your query: \n")
        result = process_query(query, revert_index, whole_doc_id_set)
        print("result: \n", result)


    while True:
        print("\n\n")
        print("-----------------")
        print("search type is listed as following")
        print("1: binary")
        print("2: text (based on models)")
        print("3: exit")
        search_type = input("enter type: \n")
        match search_type:
            case "1":
                binary_type()
            case "2":
                model_type()
            case "3":
                exit()
            case _:
                print("invalid input. allowed type (1, 2, 3, 4,)")

    # 'titles' and ( 'automatically' or 'retrieving' or 'problems' or 'concerns' or 'descriptive' or 'approximate' or 'difficulties' or 'content' or 'relevance' or 'articles' )

    # documents = {
    #     1: "The cat sat on the mat.",
    #     2: "The dog sat on the log.",
    #     3: "Cats and dogs are animals.",
    #     4: "The mat was under the table."
    # }
    #
    # from dataset_utils import preprocess, generate_inverted_index, process_query
    #
    # p_data = {}
    # for doc_id, document in documents.items():
    #     p_data.update({doc_id: preprocess(document)})
    # inverted_index = generate_inverted_index(p_data)
    #
    # query1 = "cat AND mat"
    # query2 = "dog OR mat"
    # query3 = "cat AND NOT dog"
    #
    # a = process_query(query2, inverted_index)
