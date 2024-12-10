import utils as self_utils

import re

from binary_query import process_query


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


    while True:
        print("\n\n")
        print("-----------------")
        print("search type is listed as following")
        print("1: binary")
        print("2: text (based on tf-idf)")
        print("3: exit")
        search_type = input("enter type: \n")
        match search_type:
            case "1":
                query = input("enter your query: \n")
                result = process_query(query, revert_index, whole_doc_id_set)
                print("result: \n", result)
            case "2":
                pass
            case "3":
                exit()
            case _:
                print("invalid input. allowed type (1, 2, 3, 4)")

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
