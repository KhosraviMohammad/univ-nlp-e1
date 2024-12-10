def process_query(query, inverted_index):
    tokens = query.lower().split()
    result_set = set()
    operator = None

    for token in tokens:
        if token == "and":
            operator = "and"
        elif token == "or":
            operator = "or"
        elif token == "not":
            operator = "not"
        else:
            # کلمه معمولی: بازیابی اسناد مربوطه
            current_set = inverted_index.get(token, set())
            if operator == "not":
                result_set -= current_set
            elif operator == "and":
                result_set &= current_set
            elif operator == "or" or not result_set:
                result_set |= current_set
            else:
                result_set = current_set

    return result_set
