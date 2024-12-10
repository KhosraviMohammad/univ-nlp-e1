def evaluate_operator(operator, left_set, right_set):
    """عملگر را اجرا می‌کند."""
    if operator == "and":
        return left_set & right_set
    elif operator == "or":
        return left_set | right_set
    elif operator == "not":
        return left_set - right_set
    return set()

def process_query(query, inverted_index, whole_data):
    """پرس‌وجو را با مدیریت اولویت‌ها و پرانتزها پردازش می‌کند."""
    tokens = query.lower().split()
    operators = []
    operands = []

    def apply_operator():
        """عملگر بالای پشته را اجرا کرده و نتیجه را ذخیره می‌کند."""
        operator = operators.pop()
        if operator == "not":
            right_set = operands.pop()
            operands.append(whole_data - right_set)
        else:
            right_set = operands.pop()
            left_set = operands.pop()
            result = evaluate_operator(operator, left_set, right_set)
            operands.append(result)

    precedence = {"not": 3, "and": 2, "or": 1}  # اولویت عملگرها
    for token in tokens:
        if token == "(":
            operators.append(token)
        elif token == ")":
            while operators and operators[-1] != "(":
                apply_operator()
            operators.pop()
        elif token in precedence:
            while (operators and operators[-1] != "(" and
                   precedence[operators[-1]] >= precedence[token]):
                apply_operator()
            operators.append(token)
        else:
            operands.append(inverted_index.get(token, set()))

    while operators:
        apply_operator()

    return operands[-1] if operands else set()
