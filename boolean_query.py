import re

import boolean

from utils import text_tokenizer


def clear_query(query):
    # r"'([^']*)'|\b\w+\b|[()]"
    pattern = r"'[^']*'|\b\w+\b|[()]"

    matches = re.findall(pattern, query)

    parsed_query = ""
    for match in matches:
        parsed_query += " " + match.strip("'")

    # cleaned_result = [token.strip("'") if token.startswith("'") and token.endswith("'") else token for token in matches]
    return parsed_query

def process_query(query, inverted_index, whole_data):
    """
    کوئری منطقی را پردازش می‌کند و نتیجه را بازمی‌گرداند.
    """
    # ایجاد یک الگبرای منطقی

    query = clear_query(query)

    algebra = boolean.BooleanAlgebra()

    # تبدیل کلمات به مجموعه‌ها
    def map_token_to_set(token):
        if token.lower() == "not":
            return "NOT"
        elif token.lower() == "and":
            return "AND"
        elif token.lower() == "or":
            return "OR"
        elif token == "(" or token == ")":
            return token
        else:
            # بازگرداندن مجموعه مرتبط با کلمه
            return f"SET_{token}"

    # تجزیه کوئری و جایگزینی کلمات با توکن‌ها
    tokens = query.split()
    mapped_query = " ".join(map_token_to_set(token) for token in tokens)

    # پارس و ارزیابی کوئری
    expression = algebra.parse(mapped_query)

    # تابع بازگشتی برای ارزیابی هر گره در کوئری منطقی
    def evaluate_expression(expr):
        if isinstance(expr, boolean.Symbol):
            word = expr.obj
            if word.startswith("SET_"):
                # بازگرداندن مجموعه مرتبط
                word = word[4:]
                tokens = text_tokenizer(word) or ["", ]
                token = tokens[0]
                return inverted_index.get(token, set())
        elif expr.operator == "~":
            return whole_data - evaluate_expression(expr.args[0])
        elif expr.operator == "&":
            index = 0
            result = evaluate_expression(expr.args[index])
            for _ in expr.args:
                index += 1
                result &= evaluate_expression(expr.args[index])
                if index == len(expr.args) - 1:
                    break
            return result
        elif expr.operator == "|":
            index = 0
            result = evaluate_expression(expr.args[index])
            for _ in expr.args:
                index += 1
                result |= evaluate_expression(expr.args[index])
                if index == len(expr.args) - 1:
                    break
            return result
        return set()

    return evaluate_expression(expression)