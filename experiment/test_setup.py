import os
import pandas


def get_test_data():
    return pandas.read_csv(os.path.join(os.path.dirname(__file__), 'labeled_data.csv'))


def summarize_result(result, final_result, result_key):
    expected = result['expected']
    result_val = result[result_key]

    if result_val == 0:
        final_result['num_of_hate'] += 1

        if expected == result_val:
            final_result['correct_num_of_hate'] += 1

    elif result_val == 1:
        final_result['num_of_other'] += 1

        if expected == result_val:
            final_result['correct_num_of_other  '] += 1

    return final_result
