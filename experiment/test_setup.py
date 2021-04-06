import os
import json
import pandas
from denylist.detect import is_hate_speech as denylist_is_hate_speech
from nlp.detect import is_hate_speech as nlp_is_hate_speech


def get_pef_test_data():
    return pandas.read_csv(os.path.join(os.path.dirname(__file__), 'labeled_data.csv'))


def get_bias_test_data():
    with open(os.path.join(os.path.dirname(__file__), 'bias_data.json')) as f:
        data = json.load(f)

    return data


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


def summarize_bias_result(is_hate_speech, final_result, result_key):
    if is_hate_speech:
        final_result['num_of_' + result_key] += 1

    return final_result


def run_perf_test():
    data = get_pef_test_data()
    perf_results = []
    num_of_hate = 0
    num_of_offensive = 0
    num_of_other = 0

    for i in range(len(data.values)):
        test_str = data.values[i][6]
        test_expected = data.values[i][5]

        if test_expected == 0:
            num_of_hate += 1
        elif test_expected == 1:
            num_of_offensive += 1
        else:
            num_of_other += 1

        perf_results.append({'expected': test_expected, 'result_nlp': 0 if nlp_is_hate_speech(
            test_str) else 2, 'result_denylist': 0 if denylist_is_hate_speech(test_str) else 2})

    nlp_results = {'num_of_hate': 0, 'num_of_other': 0,
                   'correct_num_of_hate': 0, 'correct_num_of_other': 0}
    denylist_results = {'num_of_hate': 0, 'num_of_other': 0,
                        'correct_num_of_hate': 0, 'correct_num_of_other': 0}

    for result in perf_results:
        nlp_results = summarize_result(result, nlp_results, 'result_nlp')
        denylist_results = summarize_result(
            result, denylist_results, 'result_denylist')

    return (denylist_results, nlp_results, num_of_hate, num_of_offensive, num_of_other)


def run_bias_test():
    data = get_bias_test_data()
    num_of_sexism = 0
    num_of_racism = 0

    nlp_results = {'num_of_sexism': 0, 'num_of_racism': 0}
    denylist_results = {'num_of_sexism': 0, 'num_of_racism': 0}

    for n in data:
        n_label = n['label']
        n_text = n['text']

        if n_label == 'sexism':
            num_of_sexism += 1
        elif n_label == 'racism':
            num_of_racism += 1
        else:
            continue

        nlp_results = summarize_bias_result(
            nlp_is_hate_speech(n_text), nlp_results, n_label)
        denylist_results = summarize_bias_result(
            denylist_is_hate_speech(n_text), denylist_results, n_label)

    return (denylist_results, nlp_results, num_of_sexism, num_of_racism)
