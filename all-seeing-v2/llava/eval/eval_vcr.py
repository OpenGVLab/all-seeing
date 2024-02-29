import json
import argparse
from collections import defaultdict

def eval_vcr(questions, answers):
    choice_freq = dict()
    res = defaultdict(list)
    for answer in answers:
        question_id = answer['question_id']
        question = questions[question_id]
        category = question['category']

        selected_option = answer['text']
        correct_option = question['correct_option']

        if len(correct_option) > 1:
            print(f'Invalid correct option: {correct_option}\ncurrent question: {question}')
            res[category].append(False)
            continue

        if len(selected_option) > 1:
            print(f'Invalid selected option: {selected_option}\ncurrent question: {question}')
            res[category].append(False)
            continue

        if category not in choice_freq:
            choice_freq[category] = defaultdict(int)
        choice_freq[category][selected_option] += 1
        res[category].append(correct_option == selected_option)

    for k, v in res.items():
        print(f'Category: {k}, # samples: {len(v)}, Acc: {sum(v) / len(v) * 100:.2f}, {choice_freq[k]}')

    correct = 0
    total = 0
    for qa, qar in zip(res['qa'], res['qar']):
        correct += qa and qar
        total += 1
    print(f'Category: q->ar, # samples: {total}, Acc: {correct / total * 100:.2f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    args = parser.parse_args()

    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question['question_id']: question for question in questions}
    answers = [json.loads(q) for q in open(args.result_file)]
    eval_vcr(
        questions=questions,
        answers=answers,
    )
