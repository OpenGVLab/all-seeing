import json
import argparse
from collections import defaultdict

def eval_crpe(questions, answers):
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

    step = 4
    for k, v in res.items():
        assert len(v) % step == 0

        merged_v = []
        for i in range(0, len(v), step):
            merged_v.append(all(v[i:i+step]))
        v = merged_v
        print(f'Category: {k}, # samples: {len(v)}, Acc: {sum(v) / len(v) * 100:.2f}, {choice_freq[k]}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    args = parser.parse_args()

    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question['question_id']: question for question in questions}
    answers = [json.loads(q) for q in open(args.result_file)]
    eval_crpe(
        questions=questions,
        answers=answers,
    )
