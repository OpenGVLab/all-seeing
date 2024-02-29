import json
import argparse
from collections import defaultdict

def eval_psg_pred_cls(questions, answers, topk, topk_each_query=1000):
    id2graph = defaultdict(list)
    id2graph_gt = defaultdict(list)
    for answer in answers:
        question_id = answer['question_id']
        question = questions[question_id]

        image_id = question['image']

        selected_options = answer['text']
        selected_options = selected_options[:topk_each_query]
        scores = answer['scores']
        correct_options = question['correct_option']
        subj = question['subj']
        obj = question['obj']

        for option, score in zip(selected_options, scores):
            id2graph[image_id].append((subj, obj, option, score))

        for option in correct_options:
            id2graph_gt[image_id].append((subj, obj, option))

    for image_id in id2graph:
        id2graph[image_id] = sorted(id2graph[image_id], key=lambda x:x[-1], reverse=True)
        id2graph[image_id] = id2graph[image_id][:topk]
        id2graph[image_id] = [tuple(item[:-1]) for item in id2graph[image_id][:-1]]

    recall = []
    mean_recall = defaultdict(list)
    for image_id in id2graph_gt:
        graph = id2graph[image_id]
        graph_gt = id2graph_gt[image_id]
        for t in graph_gt:
            recall.append(t in graph)
            mean_recall[t[-1]].append(t in graph)

    mean_recall_list = []
    for k, v in mean_recall.items():
        mean_recall_list.append(sum(v) / len(v))
        print(f'Recall({k})({len(v)}): {sum(v) / len(v) * 100:.2f}')
    print(f'Recall@{topk}: {sum(recall) / len(recall) * 100:.2f}')
    print(f'mRecall@{topk} for {len(mean_recall_list)} predicates: {sum(mean_recall_list) / len(mean_recall_list) * 100:.2f}')
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, default='/mnt/petrelfs/share_data/wangweiyun/playground/data/eval/psg_pred_cls/psg_pred_cls_full_square_pad.jsonl')
    parser.add_argument("--result-file", type=str, default='/mnt/petrelfs/share_data/wangweiyun/playground/data/eval/psg_pred_cls/answers_full/checkpoints/llava-v1.5-13b-husky-format-stage4-checkpoint-5000-exp6-6-9/checkpoint-15540/llava-v1.5-13b.jsonl')
    args = parser.parse_args()

    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question['question_id']: question for question in questions}
    answers = [json.loads(q) for q in open(args.result_file)]
    
    for topk in [20, 50, 100]:
        eval_psg_pred_cls(questions=questions, answers=answers, topk=topk)
