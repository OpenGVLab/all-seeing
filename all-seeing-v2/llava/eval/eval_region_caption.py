import os
import json
import argparse
from collections import defaultdict
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

def eval_grounding(annotations, questions, answers, save_path):
    res = defaultdict(list)
    for answer in answers:
        question_id = answer['question_id']
        question = questions[question_id]

        category = question['category']
        image_id = question['ann_id']
        caption = answer['text']
        res[category].append({'image_id': image_id, 'caption': caption})

    for category in res:
        annotation_file = annotations[category]
        prediction_file = os.path.join(save_path, f'{category}_coco_format.json')
        predictions = res[category]
        with open(prediction_file, 'w') as file:
            json.dump(predictions, file)

        coco = COCO(annotation_file)
        coco_result = coco.loadRes(prediction_file)
        coco_eval = COCOEvalCap(coco, coco_result)
        coco_eval.evaluate()

        print(f'Category: {category}')
        print(json.dumps(coco_eval.eval, indent=1))
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    args = parser.parse_args()

    base_dir = os.path.dirname(args.question_file)
    annotations = {
        'vg_test': os.path.join(base_dir, 'vg_test_coco_format.json'),
        'refcocog_val': os.path.join(base_dir, 'refcocog_val_coco_format.json'),
        'refcocog_test': os.path.join(base_dir, 'refcocog_test_coco_format.json'),
    }

    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question['question_id']: question for question in questions}
    answers = [json.loads(q) for q in open(args.result_file)]
    eval_grounding(
        annotations=annotations,
        questions=questions,
        answers=answers,
        save_path=os.path.dirname(args.result_file),
    )
