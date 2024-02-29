import re
import json
import argparse
import torch

from collections import defaultdict
from torchvision.ops.boxes import box_iou

def postprocess(bbox, width, height):
    if height == width:
        pass
    elif height < width:
        delta = (width - height) // 2
        bbox[:, 1] -= delta
        bbox[:, 3] -= delta
    else:
        delta = (height - width) // 2
        bbox[:, 0] -= delta
        bbox[:, 2] -= delta
    return bbox

def parse_box(box_str, width, height, square_pad_postprocess, bbox_scale, question_text):
    PATTERN = re.compile(r'\[\[.*?\]\]')
    predict_bbox = re.findall(PATTERN, box_str)

    if len(predict_bbox) != 1:
        print(f'[Grounding Metric Warning] the output format of {box_str=} is invalid! {question_text=}')

    try:
        predict_bbox = json.loads(predict_bbox[0])
    except:
        predict_bbox = [[0, 0, 0, 0]]
        print(f'[Grounding Metric Warning] the output format of {box_str=} is invalid! {question_text=}')

    if len(predict_bbox) != 1:
        print(f'[Grounding Metric Warning] the output format of {box_str=} is invalid! {question_text=}')

    try:
        predict_bbox = predict_bbox[0]
        predict_bbox = torch.tensor(predict_bbox, dtype=torch.float32).view(-1, 4) / bbox_scale
        predict_bbox[:, 0::2] *= max(width, height) if square_pad_postprocess else width
        predict_bbox[:, 1::2] *= max(width, height) if square_pad_postprocess else height
    except:
        print(f'[Grounding Metric Warning] the output format of {box_str=} is invalid!')
        predict_bbox = torch.tensor([0,0,0,0], dtype=torch.float32).view(-1, 4)

    return postprocess(predict_bbox, width, height) if square_pad_postprocess else predict_bbox

def eval_grounding(questions, answers, square_pad_postprocess, bbox_scale):
    res = defaultdict(list)
    for answer in answers:
        question_id = answer['question_id']
        question = questions[question_id]

        pred_box = parse_box(
            box_str=answer['text'],
            width=question['width'],
            height=question['height'],
            square_pad_postprocess=square_pad_postprocess,
            bbox_scale=bbox_scale,
            question_text=question['text'],
        )

        gold_box = question['bbox']
        gold_box = torch.tensor(gold_box, dtype=torch.float32).view(-1, 4)

        category = question['category']
        iou = box_iou(pred_box, gold_box)
        iou = iou.item()

        res[category].append(iou > 0.5)

    for k, v in res.items():
        print(f'Category: {k}, # samples: {len(v)}, # Acc@0.5: {sum(v) / len(v) * 100:.2f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--square-pad-postprocess", default=False, action='store_true')
    parser.add_argument("--bbox-scale", type=int, default=999)
    args = parser.parse_args()

    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question['question_id']: question for question in questions}
    answers = [json.loads(q) for q in open(args.result_file)]
    eval_grounding(
        questions=questions,
        answers=answers,
        square_pad_postprocess=args.square_pad_postprocess,
        bbox_scale=args.bbox_scale,
    )
