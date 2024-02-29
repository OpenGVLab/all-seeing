import re
import json
import argparse
import spacy
import torch

from collections import defaultdict
from torchvision.ops.boxes import box_iou

ECHO = False
REF_START_TAG = '<ref>'
REF_END_TAG = '</ref>'
BOX_START_TAG = '<box>'
BOX_END_TAG = '</box>'
REL_START_TAG = '<pred>'
REL_END_TAG = '</pred>'

spacy_model = "en_core_web_sm"
nlp = spacy.load(spacy_model)

def postprocess_bbox(bbox, width, height, square_pad_postprocess, bbox_scale):
    bbox = torch.tensor(bbox, dtype=torch.float32).view(-1, 4) / bbox_scale
    bbox[:, 0::2] *= max(width, height) if square_pad_postprocess else width
    bbox[:, 1::2] *= max(width, height) if square_pad_postprocess else height

    if square_pad_postprocess:
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

def postprocess_text(text_list, synonyms):
    if isinstance(text_list, str):
        text_list = [text_list]

    synonyms_text_list = []
    for text in text_list:
        # the sky --> ['the', 'sky']
        text_nlp_list = nlp(text)
        for text_nlp in text_nlp_list:
            text = text_nlp.lemma_
            synonyms_text_list.append(synonyms[text] if text in synonyms else text)
    return synonyms_text_list if len(synonyms_text_list) > 0 else ['Unknown']

def extract_objects(
    grounded_caption: str,
    grounded_pattern: str = r'<.*?>.*?<.*?>',
):
    objects = defaultdict(list)
    relations = defaultdict(list)

    clean_caption = grounded_caption
    clean_caption = clean_caption.replace(REF_START_TAG, '').replace(REF_END_TAG, '')
    clean_caption = clean_caption.replace(REL_START_TAG, '').replace(REL_END_TAG, '')
    res = re.findall(grounded_pattern, grounded_caption)

    last_tag = REF_START_TAG
    last_tag_value = 'Unknown'
    for item in res:
        clean_item = re.sub(r'<.*?>', '', item)

        if item.startswith(BOX_START_TAG):
            clean_caption = clean_caption.replace(item, '')
            try:
                clean_item = json.loads(clean_item)
            except:
                if ECHO:
                    print('Invalid format:', clean_item)
                continue
            if last_tag == REF_START_TAG:
                objects[last_tag_value].extend(clean_item)
            elif last_tag == REL_START_TAG:
                relations[last_tag_value].append(clean_item)
            else:
                raise NotImplementedError(grounded_caption)
        else:
            last_tag = REF_START_TAG if item.startswith(REF_START_TAG) else REL_START_TAG
            last_tag_value = clean_item

    bbox2category = defaultdict(list)
    for k, v in objects.items():
        for bbox in v:
            bbox2category[json.dumps(bbox)].append(k)

    return objects, relations, bbox2category

def parse_scene_graph(scene_graph_caption, width, height, square_pad_postprocess, bbox_scale, synonyms):
    _, relations, bbox2category = extract_objects(scene_graph_caption)

    scene_graph = []
    correct_format = True
    for rel_name, bbox_list in relations.items():
        if len(bbox_list) % 2 != 0:
            correct_format = False
            if ECHO:
                print(f'Invalid bbox num [{len(bbox_list)}] for relation [{rel_name}]')
                print(f'Current caption: {scene_graph_caption}')
                print()

        for i in range(0, len(bbox_list), 2):
            if i+1 >= len(bbox_list):
                continue

            subject_bboxes = bbox_list[i]
            object_bboxes = bbox_list[i+1]

            if len(subject_bboxes) == 1:
                subject_bboxes = subject_bboxes * len(object_bboxes)

            if len(object_bboxes) == 1:
                object_bboxes = object_bboxes * len(subject_bboxes)

            if len(subject_bboxes) != len(object_bboxes):
                correct_format = False
                if ECHO:
                    print(f'#subject_bboxes ({len(subject_bboxes)}) is not equal to #object_bboxes ({len(object_bboxes)})')
                    print(f'Current caption: {scene_graph_caption}')
                    print()

            for subj_bbox, obj_bbox in zip(subject_bboxes, object_bboxes):
                subj = bbox2category[json.dumps(subj_bbox)]
                obj = bbox2category[json.dumps(obj_bbox)]
                scene_graph.append([subj, subj_bbox, obj, obj_bbox, rel_name])

    parsed_scene_graph = []
    for t in scene_graph:
        try:
            t[0] = postprocess_text(t[0], synonyms=synonyms)
            t[1] = postprocess_bbox(
                t[1],
                width=width,
                height=height,
                square_pad_postprocess=square_pad_postprocess,
                bbox_scale=bbox_scale,
            )

            t[2] = postprocess_text(t[2], synonyms=synonyms)
            t[3] = postprocess_bbox(
                t[3],
                width=width,
                height=height,
                square_pad_postprocess=square_pad_postprocess,
                bbox_scale=bbox_scale,
            )

            for subj in t[0]:
                for obj in t[2]:
                    parsed_scene_graph.append([subj, t[1], obj, t[3], t[4]])

        except Exception as e:
            if ECHO:
                print(e)
                print(f'Fail to parse {scene_graph_caption}')

    return parsed_scene_graph, correct_format

def eval_psg(questions, answers, synonyms, square_pad_postprocess, bbox_scale):
    res = defaultdict(list)
    recall = []
    mean_recall = defaultdict(list)
    for answer in answers:
        question_id = answer['question_id']
        question = questions[question_id]
        # category = question['category']

        gt_scene_graph = []
        for t in question['relation_tuples']:
            gold_subj, gold_subj_bbox, gold_obj, gold_obj_bbox, gold_pred = t

            # gold_subj = gold_subj.replace('-merged', '').replace('-other', '')
            gold_subj = gold_subj.split('-')[0]
            gold_subj_bbox = torch.tensor(gold_subj_bbox, dtype=torch.float32).view(-1, 4)

            # gold_obj = gold_obj.replace('-merged', '').replace('-other', '')
            gold_obj = gold_obj.split('-')[0]
            gold_obj_bbox = torch.tensor(gold_obj_bbox, dtype=torch.float32).view(-1, 4)

            gt_scene_graph.append([gold_subj, gold_subj_bbox, gold_obj, gold_obj_bbox, gold_pred])

        pred_scene_graph, correct_format = parse_scene_graph(
            scene_graph_caption=answer['text'],
            width=question['width'],
            height=question['height'],
            square_pad_postprocess=square_pad_postprocess,
            bbox_scale=bbox_scale,
            synonyms=synonyms,
        )
        res['correct_format_ratio'].append(correct_format)
        res['num_pred_tuples'].append(len(pred_scene_graph))

        for gold in gt_scene_graph:
            match = False
            for pred in pred_scene_graph:
                pred_subj, pred_subj_bbox, pred_obj, pred_obj_bbox, pred_pred = pred
                gold_subj, gold_subj_bbox, gold_obj, gold_obj_bbox, gold_pred = gold

                subj_iou = box_iou(pred_subj_bbox, gold_subj_bbox).item()
                obj_iou = box_iou(pred_obj_bbox, gold_obj_bbox).item()

                match = (
                    pred_subj == gold_subj
                    and subj_iou >= 0.5
                    and pred_obj == gold_obj
                    and obj_iou >= 0.5
                    and pred_pred == gold_pred
                )
                if match:
                    break

            recall.append(match)
            mean_recall[gold[-1]].append(match)

    mean_recall_list = []
    for k, v in mean_recall.items():
        mean_recall_list.append(sum(v) / len(v))
        print(f'Recall({k}): {sum(v) / len(v) * 100:.2f}')

    print(f'Recall: {sum(recall) / len(recall) * 100:.2f}')
    print(f'Mean Recall for {len(mean_recall_list)} predicates: {sum(mean_recall_list) / len(mean_recall_list) * 100:.2f}')

    for k, v in res.items():
        print(f'{k}: {sum(v) / len(v):.2f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, default='/mnt/petrelfs/share_data/wangweiyun/playground/data/eval/psg/psg_eval.jsonl')
    parser.add_argument("--result-file", type=str, default='/mnt/petrelfs/share_data/wangweiyun/playground/data/eval/psg/answers/checkpoints/llava-v1.5-13b-husky-format-stage4-checkpoint-5000-exp6-6-9/checkpoint-15540/llava-v1.5-13b.jsonl')
    parser.add_argument("--synonyms-file", type=str, default='llava/eval/synonyms.txt')
    parser.add_argument("--square-pad-postprocess", default=True, action='store_true')
    parser.add_argument("--bbox-scale", type=int, default=999)
    args = parser.parse_args()

    synonyms = {}
    with open(args.synonyms_file, 'r') as file:
        lines = file.readlines()
    for line in lines:
        phrases = line.split(',')
        for p in phrases:
            synonyms[p.strip()] = phrases[0].strip()

    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question['question_id']: question for question in questions}
    answers = [json.loads(q) for q in open(args.result_file)]
    eval_psg(
        questions=questions,
        answers=answers,
        synonyms=synonyms,
        square_pad_postprocess=args.square_pad_postprocess,
        bbox_scale=args.bbox_scale,
    )
