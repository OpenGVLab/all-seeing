import re
import os
import json
from PIL import Image, ImageDraw
from collections import defaultdict

BOX_SCALE = 999
REF_START_TAG = '<ref>'
REF_END_TAG = '</ref>'
BOX_START_TAG = '<box>'
BOX_END_TAG = '</box>'
REL_START_TAG = '<pred>'
REL_END_TAG = '</pred>'

colors = [(255, 6, 27), (94, 163, 69), (50, 103, 185), (255, 184, 44), (244, 114, 54), (120, 121, 180), (249, 30, 136), (60, 228, 208), (91, 36, 197), (0, 85, 24)]

def draw_box(image, bbox, bbox2category, bbox2cid, width=8, alpha=32, square_pad=False):
    # bbox2cid is used to ensure that the specific box is always drawed with the same color
    x1, y1, x2, y2 = bbox
    category = bbox2category[json.dumps(bbox)]

    if square_pad:
        bbox = bbox.copy()
        bbox = [
            bbox[0] / BOX_SCALE * max(image.height, image.width),
            bbox[1] / BOX_SCALE * max(image.height, image.width),
            bbox[2] / BOX_SCALE * max(image.height, image.width),
            bbox[3] / BOX_SCALE * max(image.height, image.width),
        ]

        if image.height == image.width:
            pass
        elif image.height < image.width:
            delta = (image.width - image.height) // 2
            bbox[1] -= delta
            bbox[3] -= delta
        else:
            delta = (image.height - image.width) // 2
            bbox[0] -= delta
            bbox[2] -= delta

        for i in range(len(bbox)):
            if bbox[i] < 0:
                bbox[i] = 0

        bbox = tuple(bbox)

    else:
        bbox = (
            x1 / BOX_SCALE * image.width,
            y1 / BOX_SCALE * image.height,
            x2 / BOX_SCALE * image.width,
            y2 / BOX_SCALE * image.height,
        )

    if bbox not in bbox2cid:
        bbox2cid[bbox] = len(bbox2cid) % len(colors)

    # draw box
    ImageDraw.Draw(image).rectangle(bbox, outline=colors[bbox2cid[bbox]], width=width)

    # fill box
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    rectangle_position = bbox
    rectangle_color = (*colors[bbox2cid[bbox]], alpha)
    draw.rectangle(rectangle_position, fill=rectangle_color)
    image = Image.alpha_composite(image.convert('RGBA'), overlay)

    return image

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

    last_tag = None
    last_tag_value = None
    for item in res:
        clean_item = re.sub(r'<.*?>', '', item)

        if item.startswith(BOX_START_TAG):
            clean_caption = clean_caption.replace(item, '')
            try:
                clean_item = json.loads(clean_item)
            except Exception as e:
                print('Invalid format:', clean_item)
                raise e
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

    return objects, relations, bbox2category, clean_caption

def visualize(image, grounded_caption, save_path, square_pad):
    image.save(save_path.format('origin'))
    objects, relations, bbox2category, clean_caption = extract_objects(grounded_caption)

    # vis objects
    print()
    print('parsed objects:')
    bbox2cid = {}
    for tag, bbox_list in objects.items():
        print(tag, bbox_list)
        image_to_draw = image.copy()
        for bbox in bbox_list:
            image_to_draw = draw_box(image=image_to_draw, bbox=bbox, bbox2category=bbox2category, bbox2cid=bbox2cid, width=3, square_pad=square_pad)
        image_to_draw.save(save_path.format(f'object_{tag}'))

    # extract scene graph
    scene_graph = []
    for rel_name, bbox_list in relations.items():
        assert len(bbox_list) % 2 == 0
        for i in range(0, len(bbox_list), 2):
            subject_bboxes = bbox_list[i]
            object_bboxes = bbox_list[i+1]

            if len(subject_bboxes) == 1:
                subject_bboxes = subject_bboxes * len(object_bboxes)

            if len(object_bboxes) == 1:
                object_bboxes = object_bboxes * len(subject_bboxes)

            assert len(subject_bboxes) == len(object_bboxes)
            for subj_bbox, obj_bbox in zip(subject_bboxes, object_bboxes):
                subj = bbox2category[json.dumps(subj_bbox)]
                obj = bbox2category[json.dumps(obj_bbox)]
                scene_graph.append((subj, subj_bbox, obj, obj_bbox, rel_name))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print()
    print('parsed scene graph tuples:')
    for t in scene_graph:
        print(t)

    print()
    print('clean caption:')
    print(clean_caption)
    print()
    print('scene graph caption:')
    print(grounded_caption)
    print()

    # vis objects occured in the scene graph
    bbox2cid = {}
    image_to_draw = image.copy()
    for rel_name, bbox_list in relations.items():
        for bboxes in bbox_list:
            for bbox in bboxes:
                image_to_draw = draw_box(image=image_to_draw, bbox=bbox, bbox2category=bbox2category, bbox2cid=bbox2cid, square_pad=square_pad)
    image_to_draw.save(save_path.format('relation'))

    # vis relations
    image_cnt = 0
    for rel_name, bbox_list in relations.items():
        relation_cnt = 0
        for bboxes in bbox_list:
            if relation_cnt % 2 == 0:
                image_to_draw = image.copy()

            for bbox in bboxes:
                image_to_draw = draw_box(image=image_to_draw, bbox=bbox, bbox2category=bbox2category, bbox2cid=bbox2cid, alpha=64, square_pad=square_pad)

            relation_cnt += 1
            if relation_cnt % 2 == 0:
                image_to_draw.save(save_path.format(f'relation_{rel_name}_{image_cnt}'))
                image_cnt += 1

        if relation_cnt % 2 != 0:
            print(f"Format Warning: {rel_name}, {relation_cnt}")


def main():
    img_path = 'example.jpg'

    # scene_graph_caption_wo_square_pad = "In the image, a tall red <ref>bus</ref><box>[[290, 201, 835, 850]]</box> is <pred>driving on</pred><box>[[290, 201, 835, 850]]</box><box>[[0, 603, 984, 998]]</box> a <ref>road</ref><box>[[0, 603, 984, 998]]</box> through a busy intersection in a metropolitan area. The bus is surrounded by various <ref>cars</ref><box>[[827, 589, 885, 681], [1, 603, 238, 803], [927, 595, 935, 619], [905, 591, 931, 644], [930, 591, 938, 615], [878, 589, 910, 659]]</box>, with one car <pred>beside</pred><box>[[1, 603, 238, 803]]</box><box>[[290, 201, 835, 850]]</box> the bus and others <pred>driving on</pred><box>[[1, 603, 238, 803]]</box><box>[[0, 603, 984, 998]]</box> the same road. There are multiple <ref>people</ref><box>[[346, 327, 398, 389], [944, 597, 969, 698], [249, 576, 266, 653], [952, 659, 999, 991], [224, 569, 243, 653], [159, 582, 181, 617], [499, 300, 568, 379], [142, 569, 162, 613], [650, 619, 680, 663], [394, 315, 447, 383], [373, 549, 443, 652], [969, 599, 989, 686]]</box> visible in the scene, with one <pred>in</pred><box>[[373, 549, 443, 652]]</box><box>[[290, 201, 835, 850]]</box> the bus and others on the surrounding <ref>pavement</ref><box>[[207, 613, 999, 861]]</box> and <ref>road</ref><box>[[0, 603, 984, 998]]</box>. The backdrop features <ref>buildings</ref><box>[[0, 0, 892, 657]]</box> and <ref>trees</ref><box>[[0, 0, 999, 572]]</box>, with a clear <ref>sky</ref><box>[[682, 0, 999, 512]]</box> above. The scene captures the essence of city life with its bustling traffic and urban architecture."
    # save_dir = f'./temp_sgc_wo_square_pad'
    # image = Image.open(img_path)
    # os.makedirs(save_dir, exist_ok=True)
    # visualize(image, scene_graph_caption_wo_square_pad, os.path.join(save_dir, '{}.png'), square_pad=False)

    scene_graph_caption = "In the image, a tall red <ref>bus</ref><box>[[290, 260, 835, 781]]</box> is <pred>driving on</pred><box>[[290, 260, 835, 781]]</box><box>[[0, 583, 984, 900]]</box> a <ref>road</ref><box>[[0, 583, 984, 900]]</box> through a busy intersection in a metropolitan area. The bus is surrounded by various <ref>cars</ref><box>[[827, 572, 885, 646], [1, 583, 238, 744], [927, 577, 935, 596], [905, 573, 931, 616], [930, 573, 938, 593], [878, 572, 910, 628]]</box>, with one car <pred>beside</pred><box>[[1, 583, 238, 744]]</box><box>[[290, 260, 835, 781]]</box> the bus and others <pred>driving on</pred><box>[[1, 583, 238, 744]]</box><box>[[0, 583, 984, 900]]</box> the same road. There are multiple <ref>people</ref><box>[[346, 361, 398, 411], [944, 578, 969, 659], [249, 561, 266, 623], [952, 628, 999, 895], [224, 556, 243, 623], [159, 566, 181, 594], [499, 340, 568, 403], [142, 556, 162, 591], [650, 596, 680, 631], [394, 352, 447, 406], [373, 540, 443, 622], [969, 580, 989, 650]]</box> visible in the scene, with one <pred>in</pred><box>[[373, 540, 443, 622]]</box><box>[[290, 260, 835, 781]]</box> the bus and others on the surrounding <ref>pavement</ref><box>[[207, 591, 999, 790]]</box> and <ref>road</ref><box>[[0, 583, 984, 900]]</box>. The backdrop features <ref>buildings</ref><box>[[0, 98, 892, 626]]</box> and <ref>trees</ref><box>[[0, 98, 999, 558]]</box>, with a clear <ref>sky</ref><box>[[682, 98, 999, 510]]</box> above. The scene captures the essence of city life with its bustling traffic and urban architecture."
    save_dir = f'./temp_sgc'
    image = Image.open(img_path)
    os.makedirs(save_dir, exist_ok=True)
    visualize(image, scene_graph_caption, os.path.join(save_dir, '{}.png'), square_pad=True)


if __name__ == '__main__':
    main()
