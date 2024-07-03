import os
import io
import re
import json
import base64
import argparse
import gradio as gr

from PIL import ImageDraw, Image
from collections import defaultdict


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


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

    # print(objects)
    # print()
    # print(relations)
    # print()
    print(grounded_caption)
    print()
    print(clean_caption)
    print()
    return objects, relations, bbox2category


BOX_SCALE = 999
REF_START_TAG = '<ref>'
REF_END_TAG = '</ref>'
BOX_START_TAG = '<box>'
BOX_END_TAG = '</box>'
REL_START_TAG = '<pred>'
REL_END_TAG = '</pred>'
SQUARE_PAD = True
DEBUG_MODE = False
if not DEBUG_MODE:
    import sys
    sys.path.append("all-seeing-v2")

    import torch
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.utils import disable_torch_init

    disable_torch_init()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser(description="All-Seeing-Model Demo")
    parser.add_argument('--ckpt', type=str, default='OpenGVLab/ASMv2')
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--conv_mode', type=str, default='v1')
    args = parser.parse_args()

    model_path = os.path.expanduser(args.ckpt)
    model_base = None
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)

    generation_config = dict(
        do_sample=True if args.temperature > 0 else False,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        # no_repeat_ngram_size=3,
        max_new_tokens=1024,
        use_cache=True,
    )

TEXT_PLACEHOLDER_BEFORE_UPLOAD = 'Please upload your image first'
TEXT_PLACEHOLDER_AFTER_UPLOAD_BEFORE_POINT = 'Please select two points on the image to determine the position of the box'
TEXT_PLACEHOLDER_AFTER_UPLOAD = 'Type and press Enter'

BBOX_DISPLAY_INFO = "Select two points on the image and then see the bbox coordinates here."

POINT_RADIUS = 16
POINT_COLOR = (255, 0, 0)

BBOX_WIDTH = 5


def gradio_reset(user_state: dict):
    user_state = {}

    return (
        gr.update(value=None, interactive=True),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(value=None),
        gr.update(interactive=False, placeholder=TEXT_PLACEHOLDER_BEFORE_UPLOAD),
        gr.update(value=None),
        user_state,
    )


def point_reset(user_state: dict):
    user_state.pop('points', None)
    user_state.pop('bbox', None)
    user_state['image'] = user_state['original_image'].copy()
    return (
        user_state['original_image'],
        gr.update(interactive=True, placeholder=TEXT_PLACEHOLDER_AFTER_UPLOAD),
        user_state,
    )


def text_reset(user_state: dict):
    user_state.pop('input_ids', None)
    user_state.pop('attention_mask', None)
    user_state.pop('conversation', None)

    interactive = len(user_state['points']) == 2 if 'points' in user_state else True
    return (
        gr.update(value=None),
        gr.update(interactive=interactive, placeholder=TEXT_PLACEHOLDER_AFTER_UPLOAD if interactive else TEXT_PLACEHOLDER_AFTER_UPLOAD_BEFORE_POINT),
        gr.update(value=None),
        user_state,
    )


def upload_img(image, user_state):
    if image is None:
        return (
            None,
            None,
            None,
            gr.update(interactive=False, placeholder=TEXT_PLACEHOLDER_BEFORE_UPLOAD),
            user_state,
        )

    user_state['image'] = image.copy()
    user_state['original_image'] = image.copy()

    return (
        gr.update(interactive=False),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True, placeholder=TEXT_PLACEHOLDER_AFTER_UPLOAD),
        user_state,
    )


def convert_bbox_to_str(bbox, image):
    old_width = image.width
    old_height = image.height

    new_width = max(old_width, old_height)
    new_height = max(old_width, old_height)

    if SQUARE_PAD:
        bbox = [
            bbox[0] + (new_width - old_width) // 2,
            bbox[1] + (new_height - old_height) // 2,
            bbox[2] + (new_width - old_width) // 2,
            bbox[3] + (new_height - old_height) // 2,
        ]
        bbox = [[
            int(bbox[0] / new_width * BOX_SCALE),
            int(bbox[1] / new_height * BOX_SCALE),
            int(bbox[2] / new_width * BOX_SCALE),
            int(bbox[3] / new_height * BOX_SCALE),
        ]]
    else:
        bbox = [[
            int(bbox[0] / image.width * BOX_SCALE),
            int(bbox[1] / image.height * BOX_SCALE),
            int(bbox[2] / image.width * BOX_SCALE),
            int(bbox[3] / image.height * BOX_SCALE),
        ]]

    return f'{BOX_START_TAG}{bbox}{BOX_END_TAG}'


def upload_point(user_state: dict, evt: gr.SelectData):
    if 'image' not in user_state:
        raise gr.Error('Please click Upload & Start Chat button before pointing at the image')

    image = user_state['image']
    new_point = (evt.index[0], evt.index[1])

    if len(user_state.get('points', [])) >= 2:
        return (
            image,
            gr.update(interactive=True, placeholder=TEXT_PLACEHOLDER_AFTER_UPLOAD),
            user_state,
        )

    if 'points' in user_state:
        user_state['points'].append(new_point)
        assert len(user_state['points']) == 2

        point1, point2 = user_state['points']
        x1 = min(point1[0], point2[0])
        y1 = min(point1[1], point2[1])
        x2 = max(point1[0], point2[0])
        y2 = max(point1[1], point2[1])
        bbox = (x1, y1, x2, y2)
        user_state['bbox'] = bbox
        user_state['image'] = user_state['original_image'].copy()

        image = user_state['image']
        draw = ImageDraw.Draw(image)
        draw.rectangle(bbox, width=BBOX_WIDTH, outline=POINT_COLOR)
        bbox_display = convert_bbox_to_str(bbox, image)
    else:
        user_state['points'] = [new_point]
        bbox_display = BBOX_DISPLAY_INFO

        x, y = new_point
        draw = ImageDraw.Draw(image)
        draw.ellipse((x - POINT_RADIUS, y - POINT_RADIUS, x + POINT_RADIUS, y + POINT_RADIUS), fill=POINT_COLOR)

    interactive = len(user_state['points']) == 2
    return (
        image,
        gr.update(interactive=interactive, placeholder=TEXT_PLACEHOLDER_AFTER_UPLOAD if interactive else TEXT_PLACEHOLDER_AFTER_UPLOAD_BEFORE_POINT),
        bbox_display,
        user_state,
    )


def convert_images_into_markdown(image_list, caption_list):
    assert len(image_list) == len(caption_list)

    text_list = []
    for image, caption in zip(image_list, caption_list):
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        text_list.append(caption)
        text_list.append(f"![image](data:image/jpeg;base64,{img_str})")

    return '\n\n'.join(text_list)


def parse_caption_and_update_image(image: Image, caption: str):
    image = image.copy()
    image_list = []
    caption_list = []
    # sections = []
    objects, relations, bbox2category = extract_objects(caption.replace('<image>', '').replace('</s>', ''))
    for obj_name, bbox_list in objects.items():
        for idx, bbox in enumerate(bbox_list):
            x1, y1, x2, y2 = bbox
            bbox = (
                int(x1 / BOX_SCALE * image.width),
                int(y1 / BOX_SCALE * image.height),
                int(x2 / BOX_SCALE * image.width),
                int(y2 / BOX_SCALE * image.height),
            )
            # sections.append((bbox, f'{obj_name}-{idx}'))
            image_list.append(image.copy())
            caption_list.append(f'{obj_name}-{idx}')

            ImageDraw.Draw(image_list[-1]).rectangle(bbox, width=5, outline='red')

    scene_graph = []
    for rel_name, bbox_list in relations.items():
        if len(bbox_list) % 2 != 0:
            print(f'Invalid bbox num [{len(bbox_list)}] for relation [{rel_name}]')
            print(f'Current caption: {caption}')
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
                print(f'#subject_bboxes ({len(subject_bboxes)}) is not equal to #object_bboxes ({len(object_bboxes)})')
                print(f'Current caption: {caption}')
                print()

            for idx, (subj_bbox, obj_bbox) in enumerate(zip(subject_bboxes, object_bboxes)):
                subj = bbox2category[json.dumps(subj_bbox)]
                obj = bbox2category[json.dumps(obj_bbox)]
                scene_graph.append((subj, subj_bbox, obj, obj_bbox, rel_name))

                x1, y1, x2, y2 = subj_bbox
                subj_bbox = (
                    int(x1 / BOX_SCALE * image.width),
                    int(y1 / BOX_SCALE * image.height),
                    int(x2 / BOX_SCALE * image.width),
                    int(y2 / BOX_SCALE * image.height),
                )

                x1, y1, x2, y2 = obj_bbox
                obj_bbox = (
                    int(x1 / BOX_SCALE * image.width),
                    int(y1 / BOX_SCALE * image.height),
                    int(x2 / BOX_SCALE * image.width),
                    int(y2 / BOX_SCALE * image.height),
                )
                # sections.append((subj_bbox, f'{subj}-{rel_name}-{i}-{idx}-subj'))
                # sections.append((obj_bbox, f'{obj}-{rel_name}-{i}-{idx}-obj'))
                image_list.append(image.copy())
                caption_list.append(f'{subj}-{rel_name}-{obj}')

                ImageDraw.Draw(image_list[-1]).rectangle(subj_bbox, width=5, outline='red')
                ImageDraw.Draw(image_list[-1]).rectangle(obj_bbox, width=5, outline='blue')

    for t in scene_graph:
        print(t)

    # return image, sections
    return convert_images_into_markdown(image_list, caption_list)


def ask_and_answer(chatbot: list, text_input: str, user_state: dict):
    if DEBUG_MODE:
        if SQUARE_PAD:
            image = expand2square(user_state['original_image'], (0, 0, 0))
        else:
            image = user_state['original_image']

        outputs = f'hello {REF_START_TAG}world{REF_END_TAG}{BOX_START_TAG}[[0,0,444,555]]{BOX_END_TAG} {REF_START_TAG}world{REF_END_TAG}{BOX_START_TAG}[[0,0,999,999]]{BOX_END_TAG} {REL_START_TAG}on{REL_END_TAG}{BOX_START_TAG}[[0,0,999,999]]{BOX_END_TAG}{BOX_START_TAG}[[0,0,999,999]]{BOX_END_TAG}'
        chatbot.append([text_input, outputs])
        # chatbot = [[text_input, outputs]]
        return (
            chatbot,
            None,
            parse_caption_and_update_image(image, f'{text_input} {outputs}'),
            user_state,
        )

    if SQUARE_PAD:
        image = expand2square(
            user_state['original_image'],
            tuple(int(x*255) for x in image_processor.image_mean),
        )
    else:
        image = user_state['original_image']

    if 'conversation' not in user_state:
        text_input = f'<image>\n{text_input}'
        user_state['conversation'] = conv_templates[args.conv_mode].copy()

    conv = user_state['conversation']
    conv.append_message(conv.roles[0], text_input)
    conv.append_message(conv.roles[1], None)
    text_input = conv.get_prompt()

    inputs = {
        'images': image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(device),
        'input_ids': tokenizer_image_token(text_input, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device),
    }

    print('input before collator:', text_input)
    print('input after collator:', tokenizer.batch_decode(
        torch.where(inputs['input_ids'] < 0, tokenizer.pad_token_id, inputs['input_ids']),
        skip_special_tokens=False,
    )[0])

    input_ids = inputs['input_ids']
    outputs = model.generate(**inputs, **generation_config)
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != outputs[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

    # outputs = tokenizer.batch_decode(outputs[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = tokenizer.batch_decode(outputs[:, input_token_len:], skip_special_tokens=False)[0]
    outputs = outputs.replace(tokenizer.bos_token, '').replace(tokenizer.eos_token, '').strip()
    outputs = outputs.replace(tokenizer.unk_token, '').replace(tokenizer.pad_token, '').strip()

    conv.messages[-1][-1] = outputs
    chatbot.append([conv.messages[-2][-1], conv.messages[-1][-1]])
    # chatbot = [[text_input, outputs]]
    return (
        chatbot,
        None,
        parse_caption_and_update_image(image, f'{text_input} {outputs}'),
        user_state,
    )


with gr.Blocks() as demo:
    gr.HTML(
        """
        <div align='center'>
            <div style="display: inline-block;">
                <h1 style="">The All-Seeing-Model (ASM) Demo</h>
            </div>
            <div style="display: inline-block; vertical-align: bottom;">
                <img width='60' src="/file=./assets/logo.png">
            </div>
            <div style='display:flex; align-items: center; justify-content: center; gap: 0.25rem; '>
                <a href='https://github.com/OpenGVLab/all-seeing'><img src='https://img.shields.io/badge/Github-Code-blue'></a>
                <a href='https://arxiv.org/abs/2308.01907'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>
            </div>
        </div>
        """,
    )

    user_state = gr.State({})

    with gr.Row():
        with gr.Column(scale=0.5):
            image = gr.Image(type="pil", height=450)
            bbox_display = gr.Textbox(BBOX_DISPLAY_INFO, interactive=False)
            with gr.Row():
                clear_points = gr.Button("Clear points", interactive=False)
                clear_text = gr.Button("Clear text", interactive=False)
            clear_all = gr.Button("Restart", variant='primary')

        with gr.Column():
            chatbot = gr.Chatbot(label='All-Seeing-Model', height=600)
            text_input = gr.Textbox(label='User', interactive=False, placeholder=TEXT_PLACEHOLDER_BEFORE_UPLOAD)

        # with gr.Column(scale=0.5):
        #     # output_image = gr.Image(type="pil", height=450, interactive=False)
        #     output_image = gr.AnnotatedImage(height=800, width=600)

    with gr.Row():
        output_image = gr.Markdown()

    image.select(upload_point, [user_state], [image, text_input, bbox_display, user_state])
    image.upload(upload_img, [image, user_state], [image, clear_text, clear_points, text_input, user_state])

    text_input.submit(ask_and_answer, [chatbot, text_input, user_state], [chatbot, text_input, output_image, user_state])

    clear_points.click(point_reset, [user_state], [image, text_input, user_state], queue=False)
    clear_text.click(text_reset, [user_state], [chatbot, text_input, output_image, user_state], queue=False)
    clear_all.click(gradio_reset, [user_state], [image, clear_text, clear_points, chatbot, text_input, output_image, user_state], queue=False)

    gr.HTML(
        """
        <div align='left' style='font-size: large'>
            <h2 style='font-size: x-large'> User Manual: </h2>
            <ol>
                <li> Upload your image.  </li>
                <li> Select two points on the image to obtain the position of the box, which is shown in the format &lt;box&gt;[[x1, y1, x2, y2]]&lt;/box&gt;. </li>
                <li> Write a prompt based on the box coordinates. For example, `What is in this &lt;ref&gt;region&lt;/ref&gt;&lt;box&gt;[[x1, y1, x2, y2]]&lt;/box&gt;?` </li>
                <li> Begin to chat! </li>
            </ol>
            <p>NOTE1: If you want to upload another image, please click the Restart button.</p>
            <p>NOTE2: You can use "Answer the question with grounding" or "Answer the question with scene graph." to control the model behaviour.</p>
        </div>
        """
    )

server_port = 10010
for i in range(10010, 10100):
    cmd = f'netstat -aon|grep {i}'
    with os.popen(cmd, 'r') as file:
        if '' == file.read():
            server_port = i
            break

print('server_port:', server_port)
demo.queue().launch(server_name="0.0.0.0", server_port=server_port)
