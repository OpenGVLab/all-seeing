import torch
import torchvision.transforms as T
from typing import List, Dict
from torchvision.transforms.functional import InterpolationMode
from .conversation import conv_templates


DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<ImageContent>"
DEFAULT_IMG_START_TOKEN = "<img>"
DEFAULT_IMG_END_TOKEN = "</img>"


def build_transform(input_size):
    # crop_pct = 224 / 256
    # size = int(input_size / crop_pct)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        # T.Resize(size, interpolation=InterpolationMode.BICUBIC),
        # T.CenterCrop(input_size),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return transform


class AllSeeingCaptionCollator:
    def __init__(
        self,
        tokenizer,
        num_queries: int,
        input_size: int,
        conv_template="multi_model",
    ):
        self.tokenizer = tokenizer
        self.image_processor = build_transform(input_size=input_size)
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else input_size

        self.conv = conv_templates[conv_template].copy()
        self.image_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        self.image_placeholder = DEFAULT_IMG_START_TOKEN + DEFAULT_IMAGE_TOKEN * num_queries + DEFAULT_IMG_END_TOKEN

    def __call__(self, data_list: List[dict]) -> Dict[str, torch.Tensor]:
        sample_ids_list = []  # B
        input_text_list = []  # B, N
        bbox_list = []  # B, 4
        pixel_values = []  # B, 3, H, W

        for data in data_list:
            sample_ids_list.append(data['id'])
            pixel_values.append(self.image_processor(data['image']))

            box = data.get('bbox', (0, 0, data['image'].width, data['image'].height))
            box = (
                box[0] / data['image'].width * self.input_size[0],
                box[1] / data['image'].height * self.input_size[1],
                box[2] / data['image'].width * self.input_size[0],
                box[3] / data['image'].height * self.input_size[1],
            )
            bbox_list.append(torch.tensor(box, dtype=torch.float32).unsqueeze(0))  # 1, 4

            query = data['query']
            self.conv.messages = []
            self.conv.append_message(self.conv.roles[0], f'{self.image_placeholder}\n{query}')
            self.conv.append_message(self.conv.roles[1], None)
            input_text = self.conv.get_prompt()
            input_text_list.append(input_text)

        self.tokenizer.padding_side = 'left'
        inputs = self.tokenizer(
            input_text_list,
            return_tensors="pt",
            padding="longest",
        )
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        inputs = {
            'sample_ids': sample_ids_list,
            'pixel_values': torch.stack(pixel_values),
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'boxes': bbox_list,
        }
        return inputs

class AllSeeingClipCollator:
    def __init__(
        self,
        tokenizer,
        num_queries: int,
        input_size: int,
        label_ids: torch.LongTensor,
        label_mask: torch.LongTensor,
        conv_template="multi_model",
    ):
        self.tokenizer = tokenizer
        self.image_processor = build_transform(input_size=input_size)
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else input_size

        image_query = DEFAULT_IMG_START_TOKEN + DEFAULT_IMAGE_TOKEN * num_queries + DEFAULT_IMG_END_TOKEN
        conv = conv_templates[conv_template].copy()
        conv.append_message(conv.roles[0], image_query + "\n{query}")
        self.prompt_template = conv.get_prompt()

        self.label_ids = label_ids
        self.label_mask = label_mask

    def __call__(self, data_list):
        boxes = []
        pixel_values = []
        query_inputs = []
        for data in data_list:
            pixel_values.append(self.image_processor(data['image']))

            if isinstance(data['query'], (tuple, list)):
                query_inputs.extend([self.prompt_template.format(query=query) for query in data['query']])
            else:
                query_inputs.append(self.prompt_template.format(query=data['query']))

            bbox = data['bbox']
            bbox = (
                bbox[0] / data['image'].width * self.input_size[0],
                bbox[1] / data['image'].height * self.input_size[1],
                bbox[2] / data['image'].width * self.input_size[0],
                bbox[3] / data['image'].height * self.input_size[1],
            )
            bboxes_normed = [bbox]

            boxes.append(torch.tensor(bboxes_normed, dtype=torch.float32))

        pixel_values = torch.stack(pixel_values, dim=0)
        query_inputs = self.tokenizer(query_inputs, padding='longest', return_tensors='pt')

        return {
            'boxes': boxes,
            'pixel_values': pixel_values,
            'query_input_ids': query_inputs.input_ids,
            'query_attention_mask': query_inputs.attention_mask,
            'label_ids': self.label_ids,
            'label_mask': self.label_mask,
            'labels': {
                'category_id': torch.tensor([data['category_id'] for data in data_list], dtype=torch.long),
                'image_id': torch.tensor([data['image_id'] for data in data_list], dtype=torch.long),
                'boxes': torch.tensor([data['bbox'] for data in data_list], dtype=torch.float32),
            },
        }
