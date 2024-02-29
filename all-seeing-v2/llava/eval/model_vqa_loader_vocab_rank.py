import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, options):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.options = options
        self.options_tokens = tokenizer(options).input_ids
        self.echo_first = 0

    def __getitem__(self, index):
        line = self.questions[index // len(self.options)]
        image_file = line["image"]
        qs = line["text"]
        label_format = line["label_format"]
        if self.model_config.mm_use_im_start_end and self.model_config.mm_use_im_patch_token:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * self.model_config.num_query_tokens + DEFAULT_IM_END_TOKEN + '\n' + qs
        elif self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], label_format.format(self.options[index % len(self.options)]))
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        attention_mask = torch.ones_like(input_ids)

        num_option_tokens = len(self.options_tokens[index % len(self.options)])
        labels = input_ids.clone()
        labels[:-num_option_tokens] = IGNORE_INDEX
        labels[-1] = IGNORE_INDEX  # </s>
        assert labels.ndim == 1

        if self.echo_first < 5:
            self.echo_first += 1
            print('inputs:', self.tokenizer.decode(torch.where(input_ids < 0, self.tokenizer.pad_token_id, input_ids)))
            print('labels:', self.tokenizer.decode(torch.where(labels < 0, self.tokenizer.pad_token_id, labels)))

        return input_ids.tolist(), attention_mask.tolist(), image_tensor, labels.tolist()

    def __len__(self):
        return len(self.questions) * len(self.options)


def collate_fn(data_list):
    max_len = max([len(data[0]) for data in data_list])

    input_ids_list = []
    attention_mask_list = []
    image_tensor_list = []
    labels_list = []
    for data in data_list:
        input_ids, attention_mask, image_tensor, labels = data
        input_ids = input_ids + [0] * (max_len - len(input_ids))
        attention_mask = attention_mask + [0] * (max_len - len(attention_mask))
        labels = [IGNORE_INDEX] * (576 - 1) + labels + [IGNORE_INDEX] * (max_len - len(labels))

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        image_tensor_list.append(image_tensor)
        labels_list.append(labels)

    return (
        torch.tensor(input_ids_list, dtype=torch.long),
        torch.tensor(attention_mask_list, dtype=torch.long),
        torch.stack(image_tensor_list),
        torch.tensor(labels_list, dtype=torch.long),
    )


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, options, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, options=options)
    data_loader = DataLoader(dataset, batch_size=batch_size * len(options), num_workers=num_workers, collate_fn=collate_fn, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        device=args.device,
        device_map={"": args.device},
    )
    # model.config.tokenizer_padding_side = 'left'

    options = json.load(open(args.options_file, "r"))
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config, options)

    for (input_ids, attention_mask, image_tensor, labels), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        # assert len(options) == 56
        assert input_ids.shape[0] == len(options), f'{input_ids.shape=}, {len(options)=}'

        input_ids = input_ids.to(device=args.device, non_blocking=True)
        attention_mask = attention_mask.to(device=args.device, non_blocking=True)
        labels = labels.to(device=args.device, non_blocking=True)

        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=image_tensor.to(dtype=torch.float16, device=args.device, non_blocking=True),
            ).logits

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        loss = loss.view(input_ids.shape[0], -1).sum(dim=1) / (labels != IGNORE_INDEX).sum(dim=1)
        # loss = (-loss).softmax(dim=-1)
        loss = (-loss)
        # sorted_scores, sorted_idx = loss.topk(k=5, largest=True)
        sorted_scores, sorted_idx = torch.sort(loss, descending=True)
        outputs = [options[option_idx.item()] for option_idx in sorted_idx]
        scores = [s.item() for s in sorted_scores]

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "scores": scores,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--options-file", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    if 'SLURM_NTASKS' in os.environ:
        args.num_chunks = int(os.environ['SLURM_NTASKS'])
        args.chunk_idx = int(os.environ['SLURM_PROCID'])
        args.answers_file = args.answers_file.replace('.jsonl', f'_{args.chunk_idx}.jsonl')
        # torch.cuda.set_device(args.chunk_idx % torch.cuda.device_count())
        args.device = args.chunk_idx % torch.cuda.device_count()
        args.device = f'cuda:{args.chunk_idx % torch.cuda.device_count()}'

    eval_model(args)
