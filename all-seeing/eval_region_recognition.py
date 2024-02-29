import os
import math
import time
import json
import random
import argparse
import itertools
import subprocess
import torch

from PIL import Image
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.model import AllSeeingModel
from utils.collator import AllSeeingClipCollator
from utils.region_recognition_metric import DetectionMetric


QUERY = 'What is this?'
ds_collections = {
    'coco': {
        'img_path' : 'data/coco/val2017',
        'ann_path' : 'data/coco/annotations/instances_val2017.json',
    },
    'lvis': {
        'img_path' : 'data/lvis/val2017',
        'ann_path' : 'data/lvis/annotations/lvis_v1_val.json',
    },
}


def init_distributed_mode():
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = rank % torch.cuda.device_count()

        world_size = int(os.environ["SLURM_NTASKS"])
        local_size = int(os.environ["SLURM_NTASKS_PER_NODE"])

        if "MASTER_PORT" not in os.environ:
            port = 22222
            if world_size == 1:
                for i in range(22222, 65535):
                    cmd = f'netstat -aon|grep {i}'
                    with os.popen(cmd, 'r') as file:
                        if '' == file.read():
                            port = i
                            break

            print(f'MASTER_PORT = {port}')
            os.environ["MASTER_PORT"] = str(port)

            time.sleep(3)

        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr

        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['LOCAL_WORLD_SIZE'] = str(local_size)
        os.environ['WORLD_SIZE'] = str(world_size)


class CoCoDetectionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_path,
        ann_path,
        tokenizer,
        prompts_path='utils/en_zeroshot_classification_templates.json',
        add_special_tokens=True,
    ):
        super().__init__()
        self.img_path = img_path
        self.ann_path = ann_path

        meta_info = json.load(open(self.ann_path))
        self.ann = meta_info['annotations']
        self.images = {}
        for image in meta_info['images']:
            self.images[image['id']] = image

        self.idx2str = {}
        for category_info in meta_info['categories']:
            # self.idx2str[category_info['id']] = ' '.join(category_info['name'].split('_'))
            self.idx2str[category_info['id']] = category_info['name']

        sorted_key = sorted(list(self.idx2str.keys()))
        labels_name = [self.idx2str[key] for key in sorted_key]
        self.sorted_key = sorted_key
        self.sorted_labels_name = labels_name
        # self.prompts = load_json(prompts_path)['imagenet1k']
        self.prompts = json.load(open(prompts_path))['dummy']

        labels_name_with_prompt = []
        for name in self.sorted_labels_name:
            for prompt in self.prompts:
                labels_name_with_prompt.append(prompt.format(c=name))

        tokenized_labels = tokenizer(
            labels_name_with_prompt,
            padding='longest',
            return_tensors='pt',
            add_special_tokens=add_special_tokens,
        )
        # (num_classes x num_prompts, seq_len)
        self.label_ids = tokenized_labels.input_ids
        self.label_mask = tokenized_labels.attention_mask

    def __getitem__(self, index):
        image_id = self.ann[index]['image_id']

        try:
            image = Image.open(os.path.join(
                self.img_path,
                self.images[image_id].get('file_name', f'{image_id:012d}.jpg')
            )).convert('RGB')
        except:
            image = Image.open(os.path.join(
                self.img_path.replace('val2017', 'train2017'),
                self.images[image_id].get('file_name', f'{image_id:012d}.jpg')
            )).convert('RGB')

        category_id = self.ann[index]['category_id']
        label_name = self.idx2str[category_id]
        label = f'{label_name}'

        x, y, w, h = self.ann[index]['bbox']
        bbox = (x, y, x+w, y+h)

        return {
            'bbox': bbox,
            'image': image,
            'query': QUERY,
            'label': label,
            'category_id': self.sorted_key.index(category_id),
            'image_id': image_id,
        }

    def __len__(self):
        return len(self.ann)


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    init_distributed_mode()
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    model = AllSeeingModel.from_pretrained(args.checkpoint).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False)
    tokenizer.padding_side = 'right'

    random.seed(args.seed)
    dataset = CoCoDetectionDataset(**ds_collections[args.dataset], tokenizer=tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=AllSeeingClipCollator(
            tokenizer=tokenizer,
            num_queries=model.config.num_query_tokens,
            input_size=model.config.vision_config.image_size,
            label_ids=dataset.label_ids,
            label_mask=dataset.label_mask,
        ),
    )

    metrics = DetectionMetric(
        classes=dataset.sorted_labels_name,
        ann_path=dataset.ann_path,
        num_prompts=len(dataset.prompts),
    )

    logits = []
    labels = defaultdict(list)
    for inputs in tqdm(dataloader, disable=torch.distributed.get_rank() != 0):
        curr_labels = inputs.pop('labels')
        for k, v in curr_labels.items():
            labels[k].append(v)

        for k in inputs:
            if isinstance(inputs[k], list):
                for i in range(len(inputs[k])):
                    inputs[k][i] = inputs[k][i].cuda()
            else:
                inputs[k] = inputs[k].cuda()

        pred = model.generate_cls(**inputs)
        logits.append(pred)

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    labels_list = [None for _ in range(world_size)]
    logits_list = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(labels_list, labels)
    torch.distributed.all_gather_object(logits_list, logits)

    merged_labels = {}
    for k, v in labels.items():
        merged_labels[k] = torch.cat(sum([label[k] for label in labels_list], start=[]))
    merged_logits = torch.cat(sum(logits_list, start=[]))

    if torch.distributed.get_rank() == 0:
        print(f"Evaluating {args.dataset} ...")
        res = metrics(logits=merged_logits, labels=merged_labels)
        for k, v in res.items():
            print(k, v)

    torch.distributed.barrier()
