import os
import time
import json
import random
import argparse
import itertools
import subprocess
import torch

from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from transformers import AutoTokenizer
from utils.model import AllSeeingModelForCaption
from utils.collator import AllSeeingCaptionCollator


QUERY = 'Describe the image briefly.'
IMG_DIR = './'
ds_collections = {
    'coco_caption': {
        'ann_path': 'data/coco_karpathy_val.json',
        'query': QUERY,
    },
    'flickr30k': {
        'ann_path': 'data/flickr30k_karpathy_test.json',
        'query': QUERY,
    },
    'nocaps': {
        'ann_path': 'data/nocaps_val.json',
        'query': QUERY,
    },
    'vg': {
        'ann_path': 'data/vg_test_coco_format.json',
        'query': QUERY,
    },
    'refcocog_val': {
        'ann_path': 'data/refcocog_val_coco_format.json',
        'query': QUERY,
    },
    'refcocog_test': {
        'ann_path': 'data/refcocog_test_coco_format.json',
        'query': QUERY,
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


class CaptionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_dir: str,
        ann_path: str,
        query: str,
    ):
        super().__init__()
        self.ann = []
        self.img_dir = img_dir
        self.ann_path = ann_path
        self.query = query

        images = json.load(open(self.ann_path))['images']
        for ann in images:
            item = {
                'id': ann['id'],
                'image': os.path.join(img_dir, ann['image']),
            }
            if 'bbox' in ann:
                item['bbox'] = ann['bbox']
            self.ann.append(item)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        ann = self.ann[idx]

        data = {}
        data['id'] = int(ann['id'])
        data['image'] = Image.open(ann['image']).convert('RGB')

        if 'bbox' in ann:
            # x1y1x2y2
            data['bbox'] = ann['bbox']

            x1, y1, x2, y2 = data['bbox']
            assert x1 <= x2 and y1 <= y2

        data['query'] = self.query
        data['query'] = data['query'].strip()

        return data


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

    model = AllSeeingModelForCaption.from_pretrained(args.checkpoint).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False)
    tokenizer.padding_side = 'left'

    random.seed(args.seed)
    dataset = CaptionDataset(**ds_collections[args.dataset], img_dir=IMG_DIR)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=AllSeeingCaptionCollator(
            tokenizer=tokenizer,
            num_queries=model.config.num_query_tokens,
            input_size=model.config.vision_config.image_size,
        ),
    )

    image_ids = []
    captions = []
    for inputs in tqdm(dataloader, disable=torch.distributed.get_rank() != 0):
        image_ids.extend(inputs.pop('sample_ids'))
        for k in inputs:
            if isinstance(inputs[k], list):
                for i in range(len(inputs[k])):
                    inputs[k][i] = inputs[k][i].cuda()
            else:
                inputs[k] = inputs[k].cuda()

        pred = model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            use_cache=True,
        )
        captions.extend(tokenizer.batch_decode(pred, skip_special_tokens=True))

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_ids = [None for _ in range(world_size)]
    merged_captions = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_ids, image_ids)
    torch.distributed.all_gather_object(merged_captions, captions)

    merged_ids = [_ for _ in itertools.chain.from_iterable(merged_ids)]
    merged_captions = [
        _ for _ in itertools.chain.from_iterable(merged_captions)
    ]

    if torch.distributed.get_rank() == 0:
        print(f"Evaluating {args.dataset} ...")

        results = []
        for image_id, caption in zip(merged_ids, merged_captions):
            results.append({
                'image_id': int(image_id),
                'caption': caption,
            })
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file = f'{args.dataset}_{time_prefix}.json'
        json.dump(results, open(results_file, 'w'))

        coco = COCO(ds_collections[args.dataset]['ann_path'])
        coco_result = coco.loadRes(results_file)
        coco_eval = COCOEvalCap(coco, coco_result)
        coco_eval.evaluate()

        print(coco_eval.eval.items())
    torch.distributed.barrier()
