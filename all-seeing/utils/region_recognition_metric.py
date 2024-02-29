# The code is copied from https://github.com/open-mmlab/mmeval/pull/100
# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import io
import os
import itertools
import torch
import numpy as np
import os.path as osp
import tempfile
from collections import OrderedDict
from rich.console import Console
from rich.table import Table
from typing import Dict, List, Optional, Sequence, Union
from mmeval import COCODetection as CocoMetric

from mmeval import COCODetection
from mmeval.fileio import get_local_path


try:
    from lvis import LVIS, LVISEval, LVISResults
    HAS_LVISAPI = True
except ImportError:
    HAS_LVISAPI = False


class LVISDetection(COCODetection):
    """LVIS evaluation metric.

    Evaluate AR, AP for detection tasks on LVIS dataset including proposal/box
    detection and instance segmentation.

    Args:
        ann_file (str): Path to the LVIS dataset annotation file.
        metric (str | List[str]): Metrics to be evaluated. Valid metrics
            include 'bbox', 'segm', and 'proposal'. Defaults to 'bbox'.
        iou_thrs (float | List[float], optional): IoU threshold to compute AP
            and AR. If not specified, IoUs from 0.5 to 0.95 will be used.
            Defaults to None.
        classwise (bool): Whether to return the computed results of each
            class. Defaults to False.
        proposal_nums (int): Numbers of proposals to be evaluated.
            Defaults to 300.
        metric_items (List[str], optional): Metric result names to be
            recorded in the evaluation result. If None, default configurations
            in LVIS will be used.Defaults to None.
        format_only (bool): Format the output results without performing
            evaluation. It is useful when you want to format the result
            to a specific format and submit it to the test server.
            Defaults to False.
        outfile_prefix (str, optional): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            preifx of uri corresponding backend. Defaults to None.
        print_results (bool): Whether to print the results. Defaults to True.
        logger (Logger, optional): logger used to record messages. When set to
            ``None``, the default logger will be used.
            Defaults to None.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.

    Examples:
        >>> import numpy as np
        >>> from mmeval import LVISDetection
        >>> try:
        >>>     from mmeval.metrics.utils.coco_wrapper import mask_util
        >>> except ImportError as e:
        >>>     mask_util = None
        >>>
        >>> num_classes = 4
        >>> fake_dataset_metas = {
        ...     'classes': tuple([str(i) for i in range(num_classes)])
        ... }
        >>>
        >>> lvis_det_metric = LVISDetection(
        ...     ann_file='data/lvis_v1/annotations/lvis_v1_train.json'
        ...     dataset_meta=fake_dataset_metas,
        ...     metric=['bbox', 'segm']
        ... )
        >>> lvis_det_metric(predictions=predictions)  # doctest: +ELLIPSIS  # noqa: E501
        {'bbox_AP': ..., 'bbox_AP50': ..., ...,
         'segm_AP': ..., 'segm_AP50': ..., ...,}
    """

    def __init__(self,
                 ann_file: str,
                 metric: Union[str, List[str]] = 'bbox',
                 classwise: bool = False,
                 proposal_nums: int = 300,
                 iou_thrs: Optional[Union[float, Sequence[float]]] = None,
                 metric_items: Optional[Sequence[str]] = None,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 backend_args: Optional[dict] = None,
                 **kwargs) -> None:
        if not HAS_LVISAPI:
            raise RuntimeError(
                'Package lvis is not installed. Please run "pip install '
                'git+https://github.com/lvis-dataset/lvis-api.git".')
        super().__init__(
            metric=metric,
            classwise=classwise,
            iou_thrs=iou_thrs,
            metric_items=metric_items,
            format_only=format_only,
            outfile_prefix=outfile_prefix,
            backend_args=backend_args,
            **kwargs)
        self.proposal_nums = proposal_nums  # type: ignore

        with get_local_path(
                filepath=ann_file, backend_args=backend_args) as local_path:
            self._lvis_api = LVIS(local_path)

    def add_predictions(self, predictions: Sequence[Dict]) -> None:
        """Add predictions to `self._results`.

        Args:
            predictions (Sequence[dict]): A sequence of dict. Each dict
                representing a detection result for an image, with the
                following keys:

                - img_id (int): Image id.
                - bboxes (numpy.ndarray): Shape (N, 4), the predicted
                  bounding bboxes of this image, in 'xyxy' foramrt.
                - scores (numpy.ndarray): Shape (N, ), the predicted scores
                  of bounding boxes.
                - labels (numpy.ndarray): Shape (N, ), the predicted labels
                  of bounding boxes.
                - masks (list[RLE], optional): The predicted masks.
                - mask_scores (np.array, optional): Shape (N, ), the predicted
                  scores of masks.
        """
        self.add(predictions)

    def add(self, predictions: Sequence[Dict]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add the intermediate results to `self._results`.

        Args:
            predictions (Sequence[dict]): A sequence of dict. Each dict
                representing a detection result for an image, with the
                following keys:

                - img_id (int): Image id.
                - bboxes (numpy.ndarray): Shape (N, 4), the predicted
                  bounding bboxes of this image, in 'xyxy' foramrt.
                - scores (numpy.ndarray): Shape (N, ), the predicted scores
                  of bounding boxes.
                - labels (numpy.ndarray): Shape (N, ), the predicted labels
                  of bounding boxes.
                - masks (list[RLE], optional): The predicted masks.
                - mask_scores (np.array, optional): Shape (N, ), the predicted
                  scores of masks.
        """
        for prediction in predictions:
            assert isinstance(prediction, dict), 'The prediciton should be ' \
                f'a sequence of dict, but got a sequence of {type(prediction)}.'  # noqa: E501
            self._results.append(prediction)

    def __call__(self, *args, **kwargs) -> Dict:
        """Stateless call for a metric compute."""

        # cache states
        cache_results = self._results
        cache_lvis_api = self._lvis_api
        cache_cat_ids = self.cat_ids
        cache_img_ids = self.img_ids

        self._results = []
        self.add(*args, **kwargs)
        metric_result = self.compute_metric(self._results)

        # recover states from cache
        self._results = cache_results
        self._lvis_api = cache_lvis_api
        self.cat_ids = cache_cat_ids
        self.img_ids = cache_img_ids

        return metric_result

    def compute_metric(  # type: ignore
            self, results: list) -> Dict[str, Union[float, list]]:
        """Compute the LVIS metrics.

        Args:
            results (List[tuple]): A list of tuple. Each tuple is the
                prediction and ground truth of an image. This list has already
                been synced across all ranks.

        Returns:
            dict: The computed metric. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        # handle lazy init
        if len(self.cat_ids) == 0:
            self.cat_ids = self._lvis_api.get_cat_ids()
        if len(self.img_ids) == 0:
            self.img_ids = self._lvis_api.get_img_ids()

        # convert predictions to coco format and dump to json file
        result_files = self.results2json(results, outfile_prefix)

        eval_results: OrderedDict = OrderedDict()
        table_results: OrderedDict = OrderedDict()
        if self.format_only:
            self.logger.info('results are saved in '
                             f'{osp.dirname(outfile_prefix)}')
            return eval_results

        lvis_gt = self._lvis_api

        for metric in self.metrics:
            self.logger.info(f'Evaluating {metric}...')

            try:
                lvis_dt = LVISResults(lvis_gt, result_files[metric])
            except IndexError:
                self.logger.warning(
                    'The testing results of the whole dataset is empty.')
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            lvis_eval = LVISEval(lvis_gt, lvis_dt, iou_type)
            lvis_eval.params.imgIds = self.img_ids
            metric_items = self.metric_items
            if metric == 'proposal':
                lvis_eval.params.max_dets = self.proposal_nums
                lvis_eval.evaluate()
                lvis_eval.accumulate()
                lvis_eval.summarize()
                if metric_items is None:
                    metric_items = [
                        f'AR@{self.proposal_nums}',
                        f'ARs@{self.proposal_nums}',
                        f'ARm@{self.proposal_nums}',
                        f'ARl@{self.proposal_nums}'
                    ]
                results_list = []
                for k, v in lvis_eval.get_results().items():
                    if k in metric_items:
                        val = float(v)
                        results_list.append(f'{round(val * 100, 2):0.2f}')
                        eval_results[k] = val
                table_results[f'{metric}_result'] = results_list

            else:
                lvis_eval.evaluate()
                lvis_eval.accumulate()
                lvis_eval.summarize()
                lvis_results = lvis_eval.get_results()

                if metric_items is None:
                    metric_items = [
                        'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'APr',
                        'APc', 'APf'
                    ]

                results_list = []
                for metric_item, v in lvis_results.items():
                    if metric_item in metric_items:
                        key = f'{metric}_{metric_item}'
                        val = float(v)
                        results_list.append(f'{round(val * 100, 2)}')
                        eval_results[key] = val
                table_results[f'{metric}_result'] = results_list

                if self.classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = lvis_eval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        # the dimensions of precisions are
                        # [num_thrs, num_recalls, num_cats, num_area_rngs]
                        nm = self._lvis_api.load_cats([catId])[0]
                        precision = precisions[:, :, idx, 0]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{round(ap * 100, 2)}'))
                        eval_results[f'{metric}_{nm["name"]}_precision'] = ap

                    table_results[f'{metric}_classwise_result'] = \
                        results_per_category
            # Save lvis summarize print information to logger
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                lvis_eval.print_results()
            self.logger.info('\n' + redirect_string.getvalue())
        if tmp_dir is not None:
            tmp_dir.cleanup()
        # if the testing results of the whole dataset is empty,
        # does not print tables.
        if self.print_results and len(table_results) > 0:
            self._print_results(table_results)
        return eval_results

    def _print_results(self, table_results: dict) -> None:
        """Print the evaluation results table.

        Args:
            table_results (dict): The computed metric.
        """
        for metric in self.metrics:
            result = table_results[f'{metric}_result']

            if metric == 'proposal':
                table_title = ' Recall Results (%)'
                if self.metric_items is None:
                    assert len(result) == 4
                    headers = [
                        f'AR@{self.proposal_nums}',
                        f'ARs@{self.proposal_nums}',
                        f'ARm@{self.proposal_nums}',
                        f'ARl@{self.proposal_nums}'
                    ]
                else:
                    assert len(result) == len(self.metric_items)  # type: ignore # yapf: disable # noqa: E501
                    headers = self.metric_items  # type: ignore
            else:
                table_title = f' {metric} Results (%)'
                if self.metric_items is None:
                    assert len(result) == 9
                    headers = [
                        f'{metric}_AP', f'{metric}_AP50', f'{metric}_AP75',
                        f'{metric}_APs', f'{metric}_APm', f'{metric}_APl',
                        f'{metric}_APr', f'{metric}_APc', f'{metric}_APf'
                    ]
                else:
                    assert len(result) == len(self.metric_items)
                    headers = [
                        f'{metric}_{item}' for item in self.metric_items
                    ]
            table = Table(title=table_title)
            console = Console()
            for name in headers:
                table.add_column(name, justify='left')
            table.add_row(*result)
            with console.capture() as capture:
                console.print(table, end='')
            self.logger.info('\n' + capture.get())

            if self.classwise and metric != 'proposal':
                self.logger.info(
                    f'Evaluating {metric} metric of each category...')
                classwise_table_title = f' {metric} Classwise Results (%)'
                classwise_result = table_results[f'{metric}_classwise_result']

                num_columns = min(6, len(classwise_result) * 2)
                results_flatten = list(itertools.chain(*classwise_result))
                headers = ['category', f'{metric}_AP'] * (num_columns // 2)
                results_2d = itertools.zip_longest(*[
                    results_flatten[i::num_columns] for i in range(num_columns)
                ])

                table = Table(title=classwise_table_title)
                console = Console()
                for name in headers:
                    table.add_column(name, justify='left')
                for _result in results_2d:
                    table.add_row(*_result)
                with console.capture() as capture:
                    console.print(table, end='')
                self.logger.info('\n' + capture.get())

CocoMetric = COCODetection
LVISMetric = LVISDetection

class DetectionMetric:
    def __init__(
        self,
        ann_path,
        classes,
        num_prompts=1,
        output_dir=None,
    ):
        self.ann_file = ann_path
        self.classes = classes
        self.num_prompts = num_prompts
        self.output_dir = output_dir

    def __call__(self, logits, labels):
        logits = logits.cpu()
        image_id = labels['image_id'].cpu()
        boxes = labels['boxes'].cpu()
        labels = labels['category_id'].cpu()

        logits = logits.reshape(-1, len(self.classes), self.num_prompts).mean(-1)
        print(f'logits: {logits.shape}, labels: {labels.shape}, bboxes: {boxes.shape}, image_ids: {image_id.shape}')

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            torch.save(logits, os.path.join(self.output_dir, 'logits.pt'))
            # with open(os.path.join(self.output_dir, 'labels.json'), 'w') as file:
            #     json.dump(labels, file)

        correct = (logits.argmax(-1) == labels).sum()
        total = logits.shape[0]

        if logits.shape[1] == 1000:
            return dict(eval_coco_acc=correct/total)

        result_dict = dict()
        for img_id in set(image_id.tolist()):
            result_dict[img_id] = dict()
            result_dict[img_id]["bboxes"] = []
            result_dict[img_id]["scores"] = []
            result_dict[img_id]["labels"] = []

        if logits.shape[1] < 100:
            Metric = CocoMetric
            topk = logits.shape[1]
        elif logits.shape[1] < 2000:
            Metric = LVISMetric
            topk = 80
        else:
            Metric = CocoMetric
            topk = 200
        print('topk:', topk)

        metric = Metric(
            dataset_meta=dict(classes=self.classes),
            ann_file=self.ann_file,
            metric='bbox',
            classwise=True,
            format_only=False,
            print_results=False,
        )

        probs = logits.to(torch.float32).softmax(-1)
        for img_id, box, prob in zip(image_id, boxes, probs):
            _, topk_indices = prob.topk(topk)
            aug_boxes = [box] * topk
            scores = prob[topk_indices].tolist()
            labels = topk_indices

            img_id = int(img_id)
            result_dict[img_id]["bboxes"].extend(aug_boxes)
            result_dict[img_id]["scores"].extend(scores)
            result_dict[img_id]["labels"].extend(labels)

        for k, v in result_dict.items():
            img_pred = dict(img_id=k,
                            bboxes=np.array(v['bboxes']),
                            scores=np.array(v['scores']),
                            labels=np.array(v['labels']))
            metric.add_predictions([img_pred])

        metric_results = metric.compute_metric(metric._results)
        return dict(eval_coco_acc=correct/total, **metric_results)
