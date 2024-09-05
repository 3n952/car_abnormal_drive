import copy
import os.path as osp

import torch

from .datasets.base import BaseDataset
from .datasets.builder import DATASETS


@DATASETS.register_module()
class MyDataset(BaseDataset):

    def init(
        self,
        ann_file,
        pipeline,
        data_prefix=None,
        test_mode=False,
        filename_tmpl="{:04}.png",
        with_offset=False,
        multi_class=False,
        num_classes=None,
        start_index=1,
        modality="RGB",
        sample_by_class=False,
        power=0.0,
        dynamic_length=False,
    ):
        self.filename_tmpl = filename_tmpl
        self.with_offset = with_offset
        super().init(
            ann_file,
            pipeline,
            data_prefix,
            test_mode,
            multi_class,
            num_classes,
            start_index,
            modality,
            sample_by_class=sample_by_class,
            power=power,
            dynamic_length=dynamic_length,
        )

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith(".json"):
            return self.load_json_annotations()
        video_infos = []
        with open(self.ann_file, "r") as fin:
            for line in fin:
                video_info = {}
                line_split = line.strip().split()

                video_info["label"] = int(line_split.pop())
                video_info["total_frames"] = int(line_split.pop())
                video_info["frame_dir"] = " ".join(line_split)

                video_infos.append(video_info)

        return video_infos

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])

        results["filename_tmpl"] = (
            f"{results['frame_dir'].split('/')[-1]}_" + "{:04}.png"
        )
        results["modality"] = self.modality
        results["start_index"] = self.start_index

        # prepare tensor in getitem
        if self.multi_class:
            onehot = torch.zeros(self.num_classes)
            onehot[results["label"]] = 1.0
            results["label"] = onehot
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])

        results["filename_tmpl"] = (
            f"{results['frame_dir'].split('/')[-1]}_" + "{:04}.png"
        )

        results["modality"] = self.modality
        results["start_index"] = self.start_index

        # prepare tensor in getitem
        if self.multi_class:
            onehot = torch.zeros(self.num_classes)
            onehot[results["label"]] = 1.0
            results["label"] = onehot
        return self.pipeline(results)