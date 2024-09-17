from typing import Dict, Sequence, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModel

from mmaction.registry import MODELS
from mmaction.utils import ConfigType, ForwardResults, SampleList, OptSampleList

import math

# https://github.com/Pilhyeon/BaSNet-pytorch/blob/master/model.py
class AttentionModule(nn.Module):
    def __init__(self, len_feature):
        super(AttentionModule, self).__init__()
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=512, kernel_size=1,
                    stride=1, padding=0),
            nn.ReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=2, kernel_size=1,
                    stride=1, padding=0),
        )
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        # x: (B, F, T)
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.softmax(out)
        # out: (B, 2, T)
        return out
        

class CASModule(nn.Module):
    def __init__(self, len_feature, num_classes):
        super(CASModule, self).__init__()
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=2048, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU()
        )
                
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=num_classes + 1, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.ReLU()
        )
        # self.drop_out = nn.Dropout(p=0.7)

    def forward(self, x):
        # x: (B, F, T)
        out = self.conv_1(x)
        out = self.conv_2(out)
        # out = self.drop_out(out)
        out = self.conv_3(out)
        # out: (B, C + 1, T)
        return out

# https://github.com/Pilhyeon/BaSNet-pytorch/blob/master/train.py
class BasNetLoss(nn.Module):
    def __init__(self, alpha = 1.0e-4):
        super(BasNetLoss, self).__init__()
        self.alpha = alpha
        self.ce_criterion = nn.BCELoss()

    def forward(self, score_fore, score_back, attention, label):
        loss = {}

        label_fore = torch.cat((label, torch.zeros((label.shape[0], 1)).to(label.device)), dim=1)
        label_back = torch.cat((torch.zeros_like(label), torch.ones((label.shape[0], 1)).to(label.device)), dim=1)

        loss_base = self.ce_criterion(score_fore, label_fore)
        loss_back = self.ce_criterion(score_back, label_back)
        loss_norm = torch.mean(torch.norm(attention, p=1, dim=2))

        loss_total = loss_base + loss_back + self.alpha * loss_norm

        loss["loss_base"] = loss_base
        loss["loss_supp"] = loss_back
        loss["loss_norm"] = loss_norm
        loss["loss_total"] = loss_total

        return loss_total, loss

@MODELS.register_module()
class RecognizerWSTAL(BaseModel):
    """An Omni-souce recognizer model framework for joint-training of image and
    video recognition tasks.

    The `backbone` and `cls_head` should be able to accept both images and
    videos as inputs.
    """

    def __init__(self, backbone: ConfigType, cls_head: ConfigType,
                 data_preprocessor: ConfigType = None) -> None:
        if data_preprocessor is None:
            # This preprocessor will only stack batch data samples.
            data_preprocessor = dict(type='ActionDataPreprocessor')
        super(RecognizerWSTAL, self).__init__(data_preprocessor=data_preprocessor)
        self.num_classes = cls_head["num_classes"]
        self.backbone = MODELS.build(backbone)
        self.feature_head = MODELS.build(dict(type="FeatureHead", backbone_name="gcn"))
        self.filter_module = AttentionModule(cls_head["len_feature"])
        self.cas_module = CASModule(cls_head["len_feature"], self.num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.bas_net_loss = BasNetLoss(cls_head["alpha"])
        
        self.k = cls_head["topk"]

    def extract_feat(self,
                     inputs: torch.Tensor,
                     stage: str = "head",
                     **kwargs):
        """Extract features at the given stage.

        Args:
            inputs (torch.Tensor): The input skeleton with shape of
                `(B, num_clips, num_person, clip_len, num_joints, 3 or 2)`.
            stage (str): The stage to output the features.
                Defaults to ``'backbone'``.

        Returns:
            tuple: THe extracted features and a dict recording the kwargs
            for downstream pipeline, which is an empty dict for the
            GCN-based recognizer.
        """
        bs, nc = inputs.shape[:2]
        inputs = inputs.reshape((bs * nc, ) + inputs.shape[2:])

        x = self.backbone(inputs) # N, M, C, T, V
        x = self.feature_head(x).reshape(bs, nc, -1).permute(0, 2, 1) # B, C, N
        
        if stage == "backbone":
            return x

        attention = self.filter_module(x)
        x = self.cas_module(x)
        cas_fore = attention[:, 0:1] * x
        cas_back = attention[:, 1:2] * x

        # slicing after sorting is much faster than torch.topk (https://github.com/pytorch/pytorch/issues/22812)
        # score_base = torch.mean(torch.topk(cas_base, self.k, dim=1)[0], dim=-1)
        sorted_scores_fore, _= cas_fore.sort(descending=True, dim=-1)
        topk_scores_fore = sorted_scores_fore[:, :, :math.ceil(sorted_scores_fore.shape[2] * self.k)]
        score_fore = torch.mean(topk_scores_fore, dim=2)

        # score_supp = torch.mean(torch.topk(cas_supp, self.k, dim=1)[0], dim=-1)
        sorted_scores_back, _= cas_back.sort(descending=True, dim=-1)
        topk_scores_back = sorted_scores_back[:, :, :math.ceil(sorted_scores_back.shape[2] * self.k)]
        score_back = torch.mean(topk_scores_back, dim=2)

        score_fore = self.softmax(score_fore)
        score_back = self.softmax(score_back)


        return score_fore, cas_fore, score_back, cas_back, attention


    def loss(self, inputs: torch.Tensor, data_samples: SampleList,
             **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): Raw Inputs of the recognizer.
                These should usually be mean centered and std scaled.
            data_samples (List[``ActionDataSample``]): The batch
                data samples. It usually includes information such
                as ``gt_label``.

        Returns:
            dict: A dictionary of loss components.
        """
        score_base, cas_base, score_supp, cas_supp, fore_weights = self.extract_feat(inputs)

        labels = [x.gt_label for x in data_samples]
        labels = torch.stack(labels).to(score_base.device)
        labels = labels.squeeze()
        labels = nn.functional.one_hot(labels, num_classes=self.num_classes)
        labels = labels.expand(score_base.shape[0], -1)

        loss_total, loss = self.bas_net_loss(score_base, score_supp, fore_weights, labels)

        return dict(loss_cls=loss_total)

    def predict(self, inputs: torch.Tensor, data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (torch.Tensor): Raw Inputs of the recognizer.
                These should usually be mean centered and std scaled.
            data_samples (List[``ActionDataSample``]): The batch
                data samples. It usually includes information such
                as ``gt_label``.

        Returns:
            List[``ActionDataSample``]: Return the recognition results.
            The returns value is ``ActionDataSample``, which usually contains
            ``pred_scores``. And the ``pred_scores`` usually contains
            following keys.

                - item (torch.Tensor): Classification scores, has a shape
                    (num_classes, )
        """
        score_base, cas_base, score_supp, cas_supp, fore_weights = self.extract_feat(inputs)

        labels = [x.gt_label for x in data_samples]
        labels = torch.stack(labels).to(score_base.device)
        labels = labels.squeeze()
        labels = nn.functional.one_hot(labels, num_classes=self.num_classes)
        labels = labels.expand(score_base.shape[0], -1)

        loss_total, loss = self.bas_net_loss(score_base, score_supp, fore_weights, labels)

        for ds in data_samples:
            ds.pred_score = loss_total

        return data_samples

    def _forward(self,
                 inputs: torch.Tensor,
                 stage: str = 'backbone',
                 **kwargs) -> ForwardResults:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            inputs (torch.Tensor): Raw Inputs of the recognizer.
            stage (str): Which stage to output the features.

        Returns:
            Union[tuple, torch.Tensor]: Features from ``backbone`` or ``neck``
            or ``head`` forward.
        """
        feats = self.extract_feat(inputs, stage=stage)
        return feats

    def forward(self,
                inputs: torch.Tensor,
                data_samples: SampleList = None,
                mode: str = 'tensor',
                **kwargs) -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes:

        - ``tensor``: Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - ``predict``: Forward and return the predictions, which are fully
        processed to a list of :obj:`ActionDataSample`.
        - ``loss``: Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[``ActionDataSample], optional): The
                annotation data of every samples. Defaults to None.
            mode (str): Return what kind of value. Defaults to ``tensor``.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of ``ActionDataSample``.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'tensor':
            return self._forward(inputs, **kwargs)
        if mode == 'predict':
            return self.predict(inputs, data_samples, **kwargs)
        elif mode == 'loss':
            return self.loss(inputs, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')
