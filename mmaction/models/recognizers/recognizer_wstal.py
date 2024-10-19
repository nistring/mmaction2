from typing import Dict, Sequence, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModel

from mmaction.registry import MODELS
from mmaction.utils import ConfigType, ForwardResults, SampleList, OptSampleList

import math

# https://keras.io/examples/timeseries/timeseries_classification_from_scratch/
class ACMNet(nn.Module):
    def __init__(self, len_feature, num_classes):
        super(ACMNet, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=len_feature, kernel_size=3,
                    stride=1, padding=1),
            nn.BatchNorm1d(len_feature),
            nn.ReLU(),
            nn.Conv1d(in_channels=len_feature, out_channels=len_feature, kernel_size=3,
                    stride=1, padding=1),
            nn.BatchNorm1d(len_feature),
            nn.ReLU(),
            nn.Conv1d(in_channels=len_feature, out_channels=len_feature, kernel_size=3,
                    stride=1, padding=1),
            nn.BatchNorm1d(len_feature),
            nn.ReLU(),
        )
        self.conv_2 = nn.Conv1d(in_channels=len_feature, out_channels=3, kernel_size=3,
            stride=1, padding=1)
        self.fc = nn.Linear(in_features=len_feature, out_features=num_classes + 1)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        x = self.conv_1(x)
        attention = self.softmax(self.conv_2(x).permute(0, 2, 1))
        x = self.fc(x.permute(0, 2, 1))
        instance = x * attention[..., [0]]
        context = x * attention[..., [1]]
        background = x * attention[..., [2]]
        return instance, context, background
        

@MODELS.register_module()
class RecognizerWSTAL(BaseModel):
    """An Omni-souce recognizer model framework for joint-training of image and
    video recognition tasks.

    The `backbone` and `cls_head` should be able to accept both images and
    videos as inputs.
    """

    def __init__(self, backbone: ConfigType, cls_head: ConfigType, weight, freeze: bool = False, topk: float = 1/4,
                 data_preprocessor: ConfigType = None) -> None:
        if data_preprocessor is None:
            # This preprocessor will only stack batch data samples.
            data_preprocessor = dict(type='ActionDataPreprocessor')
        super(RecognizerWSTAL, self).__init__(data_preprocessor=data_preprocessor)
        self.backbone = MODELS.build(backbone)
        if freeze:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.num_classes = cls_head["num_classes"]
        self.attention = ACMNet(cls_head["in_channels"], self.num_classes)
        self.cls_head = MODELS.build(dict(type="FeatureHead", backbone_name="gcn"))
        self.ce_criterion = nn.CrossEntropyLoss(weight=torch.Tensor(weight))
        self.topk = topk
        torch.set_printoptions(sci_mode=False)

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
        
        if stage == "backbone":
            return x

        x = self.cls_head(x).reshape(bs, nc, -1).permute(0, 2, 1) # B, C, M
        instance, context, background = self.attention(x) # B, M, num_classes+1
        # x = nn.functional.softmax(self.cls_head(x).reshape(bs, nc, -1), -1)

        return instance, context, background


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
        instance, context, background = self.extract_feat(inputs)
        n = instance.shape[1]
        instance = torch.mean(torch.sort(instance, dim=1, descending=True)[0][:, :round(n * self.topk[0])], dim=1) # B, num_classes+1
        context = torch.mean(torch.sort(context, dim=1, descending=True)[0][:, :round(n * self.topk[1])], dim=1) # B, num_classes+1
        background = torch.mean(torch.sort(background, dim=1, descending=True)[0][:, :round(n * self.topk[2])], dim=1) # B, num_classes+1

        labels = nn.functional.one_hot(torch.LongTensor([x.gt_label for x in data_samples]), self.num_classes).to(instance.device).float()
        b = labels.shape[0]
        inst_label = torch.cat((labels, torch.zeros((b, 1)).to(instance.device)), dim=-1)
        cont_label = torch.cat((labels, torch.ones((b, 1)).to(instance.device)), dim=-1) / 2
        back_label = torch.cat((torch.zeros_like(labels), torch.ones((b, 1)).to(instance.device)), dim=-1)

        loss = self.ce_criterion(instance, inst_label) + \
            self.ce_criterion(context, cont_label) + \
            self.ce_criterion(background, back_label)

        return dict(loss_cls=loss)

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
        instance, context, background = self.extract_feat(inputs)
        n = instance.shape[1]
        instance = torch.mean(torch.sort(instance, dim=1, descending=True)[0][:, :round(n * self.topk[0])], dim=1) # B, num_classes+1
        context = torch.mean(torch.sort(context, dim=1, descending=True)[0][:, :round(n * self.topk[1])], dim=1) # B, num_classes+1
        background = torch.mean(torch.sort(background, dim=1, descending=True)[0][:, :round(n * self.topk[2])], dim=1) # B, num_classes+1

        labels = nn.functional.one_hot(torch.LongTensor([x.gt_label for x in data_samples]), self.num_classes).to(instance.device).float()
        b = labels.shape[0]
        inst_label = torch.cat((labels, torch.zeros((b, 1)).to(instance.device)), dim=-1)
        cont_label = torch.cat((labels, torch.ones((b, 1)).to(instance.device)), dim=-1) / 2
        back_label = torch.cat((torch.zeros_like(labels), torch.ones((b, 1)).to(instance.device)), dim=-1)

        loss = self.ce_criterion(instance, inst_label) + \
            self.ce_criterion(context, cont_label) + \
            self.ce_criterion(background, back_label)

        instance = torch.softmax(instance[:, :2], dim=1)
        for i, ds in enumerate(data_samples):
            ds.pred_score = instance[i, [1]]
            # ds.pred_score = loss

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
