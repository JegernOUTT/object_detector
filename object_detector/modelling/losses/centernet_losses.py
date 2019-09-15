from dataclasses import dataclass
from enum import Enum
from typing import Union

import torch

from object_detector.modelling.losses.common_losses import *
from object_detector.modelling.losses.common_losses import _sigmoid


class XYLossType(Enum):
    Focal = 0
    MSE = 1


class RegLossType(Enum):
    L1 = 0
    SmoothL1 = 1
    NormalizedL1 = 2
    WeightedL1 = 3


def _get_loss_by_config(config: Union[XYLossType, RegLossType]):
    if config == XYLossType.Focal:
        return FocalLoss()
    elif config == XYLossType.MSE:
        return torch.nn.MSELoss()
    elif config == RegLossType.L1:
        return RegL1Loss()
    elif config == RegLossType.SmoothL1:
        return RegLoss()
    elif config == RegLossType.NormalizedL1:
        return NormRegL1Loss()
    elif config == RegLossType.WeightedL1:
        return RegWeightedL1Loss()


@dataclass
class CenternetDetectionLossConfig():
    xy_loss: XYLossType = XYLossType.Focal
    xy_offset_loss: RegLossType = RegLossType.L1
    wh_loss: RegLossType = RegLossType.L1
    xy_weight = 1.
    xy_offset_weight = 1.
    wh_weight = 0.1
    xy_key_name = 'centernet_xy'
    xy_offset_key_name = 'centernet_xy_offset'
    wh_key_name = 'centernet_wh'
    common_loss_key_name = 'centernet_det_loss'
    xy_loss_key_name = 'xy_loss'
    xy_offset_loss_key_name = 'xy_offset_loss'
    wh_loss_key_name = 'wh_loss_loss'


class CenternetDetectionLoss(torch.nn.Module):
    def __init__(self, config: CenternetDetectionLossConfig):
        super(CenternetDetectionLoss, self).__init__()
        self._config = config

        self.xy_loss = _get_loss_by_config(self._config.xy_loss)
        self.xy_offset_loss = _get_loss_by_config(self._config.xy_offset_loss)
        self.wh_loss = _get_loss_by_config(self._config.wh_loss)

    def forward(self, y_true, y_pred):
        xy_loss, xy_offset_loss, wh_loss = 0., 0., 0.

        xy_true = y_true[self._config.xy_key_name]
        xy_offset_true = y_true[self._config.xy_offset_key_name]
        wh_true = y_true[self._config.wh_key_name]

        xy_pred = y_pred[self._config.xy_key_name]
        xy_offset_pred = y_pred[self._config.xy_offset_key_name]
        wh_pred = y_pred[self._config.wh_key_name]

        if self._config.xy_loss != XYLossType.MSE:
            xy_pred = _sigmoid(xy_pred)

        xy_loss += self.xy_loss(xy_pred, xy_true)
        if opt.dense_wh:
            mask_weight = batch['dense_wh_mask'].sum() + 1e-4
            wh_loss += (self.crit_wh(output['wh'] * batch['dense_wh_mask'],
                                     batch['dense_wh'] * batch['dense_wh_mask']) /
                        mask_weight) / opt.num_stacks
        elif opt.cat_spec_wh:
            wh_loss += self.crit_wh(
                output['wh'], batch['cat_spec_mask'],
                batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
        else:
            wh_loss += self.crit_reg(
                output['wh'], batch['reg_mask'],
                batch['ind'], batch['wh']) / opt.num_stacks

        xy_offset_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                        batch['ind'], batch['reg']) / opt.num_stacks

        loss = self._config.xy_weight * xy_loss + \
               self._config.xy_offset_weight * xy_offset_loss + \
               self._config.wh_weight * wh_loss

        return {self._config.common_loss_key_name: loss,
                self._config.xy_loss_key_name: xy_loss,
                self._config.xy_offset_loss_key_name: xy_offset_loss,
                self._config.wh_loss_key_name: wh_loss}
