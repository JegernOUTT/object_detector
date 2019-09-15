from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataclasses import dataclass
from os.path import join
from typing import Dict, List, Tuple

import torch
import torch.utils.model_zoo as model_zoo
from torch import nn

from object_detector.modelling.abstract import AbstractModel, BaseModelParams, CheckpointsParams, BaseModelConfig


__all__ = ['DLAModelConfig']

BatchNorm = nn.BatchNorm2d


def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = BatchNorm(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(AbstractModel):
    def __init__(self,
                 levels,
                 channels,
                 params: BaseModelParams,
                 num_classes=1000,
                 block=BasicBlock,
                 residual_root=False,
                 pool_size=7):
        super().__init__(params)
        self._add_possible_input(name='image', stride=1)

        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=pool_size, stride=1,
                      padding=3, bias=False),
            BatchNorm(channels[0]),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)
        self._add_possible_output('level1', stride=2)

        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self._add_possible_output('level2', stride=4)

        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self._add_possible_output('level3', stride=8)

        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self._add_possible_output('level4', stride=16)

        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)
        self._add_possible_output('level5', stride=32)

        # By default return last layer
        self.set_outputs(['level5'])

    def init_weights(self, params: CheckpointsParams):
        pass

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                BatchNorm(planes),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, input_: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = input_['image']

        x = self.base_layer(x)
        for i in range(6):
            x = self._infer_and_save_if_needed(layer_name=f'level{i}', input=x)

        return self._pop_saved_output()

    def load_pretrained_model(self, data: str, name: str, hash: str):
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        self.load_state_dict(model_weights, strict=False)


def dla34(pretrained=False, **kwargs):
    model = DLA(levels=[1, 1, 1, 2, 2, 1],
                channels=[16, 32, 64, 128, 256, 512],
                block=BasicBlock,
                params=BaseModelParams(name='dla34'),
                **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model


def dla46_c(pretrained=False, **kwargs):
    Bottleneck.expansion = 2
    model = DLA(levels=[1, 1, 1, 2, 2, 1],
                channels=[16, 32, 64, 64, 128, 256],
                block=Bottleneck,
                params=BaseModelParams(name='dla46_c'),
                **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla46_c', hash='2bfd52c3')
    return model


def dla46x_c(pretrained=False, **kwargs):
    BottleneckX.expansion = 2
    model = DLA(levels=[1, 1, 1, 2, 2, 1],
                channels=[16, 32, 64, 64, 128, 256],
                block=BottleneckX,
                params=BaseModelParams(name='dla46x_c'),
                **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla46x_c', hash='d761bae7')
    return model


def dla60x_c(pretrained=False, **kwargs):
    BottleneckX.expansion = 2
    model = DLA(levels=[1, 1, 1, 2, 3, 1],
                channels=[16, 32, 64, 64, 128, 256],
                block=BottleneckX,
                params=BaseModelParams(name='dla60x_c'),
                **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla60x_c', hash='b870c45c')
    return model


def dla60(pretrained=False, **kwargs):
    Bottleneck.expansion = 2
    model = DLA(levels=[1, 1, 1, 2, 3, 1],
                channels=[16, 32, 128, 256, 512, 1024],
                block=Bottleneck,
                params=BaseModelParams(name='dla60'),
                **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla60', hash='24839fc4')
    return model


def dla60x(pretrained=False, **kwargs):
    BottleneckX.expansion = 2
    model = DLA(levels=[1, 1, 1, 2, 3, 1],
                channels=[16, 32, 128, 256, 512, 1024],
                params=BaseModelParams(name='dla60x'),
                block=BottleneckX,
                **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla60x', hash='d15cacda')
    return model


def dla102(pretrained=False, **kwargs):
    Bottleneck.expansion = 2
    model = DLA(levels=[1, 1, 1, 3, 4, 1],
                channels=[16, 32, 128, 256, 512, 1024],
                block=Bottleneck,
                residual_root=True,
                params=BaseModelParams(name='dla102'),
                **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla102', hash='d94d9790')
    return model


def dla102x(pretrained=False, **kwargs):
    BottleneckX.expansion = 2
    model = DLA(levels=[1, 1, 1, 3, 4, 1],
                channels=[16, 32, 128, 256, 512, 1024],
                block=BottleneckX,
                residual_root=True,
                params=BaseModelParams(name='dla102x'),
                **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla102x', hash='ad62be81')
    return model


def dla102x2(pretrained=False, **kwargs):
    BottleneckX.cardinality = 64
    model = DLA(levels=[1, 1, 1, 3, 4, 1],
                channels=[16, 32, 128, 256, 512, 1024],
                block=BottleneckX,
                residual_root=True,
                params=BaseModelParams(name='dla102x2'),
                **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla102x2', hash='262837b6')
    return model


def dla169(pretrained=False, **kwargs):
    Bottleneck.expansion = 2
    model = DLA(levels=[1, 1, 2, 3, 5, 1],
                channels=[16, 32, 128, 256, 512, 1024],
                block=Bottleneck,
                residual_root=True,
                params=BaseModelParams(name='dla169'),
                **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla169', hash='0914e092')
    return model


models = {
    'dla34': dla34,
    'dla46_c': dla46_c,
    'dla46x_c': dla46x_c,
    'dla60x_c': dla60x_c,
    'dla60': dla60,
    'dla60x': dla60x,
    'dla102': dla102,
    'dla102x': dla102x,
    'dla102x2': dla102x2,
    'dla169': dla169
}


@dataclass
class DLAModelConfig(BaseModelConfig):
    type: str = 'dla34'
    batch_norm_type = nn.BatchNorm2d
    pretrained = True
    _outputs = {2: 'level1', 4: 'level2', 8: 'level3', 16: 'level4', 32: 'level5'}

    def create_module(self) -> torch.nn.Module:
        BatchNorm = self.batch_norm_type
        return self._get_model_by_type(self.type)(pretrained=self.pretrained)

    @staticmethod
    def _get_model_by_type(type: str):
        assert type in models, f'Unknown DLA type: {type}, must be one of {tuple(models.keys())}'
        return models[type]

    def _get_outputs_by_strides(self, strides: Tuple):
        return [self._outputs[s] for s in strides ]

    def _get_all_outputs(self):
        return list(self._outputs.values())


if __name__ == '__main__':
    model = dla46_c(pretrained=True)
    output = model({'image': torch.zeros((1, 3, 128, 128))})
    print(output)
