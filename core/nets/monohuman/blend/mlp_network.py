# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
from configs import cfg

# default tensorflow initialization of linear layers
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


@torch.jit.script
def fused_mean_variance(x, weight):
    mean = torch.sum(x*weight, dim=2, keepdim=True)
    var = torch.sum(weight * (x - mean)**2, dim=2, keepdim=True)
    return mean, var


class BlendNet(nn.Module):
    def __init__(self, cfg, in_feat_ch=32, n_samples=64, **kwargs):
        super(BlendNet, self).__init__()
        self.cfg = cfg
        self.anti_alias_pooling = cfg.anti_alias_pooling
        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        activation_func = nn.ReLU(inplace=False)
        self.n_samples = n_samples
        self.vis_fc = nn.Sequential(nn.Linear(5, 32),
                                    activation_func,
                                    nn.Linear(32, 1),
                                    activation_func,
                                    )
        self.weight_fc = nn.Sequential(nn.Linear(36, 36),
                                    activation_func,
                                    nn.Linear(36, 1),
                                    activation_func,
                                    )
        self.compress_fc = nn.Sequential(nn.Linear(35, 64),
                                    activation_func,
                                    nn.Linear(64, 35),
                                    )
        self.vis_fc.apply(weights_init)
        self.weight_fc.apply(weights_init)

        ###small init----
        #self.compress_fc.apply(weights_init)
        last_layer = self.compress_fc[-1]
        init_val = 1e-5
        last_layer.weight.data.uniform_(-init_val, init_val)
        last_layer.bias.data.zero_()


    def forward(self, rgb_feat):
        '''
        :param rgb_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat]
        :return: blend_rgb_feat [n_rays, n_samples, 35]
        '''

        global_feat = torch.cat([rgb_feat, torch.zeros(list(rgb_feat.shape[:-1]) + [1]).to(rgb_feat)], dim=-1)
        blend_weight = self.weight_fc(global_feat)
        blend_weight = F.softmax(blend_weight, dim=-2)
        rgb_feat = torch.sum(rgb_feat * blend_weight, dim=-2)

        return rgb_feat

