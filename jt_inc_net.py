import jittor as jt
import jittor.nn as nn
from easydict import EasyDict
import copy
import math
from jt_vit import vit_base_patch16_224_in21k_adapter
from jt_vit_lora import vit_base_patch16_224_in21k_lora, vit_base_patch16_224_lora
import numpy as np


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce

        self.weight = jt.random((self.out_features, in_features), 'float32')

        if sigma:
            self.sigma = jt.ones(1)  # Jittor 使用 jt.ones 初始化
        else:
            self.sigma = None

        self.reset_parameters()
        self.use_RP = False
        # self.W_rand = None  # 需要显式初始化

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.shape[1])
        # Jittor 的 uniform_ 实现不同
        self.weight = jt.init.uniform((self.out_features, self.in_features),
                                 low=-stdv, high=stdv, dtype='float32')

        if self.sigma is not None:
            self.sigma.assign(jt.ones(1))  # Jittor 使用 assign 进行值替换

    def execute(self, input, func=None):  # Jittor 使用 execute 而非 forward
        if not self.use_RP:
            out = nn.linear(jt.normalize(input, p=2, dim=1), jt.normalize(self.weight, p=2, dim=1))
        else:
            if self.W_rand is not None:
                inn = nn.relu(input @ self.W_rand)
                if func is not None:
                    inn = func(inn, input, is_test=True)
            else:
                inn = input

            out = nn.linear(inn, self.weight)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()
        name = args["convnet_type"].lower()
        
        load_model = ''
        if '_adapter' in name:
            ffn_num = 64
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                d_model=768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
            )

            model = vit_base_patch16_224_in21k_adapter(num_classes=0, global_pool=False, drop_path_rate=0.0,
                                                       tuning_config=tuning_config)
            load_model = 'vit_base_patch16_224.pt'
            model.out_dim = 768

        elif '_lora' in name:
            ffn_num = 64
            if args["model_name"] == "lora":
                tuning_config = EasyDict(
                    # AdaptFormer
                    ffn_adapt=True,
                    ffn_option="parallel",
                    ffn_adapter_layernorm_option="none",
                    ffn_adapter_init_option="lora",
                    ffn_adapter_scalar="0.1",
                    ffn_num=ffn_num,
                    d_model=768,
                    # VPT related
                    vpt_on=False,
                    vpt_num=0,
                )
                if name == "pretrained_vit_b16_224_in21k_lora":
                    model = vit_base_patch16_224_in21k_lora(num_classes=0,
                                                            global_pool=False,
                                                            drop_path_rate=0.0,
                                                            tuning_config=tuning_config)
                    load_model = 'vit_base_patch16_224_in21k_lora.pt'
                    model.out_dim = 768
                elif name == "pretrained_vit_b16_224_lora":
                    model = vit_base_patch16_224_lora(num_classes=0,
                                                      global_pool=False, drop_path_rate=0.0,
                                                      tuning_config=tuning_config)
                    load_model = 'vit_base_patch16_224_lora.pt'
                    model.out_dim = 768
                else:
                    raise NotImplementedError("Unknown type {}".format(name))

        else:
            raise NotImplementedError("Unknown type {}".format(name))

        self.convnet = model.eval()

        # =================================================
        params = jt.load(load_model)
        missing_keys = []
        for name, p in model.named_parameters():
            if name not in params:
                missing_keys.append(name)

        # 冻结除 adapter 外的参数
        for name, p in model.named_parameters():
            if name in missing_keys:
                # p.start_grad()  # jittor 要这样设置grad
                p.requires_grad = True
            else:
                # p.stop_grad()
                p.requires_grad = False
        # =================================================
        # self.convnet = model
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def execute(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass


class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes):
        fc = CosineLinear(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            weight = jt.cat([weight, jt.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def execute(self, x, func=None):
        x = self.convnet(x)
        out = self.fc(x, func)

        return out