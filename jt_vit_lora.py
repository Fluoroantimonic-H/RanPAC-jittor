from jittor import nn
import jittor as jt
import math
from functools import partial
from collections import OrderedDict
import numpy as np


class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()

        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        self.adapter_layernorm_option = adapter_layernorm_option
        
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = jt.ones(1)
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.dropout = dropout

        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with jt.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                # nn.init.zeros_(self.down_proj.bias)   jittor中没有nn.init.zeros_，需要用下面的形式
                # nn.init.zeros_(self.up_proj.weight)
                # nn.init.zeros_(self.up_proj.bias)
                self.down_proj.bias.assign(jt.zeros(self.down_proj.bias.shape))
                self.up_proj.weight.assign(jt.zeros(self.up_proj.weight.shape))
                self.up_proj.bias.assign(jt.zeros(self.up_proj.bias.shape))

    def execute(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual

        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.dropout(down, p=self.dropout, is_train=self.is_training())

        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = jt.rand(shape, dtype=x.dtype)
    binary_tensor = (random_tensor < keep_prob).float()

    if keep_prob > 0. and scale_by_keep:
        binary_tensor = binary_tensor / keep_prob

    return x * binary_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def execute(self, x):
        return drop_path(x, self.drop_prob, self.is_training(), self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"

# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
#         self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
#         self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
#
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def _shape(self, var: jt.Var, seq_len: int, bsz: int):
#         return var.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#
#     def execute(self, x):
#         B, N, C = x.shape
#
#         q = self.q_proj(x)
#         k = self._shape(self.k_proj(x), N, B).reshape(B * self.num_heads, N, self.head_dim)
#         v = self._shape(self.v_proj(x), N, B).reshape(B * self.num_heads, N, self.head_dim)
#         q = self._shape(q, N, B).reshape(B * self.num_heads, N, self.head_dim)
#
#         attn_weights = jt.bmm(q, k.transpose(0, 2, 1)) * self.scale
#         attn_probs = nn.softmax(attn_weights, dim=-1)
#         attn_probs = self.attn_drop(attn_probs)
#
#         attn_output = jt.bmm(attn_probs, v)
#
#         # 变回 (B, N, C)
#         attn_output = attn_output.reshape(B, self.num_heads, N, self.head_dim).transpose(1, 2)
#         attn_output = attn_output.reshape(B, N, C)
#
#         x = self.proj(attn_output)
#         x = self.proj_drop(x)
#         return x


class Attention_LoRA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        self.rank = r

        self.lora_A_k = nn.Linear(dim, r, bias=False)
        self.lora_B_k = nn.Linear(r, dim, bias=False)
        self.lora_A_v = nn.Linear(dim, r, bias=False)
        self.lora_B_v = nn.Linear(r, dim, bias=False)
        self.rank = r

        self.init_param()


    def init_param(self):
        nn.init.kaiming_uniform_(self.lora_A_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v.weight, a=math.sqrt(5))
        self.lora_B_k.weight.assign(jt.zeros(self.lora_B_k.weight.shape))
        self.lora_B_v.weight.assign(jt.zeros(self.lora_B_v.weight.shape))


    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def execute(self, x, register_hook=False, get_feat=False, get_cur_feat=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # insert lora
        weight_k = self.lora_B_k.weight @ self.lora_A_k.weight
        weight_v = self.lora_B_v.weight @ self.lora_A_v.weight
        k = k + nn.linear(x, weight_k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v + nn.linear(x, weight_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



# class Block(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None):
#         super().__init__()
#         self.config = config
#
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
#                               attn_drop=attn_drop, proj_drop=drop)
#
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.fc1 = nn.Linear(dim, mlp_hidden_dim)
#         self.fc2 = nn.Linear(mlp_hidden_dim, dim)
#         self.act = act_layer()
#         self.mlp_drop = nn.Dropout(drop)
#
#         if config and config.ffn_adapt:
#             self.adaptmlp = Adapter(self.config, dropout=0.1, bottleneck=config.ffn_num,
#                                     init_option=config.ffn_adapter_init_option,
#                                     adapter_scalar=config.ffn_adapter_scalar,
#                                     adapter_layernorm_option=config.ffn_adapter_layernorm_option)
#
#     def execute(self, x):
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         if self.config and self.config.ffn_adapt and self.config.ffn_option == 'parallel':
#             adapt_x = self.adaptmlp(x, add_residual=False)
#
#         residual = x
#         x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
#         x = self.drop_path(self.mlp_drop(self.fc2(x)))
#
#         if self.config and self.config.ffn_adapt:
#             if self.config.ffn_option == 'sequential':
#                 x = self.adaptmlp(x)
#             elif self.config.ffn_option == 'parallel':
#                 x = x + adapt_x
#             else:
#                 raise ValueError(self.config.ffn_adapt)
#
#         x = residual + x
#         return x
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, n_tasks=10, r=64):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_LoRA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, r=r)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)


    def execute(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))

        residual = x
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))

        x = residual + x
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()

        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)

        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.strict_img_size = True    # timm.models.layers.PatchEmbed
        self.dynamic_img_pad = False   # timm.models.layers.PatchEmbed

        self.proj = nn.Conv(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def execute(self, x):
        """
        x: [B, C, H, W]
        return: [B, N, embed_dim]
        """
        # print("input shape:", x.shape)
        x = self.proj(x)  # B, embed_dim, H/P, W/P
        B, D, Hp, Wp = x.shape

        x = x.view(B, D, Hp * Wp)  # B, D, N
        x = x.permute(0, 2, 1)  # B, N, D
        x = self.norm(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 representation_size=None, distilled=False, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None, act_layer=None,
                 weight_init='', tuning_config=None):
        super().__init__()

        self.tuning_config = tuning_config
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        # embed_layer = embed_layer or PatchEmbed

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size,
                                       in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(jt.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(jt.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(jt.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [float(x) for x in jt.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values = None, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                n_tasks = 10, r = 10)
            for i in range(depth)
        ])

        if not global_pool:
            self.norm = norm_layer(embed_dim)
        else:
            self.fc_norm = norm_layer(embed_dim)

        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.global_pool = global_pool

        if tuning_config and tuning_config.vpt_on:
            assert tuning_config.vpt_num > 0
            self.embeddings = nn.ParameterList([
                nn.Parameter(jt.empty(1, tuning_config.vpt_num, embed_dim)) for _ in range(depth)
            ])
            for e in self.embeddings:
                nn.init.xavier_uniform_(e)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        return (self.head, self.head_dist) if self.dist_token is not None else self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        # x = x.permute(0, 3, 1, 2)
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = jt.concat([cls_tokens, x], dim=1)

        if self.dist_token is not None:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = jt.concat([cls_tokens, dist_tokens, x[:, 1:, :]], dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            if self.tuning_config and self.tuning_config.vpt_on:
                eee = self.embeddings[i].expand(B, -1, -1)
                x = jt.concat([eee, x], dim=1)
            x = blk(x)
            if self.tuning_config and self.tuning_config.vpt_on:
                x = x[:, self.tuning_config.vpt_num:, :]

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            x = self.fc_norm(x)
        else:
            x = self.norm(x)
            x = x[:, 0]
        return x

    def execute(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x_cls, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.is_training():
                return x_cls, x_dist
            else:
                return (x_cls + x_dist) / 2
        else:
            return self.head(x)


def vit_base_patch16_224_in21k_lora(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    model.load('vit_base_patch16_224_in21k_lora.pt')

    return model


def vit_base_patch16_224_lora(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    model.load('vit_base_patch16_224_lora.pt')

    return model


if __name__ == '__main__':
    jt.flags.use_cuda = 1  # 如果有 GPU 会自动跑 CUDA

    from easydict import EasyDict

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
    model.out_dim = 768

    # model = Block(dim=768, num_heads=12, config=Config())
    x = jt.randn(2, 3, 224, 224)
    y = model(x)
    print("output shape:", y.shape)


    # B, N, C = 4, 196, 768  # batch, sequence_len, embedding_dim
    # x = jt.randn((B, N, C))
    #
    # # 实例化模型
    # model = Attention(dim=768, num_heads=12)
    #
    # # 前向计算
    # out = model(x)
    #
    # print("输入 shape:", x.shape)
    # print("输出 shape:", out.shape)

    # learning_rate = 0.1
    # momentum = 0.9
    # weight_decay = 1e-4
    #
    # optimizer = nn.SGD(model.parameters(), learning_rate, momentum, weight_decay)
    # loss = jt.sqr(out - x)
    # optimizer.step(loss)
    # # # 测试 loss & backward
    # # loss = out.sum()
    # # loss.backward()    需要替换如上
    # print("loss:", loss)
    # #
    # # # 检查梯度
    # # print("grad proj:", model.proj.weight.grad.mean())   需要替换如下
    # print("grad proj:", model.proj.weight.opt_grad(optimizer).mean())
