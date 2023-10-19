# -*- coding: utf-8 -*-
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast


from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint

from collections import OrderedDict
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT
import warnings
import torch.utils.checkpoint as checkpoint
_tokenizer = _Tokenizer()
torch.backends.cudnn.enabled = False

# Prompt Ensembling
IMAGENET_TEMPLATES = [
    "x itap of a",
    "a bad photo of",
    "x x a origami",
    "a photo of large",
    "in video game a",
    "x art of the",
    "a photo of small",
]

# IMAGENET_TEMPLATES = [
#     "a flower photo of",
#     "the flower is named",
#     "x this flower is",
#     "What's the flower ?"
# ]

# IMAGENET_TEMPLATES = [
#     "a bad photo of",
#     "an aircraft photo of",
#     "a photo of large",
#     "x photo of Boeing",
#     "the aircraft's name is",
#     "a photo of small",
# ]


# IMAGENET_TEMPLATES = [
#     "a car photo of",
#     "the car name is",
# ]


# IMAGENET_TEMPLATES = [
#     "an aircraft photo of",
#     "the aircraft's name is",
#     "aircraft photo of Boeing",
# ]

# IMAGENET_TEMPLATES = [
#     "aircraft photo of Boeing",
#     "aircraft photo of Airbus",
# ]

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, osp.expanduser("/userhome/CLIP/models"))

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


def load_checkpoint(fpath):
    r"""Load checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict

    Examples::
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError("File path is None")

    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))

    map_location = None if torch.cuda.is_available() else "cpu"

    try:
        checkpoint = torch.load(fpath, map_location=map_location)

    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )

    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise

    return checkpoint

def load_pretrained_weights(model, weight_path):
    r"""Load pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> load_pretrained_weights(model, weight_path)
    """
    checkpoint = load_checkpoint(weight_path)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            "please check the key names manually "
            "(** ignored and continue **)".format(weight_path)
        )
    else:
        print(
            'Successfully loaded pretrained weights from "{}"'.
            format(weight_path)
        )
        if len(discarded_layers) > 0:
            print(
                "** The following layers are discarded "
                "due to unmatched keys or layer size: {}".
                format(discarded_layers)
            )


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        
    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward
    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
#         print(x.shape)

        x = checkpoint.checkpoint(self.custom(self.transformer), x)

        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1), :] @ self.text_projection

        return x


from torchvision import utils as vutils

def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为图片
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # print('input_tensor.shape', input_tensor.shape)
    # 反归一化
    mean = [0.48145466, 0.4578275, 0.40821073]  # CLIP
    std = [0.26862954, 0.26130258, 0.27577711]  # CLIP
    dtype = input_tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=input_tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=input_tensor.device)
    input_tensor.mul_(std[None, :, None, None]).add_(mean[None, :, None, None])
    # print(input_tensor)
    # input_tensor = input_tensor * 255
    # print(input_tensor)
    vutils.save_image(input_tensor, filename)
    # from PIL import Image
    # im = Image.fromarray(ndarr)
    # im.save(fp


class Weight_Adapter(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        dtype = clip_model.dtype
        if 'T' in cfg.visual_pre:
            if 'ViT' in cfg.backbone:
                emb_dim = int(clip_model.visual.proj.shape[0])
            else:
                emb_dim = int(clip_model.visual.attnpool.c_proj.weight.shape[1])
        else:
            emb_dim = clip_model.visual.output_dim
            
        print('output_dim:',clip_model.visual.output_dim)
        print('emb_dim:',emb_dim)
        if cfg.adapter_dim == '_dim_x4':
            hidden_dim = emb_dim * 4
            print('emb_dim',emb_dim)
        elif cfg.adapter_dim == '_dim_x8':
            hidden_dim = emb_dim * 8
        elif cfg.adapter_dim == '_dim_x2':
            hidden_dim = emb_dim * 2
            print('emb_dim',emb_dim)
        elif cfg.adapter_dim == '_dim_x1':
            hidden_dim = emb_dim * 1
        elif cfg.adapter_dim == '_dim_x1.77':
            hidden_dim = int(emb_dim * 1.77)
        elif cfg.adapter_dim == '_dim_ch8':
            hidden_dim = emb_dim // 8
        elif cfg.adapter_dim == '_dim_ch4':
            hidden_dim = emb_dim // 4
        elif cfg.adapter_dim == '_dim_ch16':
            hidden_dim = emb_dim // 16
        else:
            hidden_dim = emb_dim // 2

        self.linear1 = nn.Linear(emb_dim, hidden_dim, bias=False).to(dtype)
        self.linear2 = nn.Linear(hidden_dim, emb_dim, bias=False).to(dtype)
        self.relu = nn.ReLU(inplace=True).to(dtype)
        self.alpha = nn.Parameter(torch.HalfTensor([0.5])).to(dtype)


    def forward(self, x):
        # print('x.shape', x.shape)
        out = self.relu(self.linear1(x))
        out = self.linear2(out)
        out = self.alpha * out + (torch.HalfTensor([1.0]).cuda() - self.alpha) * x
        out = self.relu(out)
        # print('alpha: ', self.alpha)
#         print(self.linear1.weight)
#         print(self.linear1.bias)
#         print(self.linear2.weight)
#         print(self.linear2.bias)

        return out


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.n_ctx
        ctx_init = cfg.ctx_init
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.input_size
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            if ctx_init=='ensemble':
                templates = IMAGENET_TEMPLATES
                num_temp = len(templates)
                print(f"Prompt ensembling (n={num_temp})")
                n_ctx = cfg.n_ctx #len(max(templates, key=len, default='').split(" "))
                print('n_ctx', n_ctx)
                mean_embedding = 0
                for i, temp in enumerate(templates):
                    prompt = temp.replace("_", " ")
                    prompts = clip.tokenize(prompt)
                    with torch.no_grad():
                        embedding = clip_model.token_embedding(prompts).type(dtype)
                    mean_embedding = mean_embedding+embedding

                mean_x_embedding = mean_embedding / num_temp
                ctx_vectors = mean_x_embedding[0, 1: 1 + n_ctx, :]
                prompt_prefix = " ".join(["x"] * n_ctx)

            else:
                ctx_init = ctx_init.replace("_", " ")
                n_ctx = len(ctx_init.split(" "))
                prompt = clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
#                 print('ctx_vectors',ctx_vectors)
                prompt_prefix = ctx_init

        else:
            #random initialization
            if cfg.csc:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        if prompt_prefix =='ensemble':
            templates = IMAGENET_TEMPLATES
            num_temp = len(templates)
            n_ctx = cfg.n_ctx
            # n_ctx = len(max(templates, key=len, default='').split(" "))
            print('n_ctx',n_ctx)
            mean_embedding = 0
            for i, temp in enumerate(templates):
                prompts = [temp + " " + name + "." for name in classnames]
                tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
                with torch.no_grad():
                    embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
                mean_embedding = mean_embedding + embedding
            embedding = mean_embedding / num_temp

        else:
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
#             tokenized_prompts = tokenized_prompts
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)


        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  #[100, len(p)]
        self.name_lens = name_lens
        self.class_token_position = "end"
        self.n_classes = cfg.n_classes
    
    def forward(self, class_ids):
        ctx = self.ctx
        prefix = self.token_prefix
        suffix = self.token_suffix

        tokenized_prompts = self.tokenized_prompts
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
#         if self.class_token_position == "end":
#         print('prefix',prefix)
#         print('suffix',suffix)
        whole_prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
#         if class_ids == None:
        return whole_prompts, tokenized_prompts

#         print(class_ids)
#         episode_tokenized_prompts = torch.cat([self.tokenized_prompts[i, :].view(1,-1) for i in class_ids])
#         # print('ctx.shape', ctx.shape)
#         # print('class_ids', class_ids)

#         if self.class_token_position == "end":
#             prompts = []
#             for i in class_ids:
#                 name_len = self.name_lens[i]
#                 prefix_i = prefix[i: i + 1, :, :]
#                 ctx_i = ctx[i: i + 1, :, :]
#                 class_i = suffix[i: i + 1, :name_len, :]
#                 suffix_i = suffix[i: i + 1, name_len:, :]

#                 prompt = torch.cat(
#                     [
#                         prefix_i,  # (1, 1, dim)
#                         ctx_i,   # (1, n_ctx, dim)
#                         class_i,  # (1, name_len, dim)
#                         suffix_i,  # (1, *, dim)
#                     ],
#                     dim=1,
#                 )
#                 prompts.append(prompt)
#             prompts = torch.cat(prompts, dim=0)

#         elif self.class_token_position == "middle":
#             half_n_ctx = self.n_ctx // 2
#             prompts = []
#             for i in class_ids:
#                 name_len = self.name_lens[i]
#                 prefix_i = prefix[i : i + 1, :, :]
#                 class_i = suffix[i : i + 1, :name_len, :]
#                 suffix_i = suffix[i : i + 1, name_len:, :]
#                 ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
#                 ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
#                 prompt = torch.cat(
#                     [
#                         prefix_i,     # (1, 1, dim)
#                         ctx_i_half1,  # (1, n_ctx//2, dim)
#                         class_i,      # (1, name_len, dim)
#                         ctx_i_half2,  # (1, n_ctx//2, dim)
#                         suffix_i,     # (1, *, dim)
#                     ],
#                     dim=1,
#                 )
#                 prompts.append(prompt)
#             prompts = torch.cat(prompts, dim=0)

#         elif self.class_token_position == "front":
#             prompts = []
#             for i in class_ids:
#                 name_len = self.name_lens[i]
#                 prefix_i = prefix[i : i + 1, :, :]
#                 class_i = suffix[i : i + 1, :name_len, :]
#                 suffix_i = suffix[i : i + 1, name_len:, :]
#                 ctx_i = ctx[i : i + 1, :, :]
#                 prompt = torch.cat(
#                     [
#                         prefix_i,  # (1, 1, dim)
#                         class_i,   # (1, name_len, dim)
#                         ctx_i,     # (1, n_ctx, dim)
#                         suffix_i,  # (1, *, dim)
#                     ],
#                     dim=1,
#                 )
#                 prompts.append(prompt)
#             prompts = torch.cat(prompts, dim=0)

#         else:
#             raise ValueError

#         episode_prompts = prompts


#         return whole_prompts, tokenized_prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.n_cls = len(classnames)
        self.adapter = Weight_Adapter(cfg, clip_model)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.para_update = cfg.para_update
        self.visual_pre = cfg.visual_pre
        self.proj = cfg.proj
        self.backbone = cfg.backbone
        # width = int(self.image_encoder.proj.shape[0])
        # scale = width ** -0.5
#         self.map = self.image_encoder.proj
    
    def forward(self, image, class_ids=None, phase='train'):
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))
#             print('image.type(self.dtype)',image.type(self.dtype))

        new_image_features = None
        new_image_features_proj =None
        
        if self.para_update == 'v_adapter' or self.para_update == 'prompt+v_adapter':
            if 'T' in self.visual_pre:
                if 'ViT' in self.backbone:
                    new_image_features = image_features
                    new_image_features = self.adapter(new_image_features.cuda())
                else:
#                     with torch.no_grad():
#                         new_image_features = ((image_features.float()-self.image_encoder.attnpool.c_proj.bias.float())@torch.pinverse(self.image_encoder.attnpool.c_proj.weight.t().float())).half()
                    new_image_features = self.adapter(image_features.cuda())
#                 if self.proj == 'projup':
#                     new_image_features_proj = new_image_features @ self.map

            else:
                new_image_features = self.adapter(image_features)
            new_image_features = new_image_features / new_image_features.norm(dim=-1, keepdim=True)
        elif self.para_update == 'prompt':
            if 'T' in self.visual_pre:
                if 'ViT' in self.backbone:
                    new_image_features = (image_features.float()@torch.pinverse(self.image_encoder.proj.float())).half()
                else:
                    with torch.no_grad():
                        new_image_features = ((image_features.float()-self.image_encoder.attnpool.c_proj.bias.float())@torch.pinverse(self.image_encoder.attnpool.c_proj.weight.t().float())).half()
                new_image_features = new_image_features / new_image_features.norm(dim=-1, keepdim=True)
            else:
                new_image_features = None

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
#         print('class_ids', class_ids)

        if class_ids == None:
            whole_prompts, whole_tokenized_prompts = self.prompt_learner(class_ids)
            text_features = None #self.text_encoder(whole_prompts, whole_tokenized_prompts)
            prompt_features = None #text_features / text_features.norm(dim=-1, keepdim=True)
            return image_features, new_image_features, logit_scale, prompt_features

        # for i in range(len(class_ids)):
        #     save_image_tensor(image[i,:,:,:].unsqueeze(dim=0), '/hdd/pengf/Proto_prompt/experiments/tieredimages/{}.jpg'.format(i))
        if phase == 'support_train':
            text_ids = list(range(self.n_cls))
        else:
            text_ids = []
            for key in sorted(class_ids.keys()):
                text_ids.append(class_ids[key])
        
        whole_prompts, whole_tokenized_prompts = self.prompt_learner(text_ids)
        # text_features = self.text_encoder(prompts, tokenized_prompts)
        # if self.para_update == 't_adapter':
        #     text_features = self.adapter(text_features)

        episode_text_features = None
        prompt_features = None
        # logits = logit_scale * image_features @ text_features.t() #[batch, 1, dim]@[batch, n_cls, dim].t()
        if 'train' in phase:
            prompt_features = None
            
#             print('prompts, tokenized_prompts.shape', prompts.shape, tokenized_prompts.shape)
#             print(tokenized_prompts.argmax(dim=-1))
#             print('whole_prompts, whole_tokenized_prompts.shape', whole_prompts.shape, whole_tokenized_prompts.shape)
#             print(whole_tokenized_prompts.argmax(dim=-1))
            text_features = self.text_encoder(whole_prompts, whole_tokenized_prompts)
           
            episode_text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        elif phase == 'prototype':
            episode_text_features = None
            text_features = self.text_encoder(whole_prompts, whole_tokenized_prompts)
            prompt_features = text_features / text_features.norm(dim=-1, keepdim=True)


        return image_features, new_image_features, episode_text_features, logit_scale, prompt_features


