# -*- coding: utf-8 -*-
import os
import argparse
import random
import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
# from focalloss import FocalLoss
from models.resnet12_2 import resnet12
from models.coop_4 import CustomCLIP, TextEncoder, PromptLearner, save_image_tensor, load_pretrained_weights
from models.meta_part_inference_mini import ProtoComNet
from models.PredTrainHead import LinearClassifier, LinearRotateHead
from clip import clip
from clip.model import convert_weights
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from utils import set_gpu, Timer, count_accuracy, count_true, check_dir, log, baysian, load_checkpoint, ConstantWarmupScheduler
import pickle
import math
# base_datadir = '/media/data1/huangbin/'
# base_dir = '/home/huangbin/workspace/pengf/Proto_prompt/'
# base_datadir = '/hdd/pengf/'
# base_dir = '/hdd/pengf/Proto_prompt/'
# base_datadir = '/raid2/wangjin/fewshot/'
# base_dir = '/raid2/wangjin/fewshot/Proto_prompt/'
base_datadir = '/userhome/'
base_dir = '/userhome/fewshot/Proto_prompt/'
weight=10
categories=0
torch.backends.cudnn.enabled = False
# from gpu_mem_track import MemTracker
# gpu_tracker = MemTracker()
def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """
    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    return encoded_indicies

# def get_unique_N(iterable, N):
#     """Yields (in order) the first N unique elements of iterable.
#     Might yield less if data too short."""
#     seen = []
#     seen = []
#     ids= []
#     while True:
#         k = random.randint(0, len(iterable)-1)
#         if iterable[k] in seen:
#             continue
#         seen.append(iterable[k])
#         ids.append(k)
#         if len(seen) == N:
#             return ids, seen

def get_unique_N(iterable, N):
    """Yields (in order) the first N unique elements of iterable.
    Might yield less if data too short."""
    seen = []
    ids= []
    i=0
    while True:
        k = random.randint(0, len(iterable)-1)
        if iterable[k] in seen[i*N: (i+1)*N] or iterable[k] == -1:
            continue
        seen.append(iterable[k])
        ids.append(k)
        iterable[k] = -1
        if len(seen) == len(iterable):
            return ids, seen
        elif len(seen[i*N: (i+1)*N]) == N:
            i=i+1

def load_clip_to_cpu(opt):
    backbone_name = opt.backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, os.path.expanduser(base_datadir + "CLIP/models"))
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()  

def get_model(opt):
    # Choose the embedding network
    if opt.network == 'CLIP':
        clip_model = load_clip_to_cpu(opt)
        classnames = []
        if opt.dataset == 'miniImageNet':
            class_file = "./miniimagenet_classes.txt"
        elif opt.dataset == 'tieredImageNet':
            class_file = base_datadir + "/CLIP/data/tiered-imagenet/class_names.txt"
        elif opt.dataset == 'ImageNet':
            class_file = base_datadir + "/CLIP/data/imagenet/train_classnames.txt"
        elif opt.dataset == 'Food101':
            class_file = base_datadir + "/CLIP/data/food-101/class_names.txt"
        elif opt.dataset == 'SUN397':
            class_file = base_datadir + "/CLIP/data/sun397/class_names.txt"
        elif opt.dataset == 'DTD':
            class_file = base_datadir + "/CLIP/data/dtd/class_names.txt"
        elif opt.dataset == 'Flowers102':
            class_file = base_datadir + "/CLIP/data/oxford_flowers/class_names.txt"
        elif opt.dataset == 'StanfordCars':
            class_file = base_datadir + "/CLIP/data/stanford_cars/class_names.txt"
        elif opt.dataset == 'FGVCAircraft':
            class_file = base_datadir + "/CLIP/data/fgvc_aircraft/class_names.txt"
        elif opt.dataset == 'OxfordPets':
            class_file = base_datadir + "/CLIP/data/oxford_pets/class_names.txt"
        elif opt.dataset == 'EuroSAT':
            class_file = base_datadir + "/CLIP/data/eurosat/class_names.txt"
        elif opt.dataset == 'UCF101':
            class_file = base_datadir + "/CLIP/data/ucf101/class_names.txt"
        elif opt.dataset == 'Caltech101':
            class_file = base_datadir + "/CLIP/data/caltech101/class_names.txt"
        with open(class_file, "r") as f:
            for line in f.readlines():
                line = line.strip('\n') # 去掉列表中每一个元素的换行符
                # print(line)
                classnames.append(line)

        categories = len(classnames)

        if opt.prec == "fp32" or opt.prec == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        model = CustomCLIP(opt, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in model.named_parameters():
#             print(name)
#             print(param.shape)
            if opt.para_update == 'prompt':
                if "prompt_learner" not in name:
                    param.requires_grad_(False)
                else:
                    print(name)
            elif opt.para_update == 'v_adapter' or opt.para_update == 't_adapter':
                if "adapter" not in name:
                    param.requires_grad_(False)
                else:
                    print(name)
                    if 'alpha' in name:
                        print('alpha: ', param)
            elif opt.para_update == 'prompt+v_adapter':
                if "adapter" not in name and "prompt_learner" not in name:
                    param.requires_grad_(False)
                else:
                    print(name)
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name,param.requires_grad)
        network = model.cuda()
    else:
        print ("Cannot recognize the network type")
        assert(False)

#     coop_model_path = os.path.join('/userhome/CoOp/output/{}/CoOp/{}_ep50_ctxv1_{}shots/nctx{}_cscFalse_ctp{}/seed{}'.format(
#         opt.dataset.lower(), opt.backbone.replace('-B/', '').lower(), opt.train_shot, opt.n_ctx,
#         opt.class_positions, opt.seed), 'prompt_learner', 'model.pth.tar-50')
#     checkpoint = load_checkpoint(coop_model_path)
#     state_dict = checkpoint["state_dict"]
#     epoch = checkpoint["epoch"]
#     # Ignore fixed token vectors
#     if "token_prefix" in state_dict:
#         del state_dict["token_prefix"]
#     if "token_suffix" in state_dict:
#         del state_dict["token_suffix"]
#     print("Loading weights to {} " 'from "{}" (epoch = {})'.format('prompt_learner', coop_model_path, epoch))
#     # set strict=False
#     network.prompt_learner.load_state_dict(state_dict, strict=False)
        
    if opt.phase == 'pretrain':
        from models.classification_heads_orgin import ClassificationHead
    else:
        from models.classification_heads_4 import ClassificationHead

    # Choose the classification head
    if opt.head == 'Vision':
        cls_head = ClassificationHead(base_learner='VisionCLS').cuda()
    elif opt.head == 'Vision-Text':
        cls_head = ClassificationHead(base_learner='Vision-TextCLS').cuda()
    elif opt.head == 'No-prototype':
        cls_head = ClassificationHead(base_learner='Noprototype').cuda()
    elif opt.head == 'FuseCLS':
        cls_head = ClassificationHead(base_learner='FuseCLS').cuda()
    else:
        print ("Cannot recognize the cls_head type")
        assert(False)

    return (network, cls_head), categories

def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'ImageNet':
        from data.imagenet import FewShotDataloader
        from data.imagenet import ImageNet as DataSpecific

    elif options.dataset == 'Food101':
        from data.food101 import FewShotDataloader
        from data.food101 import Food101 as DataSpecific
    elif options.dataset == 'SUN397':
        from data.sun397 import FewShotDataloader
        from data.sun397 import SUN397 as DataSpecific
    elif options.dataset == 'DTD':
        from data.dtd import FewShotDataloader
        from data.dtd import DescribableTextures as DataSpecific
    elif options.dataset == 'Flowers102':
        from data.oxford_flowers import FewShotDataloader
        from data.oxford_flowers import OxfordFlowers as DataSpecific
    elif options.dataset == 'StanfordCars':
        from data.stanford_cars import FewShotDataloader
        from data.stanford_cars import StanfordCars as DataSpecific
    elif options.dataset == 'FGVCAircraft':
        from data.fgvc_aircraft import FewShotDataloader
        from data.fgvc_aircraft import FGVCAircraft as DataSpecific
    elif options.dataset == 'OxfordPets':
        from data.oxford_pets import FewShotDataloader
        from data.oxford_pets import OxfordPets as DataSpecific
    elif options.dataset == 'EuroSAT':
        from data.eurosat import FewShotDataloader
        from data.eurosat import EuroSAT as DataSpecific
    elif options.dataset == 'UCF101':
        from data.ucf101 import FewShotDataloader
        from data.ucf101 import UCF101 as DataSpecific
    elif options.dataset == 'Caltech101':
        from data.caltech101 import FewShotDataloader
        from data.caltech101 import Caltech101 as DataSpecific
    else:
        print ("Cannot recognize the dataset type")
        assert(False)

    if options.network == 'CLIP':
        if options.phase == 'metatrain':
            dataset_train = DataSpecific(phase='train', subsample=options.subsample, num_shots=options.train_shot)
            dataset_train_notfm = DataSpecific(phase='train', subsample=options.subsample, num_shots=options.train_shot, do_not_use_random_transf = True)
            dataset_val = DataSpecific(phase='val', subsample=options.subsample, num_shots=options.train_shot)
            dataset_test = DataSpecific(phase='test', subsample=options.subsample, num_shots=options.train_shot)
        elif options.phase == 'metatest':
            dataset_train = DataSpecific(phase='train', subsample=options.subsample, num_shots=options.train_shot)
            dataset_train_notfm = DataSpecific(phase='train', subsample=options.subsample, num_shots=options.train_shot, do_not_use_random_transf = True)
            dataset_val = DataSpecific(phase='val', subsample=options.subsample, num_shots=options.train_shot)
            dataset_test = DataSpecific(phase='test', subsample=options.subsample, num_shots=options.train_shot)
        elif options.phase == 'tSNE':
            dataset_train = None
            dataset_train_notfm = DataSpecific(phase='train', subsample=options.subsample, num_shots=1000,do_not_use_random_transf = True)
            dataset_val = None
            dataset_test = DataSpecific(phase='test',subsample=options.subsample, num_shots=options.train_shot)

    data_loader = FewShotDataloader


    return (dataset_train, dataset_train_notfm, dataset_val, dataset_test, data_loader)

def seed_torch(seed=21):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_image_prototypes(opt, embedding_net, phase):
    prototypes_shot = os.path.join(opt.save_path, 'prototypes_{}_shot_{}_loss_{}_{}{}_{}{}_{}.pkl'.format(opt.subsample, opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.ctx_init, opt.n_ctx, opt.visual_pre))
 
    if not os.path.exists(prototypes_shot) or phase == 'val':
        emb_supports = []
        new_emb_supports = []
        labels_supports = []
        text_prototypes =None
        
        dloader_prototype = DataLoader(dataset=dataset_train_notfm,  # torch TensorDataset format
                                  batch_size=opt.test_way *opt.train_shot,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=8,
                                  pin_memory=torch.cuda.is_available(),
                                  )

        dloader_prototypeiter = iter(dloader_prototype)
        for k in trange(1, len(dloader_prototypeiter) + 1):
            batch = next(dloader_prototypeiter)
            data_support, real_labels_support = [
                x.cuda() for x in batch]
            train_n_support = data_support.shape[0]
            
            rl_support = real_labels_support.reshape(-1).tolist()

            text_support = {}
            for i in range(len(rl_support)):
                text_support[rl_support[i]] = rl_support[i]

            with torch.no_grad(): 
                if k == len(dloader_prototypeiter):
                    emb_support, new_emb_support,_, logit_scale, support_prompt_features = embedding_net(
                        data_support.reshape([-1] + list(data_support.shape[-3:])), text_support, 'prototype')
                else:
                    emb_support, new_emb_support,_, logit_scale, support_prompt_features = embedding_net(
                        data_support.reshape([-1] + list(data_support.shape[-3:])), text_support, 'val')

                emb_support = emb_support.reshape(1, train_n_support, -1)
                if new_emb_support != None:
                    new_emb_support = new_emb_support.reshape(1, train_n_support, -1)
                    new_emb_supports.append(new_emb_support)
            emb_supports.append(emb_support)
            labels_supports.append(real_labels_support.reshape(-1))
            text_prototypes = support_prompt_features

        emb_supports = torch.cat(emb_supports, dim=0)
        if len(new_emb_supports) != 0:
            new_emb_supports = torch.cat(new_emb_supports, dim=0)
        labels_supports = torch.cat(labels_supports, dim=0)
        text_prototypes = text_prototypes.reshape(1, opt.n_classes, -1)

        support_labels_one_hot = one_hot(labels_supports.view(opt.n_classes * opt.train_shot), opt.n_classes)  # [episodes*k*n, n]

        support_labels_one_hot = support_labels_one_hot.view(1, opt.n_classes * opt.train_shot, opt.n_classes)  # [episodes, k*n , n]
        labels_train_transposed = support_labels_one_hot.transpose(1, 2).half()  # [episodes, n , k*n]
        # print('labels_train_transposed',labels_train_transposed)
        emb_prototypes = torch.bmm(labels_train_transposed, emb_supports.view(1, opt.n_classes*opt.train_shot, -1))
        emb_prototypes = emb_prototypes.div(
            labels_train_transposed.sum(dim=2, keepdim=True).expand_as(emb_prototypes)).half()
        new_emb_prototypes = torch.bmm(labels_train_transposed, new_emb_supports.view(1, opt.n_classes * opt.train_shot, -1))
        new_emb_prototypes = new_emb_prototypes.div(
            labels_train_transposed.sum(dim=2, keepdim=True).expand_as(new_emb_prototypes)).half()
        # print('new_emb_prototypes', new_emb_prototypes)
        if phase == 'test':
            data = {"emb_prototypes": emb_prototypes.cpu().numpy(), "new_emb_prototypes": new_emb_prototypes.cpu().numpy(), 'text_prototypes': text_prototypes.cpu().numpy()}
            print(f"Saving few-shot prototypes to {prototypes_shot}")
            with open(prototypes_shot, "wb") as file:
                pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print(f"Loading {prototypes_shot}")
        with open(prototypes_shot, "rb") as file:
            data = pickle.load(file)
            emb_prototypes = torch.HalfTensor(data['emb_prototypes']).cuda()
            new_emb_prototypes = torch.HalfTensor(data['new_emb_prototypes']).cuda()
            text_prototypes = torch.HalfTensor(data['text_prototypes']).cuda()
    
    return emb_prototypes, new_emb_prototypes, text_prototypes


def meta_train(opt, dataset_train, dataset_val, dataset_test, data_loader):
    # Dataloader of Gidaris & Komodakis (CVPR 2018)
#     if opt.train_shot in [1, 2, 4]:
    epoch_size = opt.episodes_per_batch * opt.n_classes // opt.train_way
#     else:
#         epoch_size = opt.episodes_per_batch * opt.n_classes // opt.train_way * opt.train_shot//4
    dloader_train = data_loader(
        dataset=dataset_train,
        nKnovel=opt.train_way,
        nKbase=0,
        nExemplars=opt.train_shot,  # num training examples per novel category
        nTestNovel=0,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=opt.episodes_per_batch,
        num_workers=4,
        epoch_size=epoch_size,  # num of batches per epoch
    )

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)

    log_file_path = os.path.join(opt.save_path, "train_log_{}_shot{}_loss_{}_metatrain_Proto_{}{}_{}{}{}_{}.txt".format(opt.subsample, opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.ctx_init, opt.n_ctx, opt.lr_decay, opt.visual_pre))
    log(log_file_path, str(vars(opt)))

    (embedding_net, cls_head), categories = get_model(opt)

    # Load saved model checkpoints
    if os.path.exists(os.path.join(opt.save_path, 'best_model_meta_val_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.pth'.format(opt.subsample, opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre))):
        saved_models = torch.load(os.path.join(opt.save_path, 'best_model_meta_val_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.pth'.format(opt.subsample, opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre)), map_location="cpu")
        embedding_net.load_state_dict(saved_models['embedding'])

    if opt.dataset == 'miniImageNet':
        learning_rate = 1e-4
        learning_ratep = 3.5e-5
        if opt.para_update == 'prompt':
            learning_rate=3.5e-5
    elif opt.dataset == 'tieredImageNet':
        learning_rate = 1e-3
        learning_ratep = 3.5e-5
        if opt.para_update == 'prompt':
            learning_rate=3.5e-5
    elif opt.dataset in ['ImageNet', 'Food101', 'SUN397', 'DTD', 'Flowers102','StanfordCars', 'Caltech101', 'UCF101', 'EuroSAT', 'OxfordPets', 'FGVCAircraft']:
        learning_rate = 5e-4
        if opt.train_shot == 1:
            learning_ratep = 5e-5
            if opt.dataset == 'Flowers102':
                #stage2
                learning_rate = 5e-4
                learning_ratep = 1e-4
            elif opt.dataset == 'EuroSAT':
                learning_rate = 7.5e-4
                learning_ratep = 5e-5
            if opt.para_update == 'prompt':
                learning_rate=1e-4
                
        elif opt.train_shot == 2:
            learning_rate = 1.5e-3#2.2e-3 #1.5e-3#2e-3
            learning_ratep = 2e-4#1.3e-4 #2e-3#2.5e-4
#             #stage2
#             learning_rate = 5e-4
#             learning_ratep = 5e-5
            if opt.para_update == 'prompt':
                learning_rate = 2.5e-4
            if opt.dataset == 'FGVCAircraft':
                learning_rate = 1e-3
                learning_ratep = 2.5e-4 
#                 #             #stage2
#                 learning_rate = 5e-4
#                 learning_ratep = 5e-5
            elif opt.dataset == 'Flowers102':
                learning_rate = 1.5e-3
                learning_ratep = 1.6e-3
#                 #stage2
#                 learning_rate = 1e-4
#                 learning_ratep = 1e-6

        elif opt.train_shot == 4:
            learning_rate = 3e-3
            learning_ratep = 4e-4
#             #stage2
#             learning_rate = 5e-4
#             learning_ratep = 5e-5
            if opt.para_update == 'prompt':
                learning_rate = 5e-4
            elif opt.dataset == 'FGVCAircraft':
                learning_rate = 1e-3
                learning_ratep = 3e-4 
            elif opt.dataset == 'Flowers102':
                learning_rate = 2e-3
                learning_ratep = 2e-3
            elif opt.dataset == 'EuroSAT':
                learning_rate = 4e-3
                learning_ratep = 4e-4
#                 #stage2
#                 learning_rate = 1e-4
#                 learning_ratep = 1e-6        
        elif opt.train_shot == 8:
            learning_rate = 1e-3
            learning_ratep = 1e-3
            if opt.dataset in ['Food101', 'DTD', 'StanfordCars', 'OxfordPets', 'UCF101','Caltech101', 'EuroSAT']:
                learning_rate = 1.5e-3
                learning_ratep = 1e-3 #7.5e-4
#                 #2stage
#                 learning_rate = 3e-5
#                 learning_ratep = 7e-6            
            elif opt.dataset =='Flowers102':
                learning_rate = 2e-3
                learning_ratep = 4e-3
            elif opt.dataset == 'FGVCAircraft':
                learning_rate = 1e-3
                learning_ratep = 1.5e-3
            elif opt.dataset =='SUN397':
                learning_rate = 3e-3
                learning_ratep = 1.5e-3
            elif opt.dataset =='ImageNet':
                learning_rate = 2e-2
                learning_ratep = 1e-2
#                 #stage2
#                 learning_rate = 5e-4
#                 learning_ratep = 5e-5
                
        elif opt.train_shot == 16:
            learning_rate = 1e-3
            learning_ratep = 1e-3
            if opt.dataset in ['Food101', 'DTD', 'StanfordCars', 'FGVCAircraft', 'UCF101', 'Caltech101','OxfordPets', 'EuroSAT']:
                learning_rate = 3.5e-3 #3.5e-3 #2e-3
                learning_ratep = 3.5e-3 #3.5e-3 #2e-3
#                 #stage2
#                 learning_rate = 5e-5
#                 learning_ratep = 5e-5
            elif opt.dataset =='Flowers102':
                # stage1
                learning_rate = 3e-3 #3e-3
                learning_ratep = 5e-3 #3e-3
#                 #stage2
#                 learning_rate = 5e-4
#                 learning_ratep = 5e-5
            elif opt.dataset =='SUN397':
                learning_rate = 2.5e-3
                learning_ratep = 3.5e-3
            elif opt.dataset == 'FGVCAircraft':
                learning_rate = 5e-3
                learning_ratep = 5e-3
            elif opt.dataset =='ImageNet':
                learning_rate = 4e-2
                learning_ratep = 5e-2
#                 #stage2
#                 learning_rate = 3e-4
#                 learning_ratep = 3e-6
                
    if 'prompt' in opt.para_update:

        prompt_optimizer = torch.optim.SGD([{'params': embedding_net.prompt_learner.parameters()}, \
                                {'params': cls_head.parameters()}], lr=learning_ratep, momentum=0.9, \
                                weight_decay=5e-4, nesterov=True)
    if 'adapter' in opt.para_update:
        adapter_optimizer = torch.optim.SGD([{'params': embedding_net.adapter.parameters()},
                                 {'params': cls_head.parameters()}], lr=learning_rate, momentum=0.9, \
                                weight_decay=5e-4, nesterov=True)
    optimizer = torch.optim.SGD([{'params': embedding_net.parameters()},
                             {'params': cls_head.parameters()}], lr=learning_rate, momentum=0.9, \
                            weight_decay=5e-4, nesterov=True)


    if opt.lr_decay == 'cosine':
        if 'prompt' in opt.para_update:
            prompt_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(prompt_optimizer, T_0=5, T_mult=2)
#             prompt_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(prompt_optimizer, float(opt.num_epoch))
            if opt.warm_up:            
                prompt_lr_scheduler = ConstantWarmupScheduler(prompt_optimizer, prompt_lr_scheduler, warmup_epoch=1,
                cons_lr=1e-5, last_epoch=opt.num_epoch)
        if 'v_adapter' in opt.para_update:
#             adapter_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(adapter_optimizer, T_0=5, T_mult=2)
            adapter_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adapter_optimizer, float(opt.num_epoch))
            if opt.warm_up:
                adapter_lr_scheduler = ConstantWarmupScheduler(adapter_optimizer, adapter_lr_scheduler, warmup_epoch=1,
                cons_lr=1e-5,last_epoch=opt.num_epoch)
#         lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(opt.num_epoch))
        if opt.warm_up:            
            lr_scheduler = ConstantWarmupScheduler(optimizer, lr_scheduler, warmup_epoch=1,
            cons_lr=1e-5,last_epoch=opt.num_epoch)
    elif opt.lr_decay == 'expo':
        lambda_epoch = lambda e: 0.9**e if e <= 60 else (0.001)
        if 'prompt' in opt.para_update:
            prompt_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(prompt_optimizer, lr_lambda=lambda_epoch,
                                                                    last_epoch=-1)
        if 'v_adapter' in opt.para_update:
            adapter_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(adapter_optimizer, lr_lambda=lambda_epoch,
                                                                     last_epoch=-1)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)
    else:
        lambda_epoch = lambda e: 1.0 if e <= 15 else (0.1 if e <= 30 else 0.01 if e <= 60 else (0.001))
        if 'prompt' in opt.para_update:
            prompt_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(prompt_optimizer, lr_lambda=lambda_epoch, last_epoch=-1)
        if 'v_adapter' in opt.para_update:
            adapter_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(adapter_optimizer, lr_lambda=lambda_epoch, last_epoch=-1)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)

    max_val_acc = 0.0
    max_test_acc = 0.0

    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()

    val_losses_avg_new_i2i, val_losses_avg_new_i2t, val_losses_avg_i2t, val_losses_avg_t2t, val_losses_avg_kl_query, val_losses_avg_kl_support, val_losses_avg_kd = [], [], [], [], [], [], []
    val_losses_avg, val_losses_avg_v, val_losses_avg_t = [], [], []
    val_acces_avg, val_acces_avg_v, val_acces_avg_t = [], [], []

    for epoch in range(0, opt.num_epoch + 1):
        if epoch != 0:
            # Train on the training split
            if opt.para_update == 'prompt' or opt.para_update == 'v_adapter':
                lr_scheduler.step()
                # Fetch the current epoch's learning rate
                epoch_learning_rate = 0.1
                for param_group in optimizer.param_groups:
                    epoch_learning_rate = param_group['lr']

            elif opt.para_update == 'prompt+v_adapter' and 'apart_loss' in opt.loss:
                prompt_lr_scheduler.step()
                adapter_lr_scheduler.step()
                # Fetch the current epoch's learning rate
                epoch_learning_rate = 0.1
                for param_group in prompt_optimizer.param_groups:
                    epoch_learning_rate = param_group['lr']
                for param_group in adapter_optimizer.param_groups:
                    epoch_learning_rate = param_group['lr']
            else:
                lr_scheduler.step()
                # Fetch the current epoch's learning rate
                epoch_learning_rate = 0.1
                for param_group in optimizer.param_groups:
                    epoch_learning_rate = param_group['lr']
                    
#             _, _ = [x.eval() for x in (embedding_net, cls_head)]

            log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.6f}'.format(
                epoch, epoch_learning_rate))

            xx, _ = [x.train() for x in (embedding_net, cls_head)]
            xx.apply(fix_bn)

            train_accuracies = []
            train_losses = []
            if 'ViT' in opt.backbone:
                v_proj = embedding_net.image_encoder.proj
                v_bias = None
            else:
                v_proj = embedding_net.image_encoder.attnpool.c_proj.weight
                v_bias = embedding_net.image_encoder.attnpool.c_proj.bias

            for i, batch in enumerate(tqdm(dloader_train(epoch)), 0):
                # data_support: a tensor of shape [episodes, n*k, Height, Width, 3]
                # labels_support: a tensor of shape [episodes, n*k] with the category label id
                data_support, labels_support, real_labels_support, \
                data_query, labels_query, real_labels_query, k_all, _ = [
                    x.cuda() for x in batch]

                rl_support = real_labels_support.reshape(-1).tolist()
                rl_query = real_labels_query.reshape(-1).tolist()
                fl_support = labels_support.reshape(-1).tolist()
                fl_query = labels_query.reshape(-1).tolist()
                text_support = {}
                text_query = {}
                for i in range(len(rl_support)):
                    text_support[fl_support[i]]= rl_support[i]
                for i in range(len(rl_query)):
                    text_query[fl_query[i]]= rl_query[i]

                query_unique_id, labels_query_unique_1 = get_unique_N(rl_query, opt.train_way)
                labels_query_unique = torch.tensor(labels_query_unique_1).view(opt.train_shot, opt.train_way).cuda()
#                 print('labels_query_unique',labels_query_unique)
                query_unique_id = torch.tensor(query_unique_id).cuda()

                train_n_support = opt.train_way * opt.train_shot
                train_n_query = opt.train_way * opt.train_shot

                emb_support, new_emb_support, textemb_support, logit_scale, _ = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])), text_support, 'support_train')

                emb_support = emb_support.reshape(opt.episodes_per_batch, train_n_support, -1)
                if new_emb_support != None:
                    new_emb_support = new_emb_support.reshape(opt.episodes_per_batch, train_n_support, -1)
                textemb_support = textemb_support.reshape(1, opt.n_classes, -1)

                emb_query, new_emb_query, logit_scale, _ = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])), None, 'query_train')
                emb_query = emb_query.reshape(opt.episodes_per_batch, train_n_query, -1)
                if new_emb_query != None:
                    new_emb_query = new_emb_query.reshape(opt.episodes_per_batch, train_n_query, -1)
                textemb_query = textemb_support #textemb_query.reshape(1, opt.train_way, -1)

                logits_query_img2text, logits_query_text2img, logits_query_img2img, \
                logits_query_img2fuse, logits_query_text2text, logits_query_new_img2img, logits_query_new_img2text, logits_query_old_img2img, logits_query_img2text_unique, logits_query_img2text_short = cls_head(k_all, emb_query, new_emb_query, emb_support, new_emb_support, None, None, textemb_support, textemb_query, labels_support, logit_scale, v_proj,v_bias, opt.train_way, opt.train_shot, opt.n_classes, query_unique_id, real_labels_support) # opt.train_shot, opt.n_classes, is_scale=False)

                #Prompt loss
                if opt.loss == 'i2t':
                    loss = F.cross_entropy(logits_query_img2text_unique.reshape(-1, opt.n_classes), labels_query_unique.reshape(-1))

                elif opt.loss == 't2i':
                    loss = F.cross_entropy(logits_query_text2img.reshape(-1, opt.train_way), labels_query.reshape(-1))

                elif opt.loss == 'fuse2':
                    loss = F.cross_entropy(logits_query_img2text.reshape(-1, opt.train_way), labels_query.reshape(-1)) \
                        + F.cross_entropy(logits_query_text2img.reshape(-1, opt.train_way), labels_query.reshape(-1))

                elif opt.loss == 't2t':
                    loss = F.cross_entropy(logits_query_text2text.reshape(-1, opt.train_way), labels_query.reshape(-1))

                elif opt.loss == 'i2t+t2t':
                    loss = F.cross_entropy(logits_query_img2text.reshape(-1, opt.train_way), labels_query.reshape(-1)) + \
                           F.cross_entropy(logits_query_text2text.reshape(-1, opt.train_way), labels_query.reshape(-1))

                elif opt.loss == 'fuse2+t2t':
                    loss = F.cross_entropy(logits_query_img2text.reshape(-1, opt.train_way), labels_query.reshape(-1)) \
                           + F.cross_entropy(logits_query_text2img.reshape(-1, opt.train_way), labels_query.reshape(-1)) \
                           + F.cross_entropy(logits_query_text2text.reshape(-1, opt.train_way), labels_query.reshape(-1))

                # Adapter loss
                elif opt.loss == 'i2i':
                    loss = F.cross_entropy(logits_query_new_img2img.reshape(-1, opt.train_way), labels_query.reshape(-1))
                elif opt.loss == 'i2t+i2i':
                    loss = F.cross_entropy(logits_query_new_img2img.reshape(-1, opt.train_way), labels_query.reshape(-1)) \
                           + F.cross_entropy(logits_query_new_img2text.reshape(-1, opt.train_way), labels_query.reshape(-1))
                elif opt.loss == 'fuse2+i2i':
                    loss = F.cross_entropy(logits_query_new_img2text.reshape(-1, opt.train_way), labels_query.reshape(-1)) \
                           + F.cross_entropy(logits_query_text2img.reshape(-1, opt.train_way), labels_query.reshape(-1)) \
                           + F.cross_entropy(logits_query_new_img2img.reshape(-1, opt.train_way), labels_query.reshape(-1))
                elif opt.loss == 'fuse2+i2i+t2t':
                    loss_i2t = F.cross_entropy(logits_query_new_img2text.reshape(-1, opt.train_way), labels_query.reshape(-1))
                    loss_t2i = F.cross_entropy(logits_query_text2img.reshape(-1, opt.train_way), labels_query.reshape(-1))
                    loss_i2i = F.cross_entropy(logits_query_new_img2img.reshape(-1, opt.train_way), labels_query.reshape(-1))
                    loss_t2t = F.cross_entropy(logits_query_text2text.reshape(-1, opt.train_way), labels_query.reshape(-1))

                    # print('loss_i2t',loss_i2t)
                    # print('loss_t2i',loss_t2i)
                    # print('loss_i2i',loss_i2i)
                    # print('loss_t2t',loss_t2t)
                    loss = loss_i2t + loss_t2i + loss_i2i + loss_t2t
                elif opt.loss == 't2i+i2i':
                    loss = F.cross_entropy(logits_query_text2img.reshape(-1, opt.train_way), labels_query.reshape(-1)) \
                           + F.cross_entropy(logits_query_new_img2img.reshape(-1, opt.train_way), labels_query.reshape(-1))
                elif opt.loss == 'i2t+i2i+kl':
                    loss_kl_query =  F.kl_div(new_emb_query.softmax(dim=-1).log(), emb_query.softmax(dim=-1), reduction='sum')
                    loss_kl_support =  F.kl_div(new_emb_support.softmax(dim=-1).log(), emb_support.softmax(dim=-1),
                               reduction='sum')
                    # print('loss_kl_query = ', loss_kl_query)
                    # print('loss_kl_support = ', loss_kl_support)

                    loss = F.cross_entropy(logits_query_new_img2img.reshape(-1, opt.train_way),
                                           labels_query.reshape(-1)) \
                           + F.cross_entropy(logits_query_new_img2text.reshape(-1, opt.train_way),
                                             labels_query.reshape(-1)) \
                           + loss_kl_query + loss_kl_support

                elif opt.loss == 'apart_loss':
                    loss = F.cross_entropy(logits_query_new_img2img.reshape(-1, opt.train_way), labels_query.reshape(-1)) \
                           + F.cross_entropy(logits_query_new_img2text.reshape(-1, opt.train_way), labels_query.reshape(-1)) \
                           + F.cross_entropy(logits_query_img2text.reshape(-1, opt.train_way), labels_query.reshape(-1)) \
                           + F.cross_entropy(logits_query_text2text.reshape(-1, opt.train_way), labels_query.reshape(-1))

                elif opt.loss == 'apart_loss_i2i':
                    loss = F.cross_entropy(logits_query_new_img2img.reshape(-1, opt.train_way),
                                            labels_query.reshape(-1)) + F.cross_entropy(logits_query_img2text.reshape(-1, opt.train_way),
                                              labels_query.reshape(-1)) + F.cross_entropy(logits_query_text2text.reshape(-1, opt.train_way),
                                              labels_query.reshape(-1))
                elif opt.loss == 'apart_loss_kl':
                    loss = F.cross_entropy(logits_query_new_img2img.reshape(-1, opt.train_way),
                                           labels_query.reshape(-1)) \
                           + F.cross_entropy(logits_query_new_img2text.reshape(-1, opt.train_way),
                                             labels_query.reshape(-1)) \
                           + F.cross_entropy(logits_query_img2text.reshape(-1, opt.train_way), labels_query.reshape(-1)) \
                           + F.cross_entropy(logits_query_text2text.reshape(-1, opt.train_way),
                                             labels_query.reshape(-1)) \
                           +  F.kl_div(new_emb_query.softmax(dim=-1).log(), emb_query.softmax(dim=-1), reduction='sum') \
                           +  F.kl_div(new_emb_support.softmax(dim=-1).log(), emb_support.softmax(dim=-1), reduction='sum')
                elif opt.loss == 'apart_loss_kl_oldi2i':
                    loss = F.cross_entropy(logits_query_new_img2img.reshape(-1, opt.train_way),
                                           labels_query.reshape(-1)) \
                           + F.cross_entropy(logits_query_new_img2text.reshape(-1, opt.train_way),
                                             labels_query.reshape(-1)) \
                           + F.cross_entropy(logits_query_img2text.reshape(-1, opt.train_way), labels_query.reshape(-1)) \
                           + F.cross_entropy(logits_query_text2text.reshape(-1, opt.train_way),
                                             labels_query.reshape(-1)) \
                           + F.cross_entropy(logits_query_old_img2img.reshape(-1, opt.train_way),
                                             labels_query.reshape(-1)) \
                           +  F.kl_div(new_emb_query.softmax(dim=-1).log(), emb_query.softmax(dim=-1), reduction='sum') \
                           +  F.kl_div(new_emb_support.softmax(dim=-1).log(), emb_support.softmax(dim=-1),
                                      reduction='sum')
                elif opt.loss == 'newi2i+i2t':
                    loss = F.cross_entropy(logits_query_new_img2img.reshape(-1, opt.train_way),
                                    labels_query.reshape(-1)) + F.cross_entropy(
                        logits_query_img2text_unique.reshape(-1, opt.n_classes),
                        labels_query_unique.reshape(-1))
                elif opt.loss == 'newi2i+newi2t+i2t':
                    loss = F.cross_entropy(logits_query_new_img2img.reshape(-1, opt.train_way),labels_query.reshape(-1)) \
                           + F.cross_entropy(logits_query_new_img2text.reshape(-1, opt.train_way), labels_query.reshape(-1)) \
                           + F.cross_entropy(logits_query_img2text.reshape(-1, opt.train_way),
                        labels_query.reshape(-1))
                elif opt.loss == 'newi2i+kd+i2t':
                    temperature = opt.temp
                    if not opt.fl:
                        loss = F.cross_entropy(logits_query_new_img2img.reshape(-1, opt.train_way), labels_query.reshape(-1)) \
                               -(F.softmax(logits_query_img2text.reshape(-1, opt.n_classes) / temperature, dim=-1) * F.log_softmax(logits_query_new_img2text.reshape(-1, opt.n_classes) / temperature, dim=-1)).sum(dim=-1).mean() \
                               + F.cross_entropy(logits_query_img2text_unique.reshape(-1, opt.n_classes), labels_query_unique.reshape(-1))
                    else:
                        fl= FocalLoss(gamma=2)
                        loss = fl(logits_query_new_img2img.reshape(-1, opt.train_way), labels_query.reshape(-1)) \
                               -(F.softmax(logits_query_img2text.reshape(-1, opt.n_classes) / temperature, dim=-1) * F.log_softmax(logits_query_new_img2text.reshape(-1, opt.n_classes) / temperature, dim=-1)).sum(dim=-1).mean() \
                               + fl(logits_query_img2text_unique.reshape(-1, opt.n_classes), labels_query_unique.reshape(-1))

                elif opt.loss == 'newi2i+kd2+i2t':
                    temperature = opt.temp
#                     print('logits_query_img2text_short',logits_query_img2text_short.shape)
#                     print('logits_query_new_img2img',logits_query_new_img2img.shape)
                    loss = F.cross_entropy(logits_query_new_img2img.reshape(-1, opt.train_way),
                                           labels_query.reshape(-1)) \
                           - (F.softmax(logits_query_img2text_short.reshape(-1, opt.train_way) / temperature,
                                        dim=-1) * F.log_softmax(
                        logits_query_new_img2img.reshape(-1,  opt.train_way) / temperature, dim=-1)).sum(dim=-1).mean() \
                           + F.cross_entropy(logits_query_img2text.reshape(-1, opt.n_classes), labels_query_unique.reshape(-1))

                elif opt.loss == 'newi2i+klkd+i2t':
                    temperature = opt.temp
#                     print('logits_query_img2text_short',logits_query_img2text_short.shape)
#                     print('logits_query_new_img2img',logits_query_new_img2img.shape)

#                     if v_bias != None:
#                         loss_kl =  F.kl_div((new_emb_query@v_proj.t()+v_bias).softmax(dim=-1).log(), emb_query.softmax(dim=-1), reduction='sum') 
#                     elif v_bias == None and v_proj != None:
#                         loss_kl =  F.kl_div((new_emb_query@v_proj).softmax(dim=-1).log(), emb_query.softmax(dim=-1), reduction='sum') 
#                     else:
#                         loss_kl =  F.kl_div(new_emb_query.softmax(dim=-1).log(), emb_query.softmax(dim=-1), reduction='sum') 
                    loss_kl = weight* F.kl_div(F.log_softmax(
                        logits_query_new_img2img.reshape(-1,  opt.train_way)/temperature, dim=-1), F.softmax(logits_query_img2text_short.reshape(-1, opt.train_way)/temperature,dim=-1), reduction='mean')
                    loss = F.cross_entropy(logits_query_new_img2img.reshape(-1, opt.train_way),
                                           labels_query.reshape(-1)) \
                          +loss_kl \
                           + F.cross_entropy(logits_query_img2text.reshape(-1, opt.n_classes), labels_query_unique.reshape(-1))
                    print('kl_mean', loss_kl)
#                     print('kl_sum', F.kl_div(F.log_softmax(
#                         logits_query_new_img2img.reshape(-1,  opt.train_way)/temperature, dim=-1), F.softmax(logits_query_img2text_short.reshape(-1, opt.train_way)/temperature,dim=-1), reduction='sum'))

                if opt.test_head == 'Text':
                    acc = count_accuracy(logits_query_img2text.reshape(-1, opt.n_classes), \
                                         real_labels_query.reshape(-1))
                elif opt.test_head == 'Vision':
                    acc = count_accuracy(logits_query_new_img2img.reshape(-1, opt.train_way), labels_query.reshape(-1))

                elif opt.test_head == 'Fuse_af' and 'apart_loss' not in opt.loss:
                    if 'v_adapter' in opt.para_update:
                        if 'prompt' in opt.para_update:
                            acc = count_accuracy(logits_query_img2text.reshape(-1,opt.n_classes), \
                                real_labels_query.reshape(-1))
                        else:
                            acc = count_accuracy(logits_query_new_img2img.reshape(-1,opt.train_way), labels_query.reshape(-1))
                    else:
                        if opt.visual_pre == 'T':
                            acc = count_accuracy(logits_query_img2text.reshape(-1,opt.n_classes), \
                                real_labels_query.reshape(-1))
                        else:
                            acc = count_accuracy(
                                0.5 * logits_query_img2img.reshape(-1,
                                                                       opt.train_way) + 0.5 * logits_query_img2text.reshape(
                                    -1,
                                    opt.train_way), \
                                labels_query.reshape(-1))
                elif opt.test_head == 'Fuse_af' and 'apart_loss' in opt.loss:
                        acc = count_accuracy(
                            0.5 * logits_query_new_img2img.reshape(-1, opt.train_way) + 0.5 * logits_query_img2text.reshape(-1,
                                                                                                                        opt.train_way), \
                            labels_query.reshape(-1))
                elif opt.test_head == 'Fuse_be':
                    acc = count_accuracy(logits_query_img2fuse.reshape(-1, opt.train_way), \
                                         labels_query.reshape(-1))

                train_accuracies.append(acc.item())
                train_losses.append(loss.item())

                if (i % 1 == 0):
                    train_acc_avg = np.mean(np.array(train_accuracies))
                    log(log_file_path, 'Train Epoch: {}\tBatch: [{}]\tLoss: {}\tAccuracy: {} % ({} %)'.format(
                        epoch, i, loss.item(), train_acc_avg, acc))


                if opt.loss == 'apart_loss':
                    prompt_optimizer.zero_grad()
                    loss1 = F.cross_entropy(logits_query_img2text.reshape(-1, opt.train_way), labels_query.reshape(-1)) \
                     + F.cross_entropy(logits_query_text2text.reshape(-1, opt.train_way),
                                       labels_query.reshape(-1))
                    loss1.backward(retain_graph=True)
                    prompt_optimizer.step()

                    adapter_optimizer.zero_grad()
                    loss2 = F.cross_entropy(logits_query_new_img2img.reshape(-1, opt.train_way), labels_query.reshape(-1)) \
                     + F.cross_entropy(logits_query_new_img2text.reshape(-1, opt.train_way), labels_query.reshape(-1))
                    loss2.backward()
                    adapter_optimizer.step()
                elif opt.loss == 'apart_loss_kl':
                    prompt_optimizer.zero_grad()
                    loss1 = F.cross_entropy(logits_query_img2text.reshape(-1, opt.train_way), labels_query.reshape(-1)) \
                     + F.cross_entropy(logits_query_text2text.reshape(-1, opt.train_way),
                                       labels_query.reshape(-1))
                    loss1.backward(retain_graph=True)
                    prompt_optimizer.step()

                    adapter_optimizer.zero_grad()
                    loss2 = F.cross_entropy(logits_query_new_img2img.reshape(-1, opt.train_way), labels_query.reshape(-1)) \
                     + F.cross_entropy(logits_query_new_img2text.reshape(-1, opt.train_way), labels_query.reshape(-1)) \
                     +  F.kl_div(new_emb_query.softmax(dim=-1).log(), emb_query.softmax(dim=-1), reduction='sum') \
                     +  F.kl_div(new_emb_support.softmax(dim=-1).log(), emb_support.softmax(dim=-1), reduction='sum')
                    loss2.backward()
                    adapter_optimizer.step()
                elif opt.loss == 'apart_loss_kl_oldi2i':
                    prompt_optimizer.zero_grad()
                    loss1 = F.cross_entropy(logits_query_img2text.reshape(-1, opt.train_way), labels_query.reshape(-1)) \
                            + F.cross_entropy(logits_query_text2text.reshape(-1, opt.train_way),
                                              labels_query.reshape(-1))
                    loss1.backward(retain_graph=True)
                    prompt_optimizer.step()

                    adapter_optimizer.zero_grad()
                    loss2 = F.cross_entropy(logits_query_new_img2img.reshape(-1, opt.train_way),
                                            labels_query.reshape(-1)) \
                            + F.cross_entropy(logits_query_new_img2text.reshape(-1, opt.train_way),
                                              labels_query.reshape(-1)) \
                            + F.cross_entropy(logits_query_old_img2img.reshape(-1, opt.train_way),
                                            labels_query.reshape(-1)) \
                            +  F.kl_div(new_emb_query.softmax(dim=-1).log(), emb_query.softmax(dim=-1), reduction='sum') \
                            +  F.kl_div(new_emb_support.softmax(dim=-1).log(), emb_support.softmax(dim=-1),
                                       reduction='sum')
                    loss2.backward()
                    adapter_optimizer.step()
                elif opt.loss == 'newi2i+i2t':
                    prompt_optimizer.zero_grad()
                    loss1 = F.cross_entropy(logits_query_img2text_unique.reshape(-1, opt.n_classes), labels_query_unique.reshape(-1))
                    loss1.backward(retain_graph=True)
                    prompt_optimizer.step()

                    adapter_optimizer.zero_grad()
                    loss2 = F.cross_entropy(logits_query_new_img2img.reshape(-1, opt.train_way), labels_query.reshape(-1))
                    loss2.backward()
                    adapter_optimizer.step()
                elif opt.loss == 'newi2i+newi2t+i2t':
                    prompt_optimizer.zero_grad()
                    fl= FocalLoss(gamma=2)
                    loss1 = fl(logits_query_img2text.reshape(-1, opt.train_way), labels_query.reshape(-1))
                    loss1.backward(retain_graph=True)
                    prompt_optimizer.step()

                    adapter_optimizer.zero_grad()
                    loss2 = F.cross_entropy(logits_query_new_img2img.reshape(-1, opt.train_way), labels_query.reshape(-1)) \
                            + F.cross_entropy(logits_query_new_img2text.reshape(-1, opt.train_way), labels_query.reshape(-1))
                    loss2.backward()
                    adapter_optimizer.step()
                elif opt.loss == 'newi2i+kd+i2t':
                    prompt_optimizer.zero_grad()
                    if not opt.fl:
                        loss1 = F.cross_entropy(logits_query_img2text_unique.reshape(-1, opt.n_classes), labels_query_unique.reshape(-1))
                    else:
                        fl = FocalLoss(gamma=2)
                        loss1 = fl(logits_query_img2text_unique.reshape(-1, opt.n_classes), labels_query_unique.reshape(-1))
                    loss1.backward(retain_graph=True)
                    prompt_optimizer.step()
                    
                    adapter_optimizer.zero_grad()
                    temperature = opt.temp
                    if not opt.fl:
                        loss2 = F.cross_entropy(logits_query_new_img2img.reshape(-1, opt.train_way), labels_query.reshape(-1)) \
                                - (F.softmax(logits_query_img2text.reshape(-1, opt.n_classes) / temperature,
                                             dim=-1) * F.log_softmax(logits_query_new_img2text.reshape(-1, opt.n_classes) / temperature, dim=-1)).sum(dim=-1).mean()
                    else:
                        fl = FocalLoss(gamma=2)
                        loss2 = fl(logits_query_new_img2img.reshape(-1, opt.train_way), labels_query.reshape(-1)) \
                                - (F.softmax(logits_query_img2text.reshape(-1, opt.n_classes) / temperature,
                                             dim=-1) * F.log_softmax(logits_query_new_img2text.reshape(-1, opt.n_classes) / temperature, dim=-1)).sum(dim=-1).mean()
                    loss2.backward()
                    adapter_optimizer.step()
                    
                elif opt.loss == 'newi2i+kd2+i2t':
                    prompt_optimizer.zero_grad()
                    if not opt.fl:
                        loss1 = F.cross_entropy(logits_query_img2text_unique.reshape(-1, opt.n_classes), labels_query_unique.reshape(-1))
                    else:
                        fl = FocalLoss(gamma=2)
                        loss1 = fl(logits_query_img2text_unique.reshape(-1, opt.n_classes), labels_query_unique.reshape(-1))
                    loss1.backward(retain_graph=True)
                    prompt_optimizer.step()
                    
                    adapter_optimizer.zero_grad()
                    temperature = opt.temp
                    if not opt.fl:
                        loss2 = F.cross_entropy(logits_query_new_img2img.reshape(-1, opt.train_way), labels_query.reshape(-1)) \
                                - (F.softmax(logits_query_img2text_short.reshape(-1, opt.train_way) / temperature,
                                             dim=-1) * F.log_softmax(logits_query_new_img2img.reshape(-1, opt.train_way) / temperature, dim=-1)).sum(dim=-1).mean()
                    else:
                        fl = FocalLoss(gamma=2)
                        loss2 = fl(logits_query_new_img2img.reshape(-1, opt.train_way), labels_query.reshape(-1)) \
                                - (F.softmax(logits_query_img2text_short.reshape(-1, opt.train_way) / temperature,
                                             dim=-1) * F.log_softmax(logits_query_new_img2img.reshape(-1, opt.train_way) / temperature, dim=-1)).sum(dim=-1).mean()
                    loss2.backward()
                    adapter_optimizer.step()
                elif opt.loss == 'newi2i+klkd+i2t':
                    prompt_optimizer.zero_grad()
                    loss1 = F.cross_entropy(logits_query_img2text_unique.reshape(-1, opt.n_classes), labels_query_unique.reshape(-1))
                    
                    loss1.backward(retain_graph=True)
                    prompt_optimizer.step()
#                     if v_bias != None:
#                         loss_kl =  F.kl_div((new_emb_query@v_proj.t()+v_bias).softmax(dim=-1).log(), emb_query.softmax(dim=-1), reduction='sum') 
#                     elif v_bias == None and v_proj != None:
#                         loss_kl =  F.kl_div((new_emb_query@v_proj).softmax(dim=-1).log(), emb_query.softmax(dim=-1), reduction='sum') 
#                     else:
#                         loss_kl =  F.kl_div(new_emb_query.softmax(dim=-1).log(), emb_query.softmax(dim=-1), reduction='sum') 
                        
                    temperature = opt.temp
                    loss_kl = weight* F.kl_div(F.log_softmax(logits_query_new_img2img.reshape(-1,  opt.train_way)/temperature, dim=-1), F.softmax(logits_query_img2text_short.reshape(-1, opt.train_way)/temperature,dim=-1), reduction='mean')
                    loss2 = F.cross_entropy(logits_query_new_img2img.reshape(-1, opt.train_way), labels_query.reshape(-1)) + loss_kl
                    
                    
                    
                    loss2.backward()
                    adapter_optimizer.step()
                elif opt.loss == 'newi2i+kl+i2t':
                    loss1 = F.cross_entropy(logits_query_img2text_unique.reshape(-1, opt.n_classes), labels_query_unique.reshape(-1))
                    
                    loss1.backward(retain_graph=True)
                    prompt_optimizer.step()
                    loss2 = F.cross_entropy(logits_query_new_img2img.reshape(-1, opt.train_way), labels_query.reshape(-1)) +  F.kl_div(new_emb_query.softmax(dim=-1).log(), emb_query.softmax(dim=-1), reduction='sum')
                    
                    loss2.backward()
                    adapter_optimizer.step()
                    
                elif opt.loss == 'apart_loss_i2i':
                    prompt_optimizer.zero_grad()
                    loss1 = F.cross_entropy(logits_query_img2text.reshape(-1, opt.train_way), labels_query.reshape(-1)) \
                     + F.cross_entropy(logits_query_text2text.reshape(-1, opt.train_way),
                                       labels_query.reshape(-1))
                    loss1.backward(retain_graph=True)
                    prompt_optimizer.step()

                    adapter_optimizer.zero_grad()
                    loss2 = F.cross_entropy(logits_query_new_img2img.reshape(-1, opt.train_way), labels_query.reshape(-1))
                    loss2.backward()
                    adapter_optimizer.step()
                else:
                    
                    optimizer.zero_grad()       
                    loss.backward()
                    optimizer.step()
                    

        # Evaluate on the validation split
        _, _ = [x.eval() for x in (embedding_net, cls_head)]
        print('embedding_net.adapter.alpha:', embedding_net.adapter.alpha)
        if 'ViT' in opt.backbone:
            v_proj = embedding_net.image_encoder.proj
            v_bias = None
        else:
            v_proj = embedding_net.image_encoder.attnpool.c_proj.weight
            v_bias = embedding_net.image_encoder.attnpool.c_proj.bias

        val_accuracies = []
        val_losses = []
        val_accuracies_vision = []
        val_accuracies_text = []
        val_losses_vision = []
        val_losses_text = []

        ###
        val_loss_new_i2i, val_loss_new_i2t, val_loss_i2t, val_loss_t2t, val_loss_kl_query, val_loss_kl_support, val_loss_kd= [],[],[],[],[],[],[]

        if epoch == 0:
            continue
        if epoch != 1 and epoch % 5 != 0:
#         if epoch % 20 != 0:
            continue
        image_prototypes, new_image_prototypes, support_prompt_features = get_image_prototypes(opt, embedding_net, 'val')

        dloader_val = DataLoader(dataset=dataset_val,  # torch TensorDataset format
                                  batch_size=100,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=8,
                                  pin_memory=torch.cuda.is_available(),
                                  )

        dloader_valiter = iter(dloader_val)
        for i in trange(1, len(dloader_valiter) + 1):
            batch = next(dloader_valiter)
            data_query, real_labels_query = [
                x.cuda() for x in batch]
            test_n_query = data_query.shape[0]

            with torch.no_grad():
                emb_query, new_emb_query, logit_scale, _ = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
                emb_query = emb_query.reshape(1, test_n_query, -1)
                if new_emb_query != None:
                    new_emb_query = new_emb_query.reshape(1, test_n_query, -1)
                textemb_query = None

                logits_query_img2text, logits_query_img2img, logits_query_img2fuse, logits_query_new_img2img, logits_query_new_img2text, logits_query_old_img2img, _ = cls_head(None, emb_query, new_emb_query, None, None, image_prototypes, new_image_prototypes, support_prompt_features, textemb_query, None, logit_scale, v_proj, v_bias, opt.test_way, opt.val_shot, opt.n_classes, None, None)


            loss_t = x_entropy(logits_query_img2text.reshape(-1, categories), real_labels_query.reshape(-1))
            acc_t = count_true(logits_query_img2text.reshape(-1, categories), \
                                   real_labels_query.reshape(-1))
            if logits_query_new_img2img is not None:
                loss_v = x_entropy(logits_query_new_img2img.reshape(-1, categories), real_labels_query.reshape(-1))
                acc_v = count_true(logits_query_new_img2img.reshape(-1, categories), real_labels_query.reshape(-1))
            else:
                loss_v = x_entropy(logits_query_img2img.reshape(-1, categories), real_labels_query.reshape(-1))
                acc_v = count_true(logits_query_img2img.reshape(-1, categories), real_labels_query.reshape(-1))

            if opt.test_head == 'Text':
                loss = loss_t
                acc = acc_t
            elif opt.test_head == 'Vision':
                loss = loss_v
                acc = acc_v

            elif opt.test_head == 'Fuse_af' and 'apart_loss' not in opt.loss:
                if 'v_adapter' in opt.para_update:
                    if 'kl' not in opt.loss or 'klkd' in opt.loss:
                        loss_new_i2i, loss_new_i2t, loss_i2t, loss_kd= 0, 0, 0, 0
                        if 'i2i' in opt.loss:
                            loss_new_i2i = x_entropy(logits_query_new_img2img.reshape(-1, categories), real_labels_query.reshape(-1))
                        if 'newi2t' in opt.loss:
                            loss_new_i2t = x_entropy(logits_query_new_img2text.reshape(-1, categories), real_labels_query.reshape(-1))
                        if 'i2t' in opt.loss.replace('newi2t',''):
                            loss_i2t = x_entropy(logits_query_img2text.reshape(-1, categories), real_labels_query.reshape(-1))
                        if 'kd' in opt.loss and 'kd2' not in opt.loss and 'kl' not in opt.loss:
                            temperature = opt.temp
                            loss_kd = (F.softmax(logits_query_img2text.reshape(-1, categories) / temperature,
                                         dim=-1) * F.log_softmax(logits_query_new_img2text.reshape(-1, categories) / temperature, dim=-1)).sum(dim=-1).mean()
                        if 'kd2' in opt.loss:
                            temperature = opt.temp
                            loss_kd = - (F.softmax(logits_query_img2text.reshape(-1, categories) / temperature,
                                             dim=-1) * F.log_softmax(logits_query_new_img2img.reshape(-1, opt.categories) / temperature, dim=-1)).sum(dim=-1).mean()
                        if 'klkd' in opt.loss:
                            temperature = opt.temp
#                             loss_kd = weight*F.kl_div(F.log_softmax(
# #                         logits_query_new_img2img.reshape(-1,  categories)/temperature, dim=-1), F.softmax(logits_query_img2text.reshape(-1, categories)/temperature,dim=-1), reduction='sum')
#                             if v_bias != None:
#                                 loss_kl =  F.kl_div((new_emb_query@v_proj.t()+v_bias).softmax(dim=-1).log(), emb_query.softmax(dim=-1), reduction='sum') 
#                             elif v_bias == None and v_proj != None:
#                                 loss_kl =  F.kl_div((new_emb_query@v_proj).softmax(dim=-1).log(), emb_query.softmax(dim=-1), reduction='sum') 
#                             else:
#                                 loss_kl =  F.kl_div(new_emb_query.softmax(dim=-1).log(), emb_query.softmax(dim=-1), reduction='sum') 
                            temperature = opt.temp
                            loss_kl = weight* F.kl_div(F.log_softmax(
                        logits_query_new_img2img.reshape(-1,  categories)/temperature, dim=-1), F.softmax(logits_query_img2text_short.reshape(-1, categories)/temperature,dim=-1), reduction='mean')
                            loss_kd = loss_kl

                        loss = loss_new_i2i + loss_new_i2t + loss_i2t + loss_kd


                    elif opt.loss == 'i2t+i2i+kl':

                        loss_kl_query =  F.kl_div(new_emb_query.softmax(dim=-1).log(), emb_query.softmax(dim=-1),
                                                 reduction='sum')
                        loss_kl_support =  F.kl_div(new_emb_support.softmax(dim=-1).log(), emb_support.softmax(dim=-1),
                                                   reduction='sum')
                        loss_new_i2i = F.cross_entropy(logits_query_new_img2img.reshape(-1, categories),
                                                       real_labels_query.reshape(-1))
                        loss_i2t = F.cross_entropy(logits_query_img2text.reshape(-1, categories),
                                                       real_labels_query.reshape(-1))

                        loss = loss_new_i2i + loss_i2t + loss_kl_query + loss_kl_support

                    acc = count_true(
                        0.5 * logits_query_new_img2img.reshape(-1, categories) + 0.5 * logits_query_img2text.reshape(
                            -1, categories), \
                        real_labels_query.reshape(-1))
                else:
                    if opt.visual_pre == 'T':
                        loss_i2t = x_entropy(logits_query_img2text.reshape(-1, categories), real_labels_query.reshape(-1))
                        loss = x_entropy(logits_query_new_img2img.reshape(-1, categories),
                                         real_labels_query.reshape(-1)) + loss_i2t
                        acc = count_true(
                            0.5 * logits_query_new_img2img.reshape(-1, categories) + 0.5 * logits_query_img2text.reshape(
                                -1, categories), real_labels_query.reshape(-1))
                    else:
                        loss_i2t = x_entropy(logits_query_img2text.reshape(-1, categories), real_labels_query.reshape(-1))
                        loss = x_entropy(logits_query_img2img.reshape(-1, categories), real_labels_query.reshape(-1)) + loss_i2t
                        acc = count_true(
                            0.5 * logits_query_img2img.reshape(-1, categories) + 0.5 * logits_query_img2text.reshape(
                                -1,
                                categories), \
                            real_labels_query.reshape(-1))
            elif opt.test_head == 'Fuse_af' and 'apart_loss' in opt.loss:

                loss_new_i2i = F.cross_entropy(logits_query_new_img2img.reshape(-1, categories),
                                   real_labels_query.reshape(-1))
                loss_new_i2t = F.cross_entropy(logits_query_new_img2text.reshape(-1, categories),
                                     real_labels_query.reshape(-1))
                loss_i2t = F.cross_entropy(logits_query_img2text.reshape(-1, categories), real_labels_query.reshape(-1))

                if opt.loss =='apart_loss_kl':
                    # loss_t2t = F.cross_entropy(logits_query_text2text.reshape(-1, categories),
                    #                      real_labels_query.reshape(-1))

                    loss_kl_query =  F.kl_div(new_emb_query.softmax(dim=-1).log(), emb_query.softmax(dim=-1), reduction='sum')
                    loss_kl_support =  F.kl_div(new_emb_support.softmax(dim=-1).log(), emb_support.softmax(dim=-1), reduction='sum')

                    loss = loss_new_i2i + loss_new_i2t + loss_i2t + loss_kl_query + loss_kl_support
                elif opt.loss =='apart_loss':
                    loss = loss_new_i2i + loss_new_i2t + loss_i2t

                acc = count_true(
                    0.5 * logits_query_new_img2img.reshape(-1, categories) + 0.5 * logits_query_img2text.reshape(-1,
                                                                                                                categories), \
                    real_labels_query.reshape(-1))
            elif opt.test_head == 'Fuse_be':
                loss = x_entropy(logits_query_img2fuse.reshape(-1, categories), real_labels_query.reshape(-1))
                acc = count_true(logits_query_img2fuse.reshape(-1, categories), \
                                     real_labels_query.reshape(-1))

            val_accuracies.append(acc.item())
            val_accuracies_vision.append(acc_v.item())
            val_accuracies_text.append(acc_t.item())
            val_losses.append(loss.item())
            val_losses_vision.append(loss_v.item())
            val_losses_text.append(loss_t.item())
            ###
            if 'kl' in opt.loss and 'kd' not in opt.loss:
                val_loss_kl_query.append(loss_kl_query.item())
                val_loss_kl_support.append(loss_kl_support.item())
            if 'apart_loss' in opt.loss:
                val_loss_new_i2i.append(loss_new_i2i.item())
                val_loss_new_i2t.append(loss_new_i2t.item())
                val_loss_i2t.append(loss_i2t.item())
                # val_loss_t2t.append(loss_t2t.item())
            if 'newi2t' in opt.loss:
                val_loss_new_i2t.append(loss_new_i2t.item())
            if 'i2t' in opt.loss.replace('newi2t',''):
                val_loss_i2t.append(loss_i2t.item())
            if 'i2i' in opt.loss:
                val_loss_new_i2i.append(loss_new_i2i.item())
            if 'kd' in opt.loss:
                val_loss_kd.append(loss_kd.item())

        val_acc_avg = 100*np.sum(np.array(val_accuracies))/len(dataset_val)
        val_acc_avg_v = 100*np.sum(np.array(val_accuracies_vision))/len(dataset_val)
        val_acc_avg_t = 100*np.sum(np.array(val_accuracies_text))/len(dataset_val)

        val_loss_avg = np.mean(np.array(val_losses))
        val_loss_avg_v = np.mean(np.array(val_losses_vision))
        val_loss_avg_t = np.mean(np.array(val_losses_text))
        ###
        val_acces_avg.append(val_acc_avg)
        val_acces_avg_v.append(val_acc_avg_v)
        val_acces_avg_t.append(val_acc_avg_t)
        val_losses_avg.append(val_loss_avg)
        val_losses_avg_v.append(val_loss_avg_v)
        val_losses_avg_t.append(val_loss_avg_t)

        if 'kl' in opt.loss and 'klkd' not in opt.loss:
            val_loss_avg_kl_query = np.mean(np.array(val_loss_kl_query))
            val_loss_avg_kl_support = np.mean(np.array(val_loss_kl_support))
            log(log_file_path,'val_loss_avg_kl_query = '+ str(round(val_loss_avg_kl_query, 3)))
            log(log_file_path,'val_loss_avg_kl_support = '+ str(round(val_loss_avg_kl_support, 3)))

            val_losses_avg_kl_query.append(val_loss_avg_kl_query)
            val_losses_avg_kl_support.append(val_loss_avg_kl_support)

        if 'apart_loss' in opt.loss:
            val_loss_avg_new_i2i = np.mean(np.array(val_loss_new_i2i))
            val_loss_avg_new_i2t = np.mean(np.array(val_loss_new_i2t))
            log(log_file_path,'val_loss_avg_new_i2t = '+ str(round(val_loss_avg_new_i2t, 3)))
            val_loss_avg_i2t = np.mean(np.array(val_loss_i2t))
            # val_loss_avg_t2t = np.mean(np.array(val_loss_t2t))

            val_losses_avg_new_i2i.append(val_loss_avg_new_i2i)
            val_losses_avg_new_i2t.append(val_loss_avg_new_i2t)
            val_losses_avg_i2t.append(val_loss_avg_i2t)
            # val_losses_avg_t2t.append(val_loss_avg_t2t)
        if 'newi2t' in opt.loss:
            val_loss_avg_new_i2t = np.mean(np.array(val_loss_new_i2t))
            log(log_file_path, 'val_loss_avg_new_i2t = ' + str(round(val_loss_avg_new_i2t, 3)))
            val_losses_avg_new_i2t.append(val_loss_avg_new_i2t)
        if 'i2t' in opt.loss.replace('newi2t',''):
            val_loss_avg_i2t = np.mean(np.array(val_loss_i2t))
            val_losses_avg_i2t.append(val_loss_avg_i2t)
        if 'i2i' in opt.loss:
            val_loss_avg_new_i2i = np.mean(np.array(val_loss_new_i2i))
            val_losses_avg_new_i2i.append(val_loss_avg_new_i2i)
        if 'kd' in opt.loss:
            val_loss_avg_kd = np.mean(np.array(val_loss_kd))
            val_losses_avg_kd.append(val_loss_avg_kd)

        if (val_acc_avg+val_acc_avg_v+val_acc_avg_t)/3 >= max_val_acc:
            max_val_acc = (val_acc_avg+val_acc_avg_v+val_acc_avg_t)/3
            torch.save({'embedding': embedding_net.state_dict(), 'cls_head': cls_head.state_dict()}, \
                       os.path.join(opt.save_path, 'best_model_meta_val_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.pth'.format(opt.subsample, opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre)))
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy_Fuse_af: {:.3f} % (Best)' \
                .format(epoch, val_loss_avg, val_acc_avg))
        else:
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy_Fuse_af: {:.3f} %' \
                .format(epoch, val_loss_avg, val_acc_avg))
        log(log_file_path, 'Validation Epoch: {}\t\t\tLoss_Vision: {:.4f}\tAccuracy_Vision: {:.3f} %' \
            .format(epoch, val_loss_avg_v, val_acc_avg_v))
        log(log_file_path, 'Validation Epoch: {}\t\t\tLoss_Text: {:.4f}\tAccuracy_Text: {:.3f} %' \
            .format(epoch, val_loss_avg_t, val_acc_avg_t))

        if epoch == opt.num_epoch or epoch == 1 or epoch % 5 == 0:
            log(log_file_path,'val_acces_avg = '+ str([round(x,3) for x in val_acces_avg]))
            log(log_file_path, 'val_acces_avg_v = ' + str([round(x, 3) for x in val_acces_avg_v]))
            log(log_file_path, 'val_acces_avg_t = ' + str([round(x, 3) for x in val_acces_avg_t]))

            log(log_file_path, 'val_losses_avg = ' + str([round(x, 4) for x in val_losses_avg]))
            log(log_file_path, 'val_losses_avg_v = ' + str([round(x, 4) for x in val_losses_avg_v]))
            log(log_file_path, 'val_losses_avg_t = ' + str([round(x, 4) for x in val_losses_avg_t]))

            if 'apart_loss' in opt.loss:
                log(log_file_path, 'val_losses_avg_new_i2i = ' + str([round(x, 4) for x in val_losses_avg_new_i2i]))
                log(log_file_path, 'val_losses_avg_new_i2t = ' + str([round(x, 4) for x in val_losses_avg_new_i2t]))
                log(log_file_path, 'val_losses_avg_i2t = ' + str([round(x, 4) for x in val_losses_avg_i2t]))
            if 'kl' in opt.loss and 'klkd' not in opt.loss:
                log(log_file_path, 'val_losses_avg_kl_query = ' + str([round(x, 4) for x in val_losses_avg_kl_query]))
                log(log_file_path, 'val_losses_avg_kl_support = ' + str([round(x, 4) for x in val_losses_avg_kl_support]))

            if 'newi2t' in opt.loss:
                log(log_file_path, 'val_losses_avg_new_i2t = ' + str([round(x, 4) for x in val_losses_avg_new_i2t]))
            if 'i2t' in opt.loss.replace('newi2t',''):
                log(log_file_path, 'val_losses_avg_i2t = ' + str([round(x, 4) for x in val_losses_avg_i2t]))
            if 'i2i' in opt.loss:
                log(log_file_path, 'val_losses_avg_new_i2i = ' + str([round(x, 4) for x in val_losses_avg_new_i2i]))
            if 'kd' in opt.loss:
                log(log_file_path, 'val_losses_avg_kd = ' + str([round(x, 4) for x in val_losses_avg_kd]))
        if epoch == opt.num_epoch:
            length = len(val_acces_avg)
            x = [1] + list(np.arange(5,length*5,5))
            import matplotlib.pyplot as plt
            plt.figure(figsize=(16, 12))
            plt.subplot(221)
            plt.title('Comparison of loss in validation')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            val_losses_sum = list(np.array(val_losses_avg_v) + np.array(val_losses_avg_t))
            plt.plot(x, val_losses_sum, label='loss_avg_sum')
            plt.plot(x, val_losses_avg_v, label='loss_avg_v')
            plt.plot(x, val_losses_avg_t, label='loss_avg_t')
            min_val_losses_sum = np.argmin(val_losses_sum)
            min_val_losses_avg_v = np.argmin(val_losses_avg_v)
            min_val_losses_avg_t = np.argmin(val_losses_avg_t)
            plt.scatter(x[min_val_losses_sum], val_losses_sum[min_val_losses_sum], marker='o')
            plt.annotate(s="min (%.3f, %.3f)" % (x[min_val_losses_sum], val_losses_sum[min_val_losses_sum]), \
             xy=(x[min_val_losses_sum], val_losses_sum[min_val_losses_sum]) , xytext=(-20, 6), textcoords='offset points')
            plt.scatter(x[min_val_losses_avg_v], val_losses_avg_v[min_val_losses_avg_v], marker='o')
            plt.annotate(s="min (%.3f, %.3f)" % (x[min_val_losses_avg_v], val_losses_avg_v[min_val_losses_avg_v]), \
             xy=(x[min_val_losses_avg_v], val_losses_avg_v[min_val_losses_avg_v]) , xytext=(-20, 6), textcoords='offset points')
            plt.scatter(x[min_val_losses_avg_t], val_losses_avg_t[min_val_losses_avg_t], marker='o')
            plt.annotate(s="min (%.3f, %.3f)" % (x[min_val_losses_avg_t], val_losses_avg_t[min_val_losses_avg_t]), \
             xy=(x[min_val_losses_avg_t], val_losses_avg_t[min_val_losses_avg_t]) , xytext=(-20, 6), textcoords='offset points')

            plt.legend()
            #
            plt.subplot(222)
            plt.title('Comparison of acc in validation')
            plt.xlabel('epoch')
            plt.ylabel('acc')
            val_acces_avg_mean = list((np.array(val_acces_avg_v)+np.array(val_acces_avg_t)+np.array(val_acces_avg))/3)
            plt.plot(x, val_acces_avg, label='acc_avg_fuse')
            plt.plot(x, val_acces_avg_v, label='acc_avg_v')
            plt.plot(x, val_acces_avg_t, label='acc_avg_t')
            plt.plot(x, val_acces_avg_mean, label='acc_avg_mean')
            max_val_acces_avg = np.argmax(val_acces_avg)
            max_val_acces_avg_v = np.argmax(val_acces_avg_v)
            max_val_acces_avg_t = np.argmax(val_acces_avg_t)
            max_val_acces_avg_mean = np.argmax(val_acces_avg_mean)
            plt.scatter(x[max_val_acces_avg], val_acces_avg[max_val_acces_avg], marker='o')
            plt.annotate(s="max (%.3f, %.3f)" % (x[max_val_acces_avg], val_acces_avg[max_val_acces_avg]), \
             xy=(x[max_val_acces_avg], val_acces_avg[max_val_acces_avg]) , xytext=(-20, 6), textcoords='offset points')
            plt.scatter(x[max_val_acces_avg_v], val_acces_avg_v[max_val_acces_avg_v], marker='o')
            plt.annotate(s="max (%.3f, %.3f)" % (x[max_val_acces_avg_v], val_acces_avg_v[max_val_acces_avg_v]), \
             xy=(x[max_val_acces_avg_v], val_acces_avg_v[max_val_acces_avg_v]) , xytext=(-20, 6), textcoords='offset points')
            plt.scatter(x[max_val_acces_avg_t], val_acces_avg_t[max_val_acces_avg_t], marker='o')
            plt.annotate(s="max (%.3f, %.3f)" % (x[max_val_acces_avg_t], val_acces_avg_t[max_val_acces_avg_t]), \
             xy=(x[max_val_acces_avg_t], val_acces_avg_t[max_val_acces_avg_t]) , xytext=(-20, 6), textcoords='offset points')
            plt.scatter(x[max_val_acces_avg_mean], val_acces_avg_mean[max_val_acces_avg_mean], marker='o')
            plt.annotate(s="max (%.4f, %.4f)" % (x[max_val_acces_avg_mean], val_acces_avg_mean[max_val_acces_avg_mean]), \
             xy=(x[max_val_acces_avg_mean], val_acces_avg_mean[max_val_acces_avg_mean]) , xytext=(-20, 6), textcoords='offset points')

            plt.legend()
            
            if val_losses_avg_new_i2i or val_losses_avg_i2t:
                plt.subplot(223)
                plt.title('Comparison of loss in validation')
                plt.xlabel('epoch')
                plt.ylabel('loss')
                if val_losses_avg_new_i2i:
                    plt.plot(x, val_losses_avg_new_i2i, label='new_i2i')
                else:
                    val_losses_avg_new_i2i = [0]*length
                if val_losses_avg_i2t:
                    plt.plot(x, val_losses_avg_i2t, label='i2t')
                else:
                    val_losses_avg_i2t = [0]*length

                if val_losses_avg_new_i2t:
                    plt.plot(x, val_losses_avg_new_i2t, label='new_i2t')
                else:
                    val_losses_avg_new_i2t = [0]*length

                if val_losses_avg_kd:
                    plt.plot(x, val_losses_avg_kd, label='kd')
                else:
                    val_losses_avg_kd = [0]*length

                # plt.plot(x, val_losses_avg_t2t, label='t2t')
                if val_losses_avg_kl_query:
                    plt.plot(x, val_losses_avg_kl_query, label='kl_query')
                    plt.plot(x, val_losses_avg_kl_support, label='kl_support')
                else:
                    val_losses_avg_kl_query = [0] * length
                    val_losses_avg_kl_support = [0] * length

                loss_sum = list(np.array(val_losses_avg_new_i2i)+np.array(val_losses_avg_i2t)+np.array(val_losses_avg_new_i2t)+np.array(val_losses_avg_kl_query)+np.array(val_losses_avg_kl_support)+np.array(val_losses_avg_kd))
                plt.plot(x, loss_sum, label='loss_sum')
                plt.legend()

            
            loss_figure_path = os.path.join(opt.save_path,
                                            "train_loss_figure_{}_shot_{}_loss_{}_metatrain_Proto_{}{}_{}{}{}_{}.jpg".format(
                                                opt.subsample, opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim,
                                                opt.ctx_init, opt.n_ctx, opt.lr_decay, opt.visual_pre))
            plt.savefig(loss_figure_path)


def meta_test(opt, dataset_train, dataset_val, dataset_test, data_loader, epoch_size, lamda):
    # Dataloader of Gidaris & Komodakis (CVPR 2018)
    if dataset_test == None:
        dataset_test = dataset_val

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)

    log_file_path = os.path.join(opt.save_path, "test_ing_log.txt")
    if epoch_size == opt.val_episode and dataset_test != None:
        log_file_path = os.path.join(opt.save_path, "test_log.txt")
        log(log_file_path, 'best_lamda = ' + str(lamda))
    log(log_file_path, 'lamda = ' + str(lamda))
    log(log_file_path, str(vars(opt)))

    (embedding_net, cls_head), categories = get_model(opt)

    # Load saved model checkpoints
    if os.path.exists(os.path.join(opt.save_path, 'best_model_meta_val_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.pth'.format(opt.subsample, opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre))):
        saved_models = torch.load(os.path.join(opt.save_path, 'best_model_meta_val_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.pth'.format(opt.subsample, opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre)), map_location="cpu")
        embedding_net.load_state_dict(saved_models['embedding'])
#         if os.path.exists(os.path.join('experiments/my_meta_prompt_rn50_flowers102_ensemblefp16/', 'best_model_meta_val_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.pth'.format(opt.subsample, opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, 'ensemble', opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre))):
#             saved_models_2 = torch.load(os.path.join('experiments/my_meta_prompt_rn50_flowers102_ensemblefp16/', 'best_model_meta_val_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.pth'.format(opt.subsample, opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, 'ensemble', opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre)), map_location="cpu")
#             model_dict=["adapter.alpha", "adapter.linear1.weight", "adapter.linear2.weight"]
#             # 1. filter out unnecessary keys
#             pretrained_dict = {k: v for k, v in saved_models_2['embedding'].items() if k in model_dict}
#             # 2. overwrite entries in the existing state dict
#             embedding_dict = embedding_net.state_dict()
#             embedding_dict.update(pretrained_dict)
#             embedding_net.load_state_dict(embedding_dict)

        embedding_net.eval()
    else:
        _, _ = [x.eval() for x in (embedding_net, cls_head)]
    
    print('embedding_net.adapter.alpha:',embedding_net.adapter.alpha)
    # from torchsummary import summary
    # summary(embedding_net, input_size=(3, 224, 224), batch_size=-1)np.sum([int(np.prod(p.shape)) for p in model.parameters()])
    print("Total number of param in CLIP's image_encoder is ", np.sum([int(np.prod(p.shape)) for p in embedding_net.image_encoder.parameters()]))
    print("Total number of param in CLIP's text_encoder is ",  np.sum([int(np.prod(p.shape)) for p in embedding_net.text_encoder.parameters()]))
    print("Total number of param in CLIP's adapter is ",
          np.sum([int(np.prod(p.shape)) for p in embedding_net.adapter.parameters()]))
    print("Total number of param in CLIP's prompt_learner is ",
          np.sum([int(np.prod(p.shape)) for p in embedding_net.prompt_learner.parameters()]))
    if 'ViT' in opt.backbone:
        v_proj = embedding_net.image_encoder.proj
        v_bias = None
    else:
        v_proj = embedding_net.image_encoder.attnpool.c_proj.weight
        v_bias = embedding_net.image_encoder.attnpool.c_proj.bias
    max_val_acc = 0.0
    max_test_acc = 0.0
    
    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()

    # Evaluate on the validation split
    _, _ = [x.eval() for x in (embedding_net, cls_head)]
    test_accuracies = []
    test_losses = []
    test_accuracies_vision = []
    test_losses_vision = []
    test_accuracies_text = []
    test_losses_text = []
    test_X = np.empty(shape=[0,2,categories])
    test_y = np.empty(shape=[0])

    image_prototypes, new_image_prototypes, support_prompt_features = get_image_prototypes(opt, embedding_net, 'test')
    dloader_test = DataLoader(dataset=dataset_test,  # torch TensorDataset format
            batch_size=100,
            shuffle=False,
            drop_last=False,
            num_workers=8,
            pin_memory=torch.cuda.is_available(),
            )

    dloader_testiter = iter(dloader_test)
    for i in trange(1, len(dloader_testiter) + 1):
        batch = next(dloader_testiter)
        data_query, real_labels_query = [
            x.cuda() for x in batch]

        test_n_query = data_query.shape[0]
#         print('data_query.shape', data_query.shape)

        with torch.no_grad():
            emb_query, new_emb_query, logit_scale, _ = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(1, test_n_query, -1)
            if new_emb_query != None:
                new_emb_query = new_emb_query.reshape(1, test_n_query, -1)
            textemb_query = None

            logits_query_img2text, logits_query_img2img, logits_query_img2fuse, logits_query_new_img2img, logits_query_new_img2text, logits_query_old_img2img, _ = cls_head(None, emb_query, new_emb_query, None, None, image_prototypes, new_image_prototypes, support_prompt_features, textemb_query,
                                                                   None, logit_scale, v_proj, v_bias,
                                                                   opt.test_way, opt.val_shot, opt.n_classes, None,None)
#         print('logits_query_img2text', logits_query_img2text)
#         print('emb_query', emb_query)
#         print('support_prompt_features', support_prompt_features)

        loss_t = x_entropy(logits_query_img2text.reshape(-1, categories), real_labels_query.reshape(-1))
        acc_t = count_true(logits_query_img2text.reshape(-1, categories), \
                             real_labels_query.reshape(-1))
        X, y = baysian(logits_query_new_img2img.reshape(-1, categories), logits_query_img2text.reshape(-1, categories), real_labels_query.reshape(-1))

        if logits_query_new_img2img is not None:
            loss_v = x_entropy(logits_query_new_img2img.reshape(-1, categories), real_labels_query.reshape(-1))
            acc_v = count_true(logits_query_new_img2img.reshape(-1, categories), \
                                 real_labels_query.reshape(-1))
        else:
            loss_v = x_entropy(logits_query_img2img.reshape(-1, categories), real_labels_query.reshape(-1))
            acc_v = count_true(logits_query_img2img.reshape(-1, categories), \
                                 real_labels_query.reshape(-1))
        if opt.test_head == 'Text':
            loss = loss_t
            acc = acc_t

        elif opt.test_head == 'Vision':
            loss = loss_v
            acc = acc_v

        elif opt.test_head == 'Fuse_af' and 'apart_loss' not in opt.loss:
            if 'v_adapter' in opt.para_update:
                loss = x_entropy(logits_query_new_img2img.reshape(-1, categories), real_labels_query.reshape(-1)) + \
                       x_entropy(logits_query_img2text.reshape(-1, categories), real_labels_query.reshape(-1))
                acc = count_true(lamda*logits_query_new_img2img.reshape(-1, categories)+(1-lamda)*logits_query_img2text.reshape(-1, categories), \
                             real_labels_query.reshape(-1))
            else:
                if opt.visual_pre == 'T':
                    loss = x_entropy(logits_query_new_img2img.reshape(-1, categories), real_labels_query.reshape(-1)) + \
                           x_entropy(logits_query_img2text.reshape(-1, categories), real_labels_query.reshape(-1))
                    acc = count_true(
                        lamda * logits_query_new_img2img.reshape(-1,
                                                               categories) + (1-lamda) * logits_query_img2text.reshape(
                            -1, categories), \
                        real_labels_query.reshape(-1))
                else:
                    loss = x_entropy(logits_query_img2img.reshape(-1, categories), real_labels_query.reshape(-1)) + \
                           x_entropy(logits_query_img2text.reshape(-1, categories), real_labels_query.reshape(-1))
                    acc = count_true(
                        lamda * logits_query_img2img.reshape(-1, categories) + (1-lamda) * logits_query_img2text.reshape(-1,
                                                                                                                       categories), \
                        real_labels_query.reshape(-1))
        elif opt.test_head == 'Fuse_af' and 'apart_loss' in opt.loss:
            loss = x_entropy(logits_query_new_img2img.reshape(-1, categories), real_labels_query.reshape(-1)) + \
                   x_entropy(logits_query_img2text.reshape(-1, categories), real_labels_query.reshape(-1))
            acc = count_true(lamda*logits_query_new_img2img.reshape(-1, categories)+(1-lamda)*logits_query_img2text.reshape(-1, categories), \
                             real_labels_query.reshape(-1))
        elif opt.test_head == 'Fuse_be':
            loss = x_entropy(logits_query_img2fuse.reshape(-1, categories), real_labels_query.reshape(-1))
            acc = count_true(logits_query_img2fuse.reshape(-1, categories), \
                             real_labels_query.reshape(-1))

        test_accuracies.append(acc.item())
        test_losses.append(loss.item())
        test_accuracies_vision.append(acc_v.item())
        test_losses_vision.append(loss_v.item())
        test_accuracies_text.append(acc_t.item())
        test_losses_text.append(loss_t.item())
        test_X = np.append(test_X,X,0)
        test_y = np.append(test_y,y)

    test_acc_avg = 100*np.sum(np.array(test_accuracies))/len(dataset_test)
    test_acc_avg_v = 100*np.sum(np.array(test_accuracies_vision))/len(dataset_test)
    test_acc_avg_t = 100*np.sum(np.array(test_accuracies_text))/len(dataset_test)

    print('test_accuracies:', test_accuracies[0:10], test_accuracies[-10:])
#     print('test_accuracies_vision:', test_accuracies_vision)
#     print('test_accuracies_text:', test_accuracies_text)

    test_loss_avg = np.mean(np.array(test_losses))
    test_loss_avg_v = np.mean(np.array(test_losses_vision))
    test_loss_avg_t = np.mean(np.array(test_losses_text))

    log(log_file_path, 'Test Loss: {:.4f}\tAccuracy_Fuse_af: {:.3f} % (Best)' \
        .format(test_loss_avg, test_acc_avg))
    log(log_file_path, 'Test Loss: {:.4f}\tAccuracy_Vision: {:.3f} % (Best)' \
        .format(test_loss_avg_v, test_acc_avg_v))
    log(log_file_path, 'Test Loss: {:.4f}\tAccuracy_Text: {:.3f} % (Best)' \
        .format(test_loss_avg_t, test_acc_avg_t))

    return test_acc_avg, test_X, test_y

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def tSNE(opt, dataset_train_notfm, dataset_val, dataset_test, data_loader):
    # Dataloader of Gidaris & Komodakis (CVPR 2018)
    test_way =10
    query_sample = 150
    episodes = 1
    val_shot = 0

    ##test_labelIds_novel=[448-607] tieredImagenet
    ##test_labelIds_novel=[80-99] miniImagenet
#     sample_categories = [916,152,251,259,518,670,865]
    sample_categories = [916,152,251,259,518,670,781,865,936,221] #[39, 40, 54, 60, 62, 68, 110, 124, 126, 128]#
#     sample_categories = [916,152,251,259,518,670,781,865,936,221]

# [39, 40, 54, 60, 62, 68, 110, 124, 126, 128] #ImageNet

    
#     [351, 64, 34, 45, 148, 159, 112, 203, 90, 173] #SUN397

    #[351, 14, 148, 112, 64, 147, 385, 45, 90, 140]# random.sample(list(range(0,opt.n_classes)), test_way)
    print(sample_categories)

    
#     image_prototypes, new_image_prototypes, support_prompt_features = get_image_prototypes(opt, embedding_net, 'test')
    
    dloader_test = data_loader(
        dataset=dataset_train_notfm,
        nKnovel=test_way,
        nKbase=0,
        nExemplars=0,  # num training examples per novel category
        nTestNovel=query_sample*test_way,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=1,
        num_workers=1,
        epoch_size=1 * episodes,
        sample_categeries = sample_categories,# num of batches per epoch
    )

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)



    (embedding_net, cls_head), categories = get_model(opt)

    
    if os.path.exists(os.path.join(opt.save_path, 'best_model_meta_val_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.pth'.format(opt.subsample, opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre))):
        saved_models = torch.load(os.path.join(opt.save_path, 'best_model_meta_val_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.pth'.format(opt.subsample, opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre)), map_location="cpu")
        embedding_net.load_state_dict(saved_models['embedding'])
    # Load saved model checkpoints
    embedding_net.eval()
    print('embedding_net.adapter.alpha:',embedding_net.adapter.alpha)
    
    # from torchsummary import summary
    # summary(embedding_net, input_size=(3, 224, 224), batch_size=-1)

    if 'ViT' in opt.backbone:
        v_proj = embedding_net.image_encoder.proj
        v_bias = None
    else:
        v_proj = embedding_net.image_encoder.attnpool.c_proj.weight
        v_bias = embedding_net.image_encoder.attnpool.c_proj.bias
    max_val_acc = 0.0
    max_test_acc = 0.0

    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()

    # Evaluate on the validation split
    _, _ = [x.eval() for x in (embedding_net, cls_head)]
    test_accuracies = []
    test_losses = []
    test_accuracies_vision = []
    test_losses_vision = []
    test_accuracies_text = []
    test_losses_text = []
    if opt.backbone == 'RN50':
        x_hidden = 1024
        m_hidden = 2048
    elif opt.backbone == 'ViT-B/16':
        x_hidden = 512
        m_hidden = 768
    X = np.zeros(shape=(episodes,test_way*query_sample, x_hidden))
    M = np.zeros(shape=(episodes,test_way*query_sample, m_hidden))
    y = np.zeros(shape=(episodes,test_way*query_sample))

    for i, batch in enumerate(tqdm(dloader_test(opt.seed)), 0):
        data_support, labels_support, real_labels_support, \
        data_query, labels_query, real_labels_query, k_all, _ = [
            x.cuda() for x in batch]

        test_n_query = data_query.shape[0]
#         print('data_query.shape', data_query.shape)

        with torch.no_grad():
            emb_query, new_emb_query, logit_scale, _ = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(1, test_n_query, -1)
            if new_emb_query != None:
                new_emb_query = new_emb_query.reshape(1, test_n_query, -1)
            textemb_query = None

            # logits_query_img2text, logits_query_img2img, logits_query_img2fuse, logits_query_new_img2img, logits_query_new_img2text, logits_query_old_img2img,_ = cls_head(k_all, emb_query, new_emb_query, emb_support, new_emb_support, textemb_query,
            #                                                        textemb_support, labels_support, logit_scale, v_proj,v_bias,
            #                                                        opt.test_way, opt.val_shot, is_scale=True)
#         print('new_emb_query.shape', new_emb_query.shape)
        text_support = real_labels_query.reshape(-1).tolist()
#         print(text_support)
        X[i] = emb_query.squeeze().cpu().numpy().reshape(test_way*query_sample, x_hidden)
        M[i] = new_emb_query.squeeze().cpu().numpy().reshape(test_way*query_sample, m_hidden)
#         print(X[i].shape)
        yy = []
        sorted_categories = list(set(text_support))
        sorted_categories = sorted(sorted_categories)
        for t in text_support:
            yy.append(sorted_categories.index(t))
        y[i] = np.array(yy).squeeze()
#         print(text_support)

#         from sklearn import manifold
#         '''X是特征，不包含target;X_tsne是已经降维之后的特征'''
#         tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
#         xx = X[i].reshape(test_way * val_shot, -1)
#         yy = y[i].reshape(test_way * val_shot, -1).astype(np.uint8).squeeze()
#         xx_tsne = tsne.fit_transform(xx)
        
#         '''嵌入空间可视化'''
#         x_min, x_max = xx_tsne.min(0), xx_tsne.max(0)
#         xx_norm = (xx_tsne - x_min) / (x_max - x_min)  # 归一化
#         plt.figure()
        
#         scatter = plt.scatter(xx_norm[:, 0], xx_norm[:, 1], c=yy, cmap='gist_rainbow', s=2)
#         classes = [str(i) for i in range(test_way)]
#         plt.xticks([])
#         plt.yticks([])
#         plt.title('t-SNE embedding', fontsize=14)
#         plt.legend(handles=scatter.legend_elements()[0], labels=classes, fontsize=6)
#         plt.savefig("./vision_tSNE_k{}.jpg".format(str(i)))


    # from sklearn import manifold
    '''X是特征，不包含target;X_tsne是已经降维之后的特征'''
#     from cuml.manifold import TSNE
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, init='pca', random_state=501)
    X = X.reshape(test_way*query_sample*episodes,-1)
    M = M.reshape(test_way*query_sample*episodes,-1)
    # print(X)M
    # print(X.shape)
    y = y.reshape(test_way*query_sample*episodes,-1)
    Z = np.concatenate((X, y), axis=1)

    X = Z[:, :-1]
    # # X = X / np.linalg.norm(x=X, axis=-1, keepdims=True)
    y1 = Z[:, -1].astype(np.uint8)

    # print(X.shape)
#     print(y)
    print("Org data dimension is {}".format(X.shape))
    X_tsne = tsne.fit_transform(X)
    # print(X_tsne)
    print("Embedded data dimension is {}".format(X_tsne.shape))

    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(7.5, 3.5))

    plt.subplot(121)
    plt.axis('off')
    
    from matplotlib import cm
    # colors = cm.hsv(np.arange(test_way) / float(test_way))
    # print(colors)
    # cmap = get_cmap(test_way)
    # for i in range(X_norm.shape[0]):
    #     plt.scatter(X_norm[i, 0], X_norm[i, 1], c=plt.cm.tab20(int(y[i])))
    scatter = plt.scatter(X_norm[:,0], X_norm[:,1], c=y1, cmap ='tab10', s=1)
    classes = [str(i) for i in range(test_way)]
    plt.xticks([])
    plt.yticks([])
    plt.title('(a) Cross-modal features', fontsize=8, y=-0.1)
    plt.legend(handles=scatter.legend_elements()[0], labels=classes, fontsize=6, loc='upper left',borderaxespad=0,ncol=2)
#     plt.savefig("./vision_tSNE_v11_original.pdf",dpi=500)
    
    print('M.shape', M.shape)
    print('y.shape', y.shape)
    Z_m = np.concatenate((M, y), axis=1)
    M = Z_m[:, :-1]
    # # X = X / nMp.linalg.norm(x=X, axis=-1, keepdims=True)
    y2 = Z_m[:, -1].astype(np.uint8)

    # print(X.shape)
#     print(y)
    print("Org data dimension is {}".format(M.shape))
    X_tsne = tsne.fit_transform(M)
    # print(X_tsne)
    print("Embedded data dimension is {}".format(X_tsne.shape))

    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    
    plt.subplot(122)
    plt.axis('off')

    # colors = cm.hsv(np.arange(test_way) / float(test_way))
    # print(colors)
    # cmap = get_cmap(test_way)
    # for i in range(X_norm.shape[0]):
    #     plt.scatter(X_norm[i, 0], X_norm[i, 1], c=plt.cm.tab20(int(y[i])))
    scatter = plt.scatter(X_norm[:,0], X_norm[:,1], c=y2, cmap ='tab10', s=1)
    classes = [str(i) for i in range(test_way)]
    plt.xticks([])
    plt.yticks([])
    plt.title('(b) task-specific visual features', fontsize=8, y=-0.1)
    plt.legend(handles=scatter.legend_elements()[0], labels=classes, fontsize=6, loc='upper left',ncol=2, borderaxespad=0)
    plt.savefig("./Fig5_x1207.pdf",dpi=500)

def py_intersect(string_1, string_2):
    """
    :param string_1: 字符串
    :param string_2: 字符串
    :return: 两字符串的交集
    """
    result = ''
    for char in string_1:
        if char in string_2 and char not in result:
            result += char
    return result


def dempster(mp1, mp2, P):
    """
    :param mp1: 证据源1，numpy数组，存储信度
    :param mp2: 证据源2，numpy数组，存储信度
    :param P: 辨识框架
    :return: 返回融合信度和冲突因子
    """
    l = len(P)  # 幂集长度
    mp = np.zeros((1, l), 'float64')  # 初始化最终结果mp
    k_matrix = np.zeros((l, l))  # 冲突因子乘子
    for k in range(l):
        tmp = P[k]
        f_matrix = np.zeros((l, l))  # 融合乘子
        for i in range(l):
            for j in range(l):
                tmp_ij = py_intersect(P[i], P[j])  # 有无交集
                if not tmp_ij:  # 若空集
                    k_matrix[i][j] = 1
                if tmp_ij == tmp:  # 若交集等于P[k]
                    f_matrix[i][j] = 1
        mp[0][k] = sum(sum(np.dot(mp1.T, mp2) * f_matrix))
    k = sum(sum(np.dot(mp1.T, mp2) * k_matrix))
    mp = mp / (1 - k)
    return mp, k

def bit_and(a,b):
    l = len(a)
    ans =''
    for i in range(l):
        ans += str(int(a[i]) and int(b[i]))
    return ans

def DS_fusions(A):
    #编码那个太难想了，简简单单字符串比较好
    #A为n*3矩阵，首列为证据集合编码，后续两列为支持度。
    #mass函数假设只有两个
    num_terrorist = len(A[0][0])#字符串长度就是单个恐怖分子的数目
    all_terrorist = len(A) #整体数目，用来遍历
    ans = [0] * all_terrorist#最后的答案
    k = 0#算分母
    intersection = 0#算分子，用一个循环算完
    for i in range(all_terrorist):
        first = A[i][0]
        for j in range(all_terrorist):
            second = A[j][0]
            #tmp = first and second
            tmp = bit_and(first,second)
#             print(first,second,tmp)
            if tmp == '0'*num_terrorist:
                k += A[i][1]*A[j][2]
            for ii in range(all_terrorist):
                if tmp ==  A[ii][0]:
                    ans[ii] += A[i][1]*A[j][2]
#     print(1-k)
    for i in range(all_terrorist):
        ans[i] = ans[i]/(1-k)
    return ans

def softmax(X):
    X = np.exp(X) / (np.expand_dims(np.sum(np.exp(X),axis=-1),-1).repeat(opt.test_way,axis=-1))
    return X

def reshape(X):
    X = X.reshape(-1,opt.test_way*2)
    return X

def visualize(opt, dataset_train_notfm, dataset_val, dataset_test, data_loader):
    # Dataloader of Gidaris & Komodakis (CVPR 2018)
    test_way =10
    query_sample = 10
    episodes = 1
    val_shot = 0

    ##test_labelIds_novel=[448-607] tieredImagenet
    ##test_labelIds_novel=[80-99] miniImagenet
#     sample_categories = [916,152,251,259,518,670,865]
    sample_categories = [916,152,251,259,518,670,781,865,936,221] #[39, 40, 54, 60, 62, 68, 110, 124, 126, 128]#
#     sample_categories = [916,152,251,259,518,670,781,865,936,221]

# [39, 40, 54, 60, 62, 68, 110, 124, 126, 128] #ImageNet

    
#     [351, 64, 34, 45, 148, 159, 112, 203, 90, 173] #SUN397

    #[351, 14, 148, 112, 64, 147, 385, 45, 90, 140]# random.sample(list(range(0,opt.n_classes)), test_way)
    print(sample_categories)

    
#     image_prototypes, new_image_prototypes, support_prompt_features = get_image_prototypes(opt, embedding_net, 'test')
    
    dloader_test = data_loader(
        dataset=dataset_train_notfm,
        nKnovel=test_way,
        nKbase=0,
        nExemplars=0,  # num training examples per novel category
        nTestNovel=query_sample*test_way,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=1,
        num_workers=1,
        epoch_size=1 * episodes,
        sample_categeries = sample_categories,# num of batches per epoch
    )

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)



    (embedding_net, cls_head), categories = get_model(opt)

    
    if os.path.exists(os.path.join(opt.save_path, 'best_model_meta_val_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.pth'.format(opt.subsample, opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre))):
        saved_models = torch.load(os.path.join(opt.save_path, 'best_model_meta_val_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.pth'.format(opt.subsample, opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre)), map_location="cpu")
        embedding_net.load_state_dict(saved_models['embedding'])
    # Load saved model checkpoints
    embedding_net.eval()
    print('embedding_net.adapter.alpha:',embedding_net.adapter.alpha)
    
    # from torchsummary import summary
    # summary(embedding_net, input_size=(3, 224, 224), batch_size=-1)

    if 'ViT' in opt.backbone:
        v_proj = embedding_net.image_encoder.proj
        v_bias = None
    else:
        v_proj = embedding_net.image_encoder.attnpool.c_proj.weight
        v_bias = embedding_net.image_encoder.attnpool.c_proj.bias
    max_val_acc = 0.0
    max_test_acc = 0.0

    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()

    # Evaluate on the validation split
    _, _ = [x.eval() for x in (embedding_net, cls_head)]
    test_accuracies = []
    test_losses = []
    test_accuracies_vision = []
    test_losses_vision = []
    test_accuracies_text = []
    test_losses_text = []
    if opt.backbone == 'RN50':
        x_hidden = 1024
        m_hidden = 2048
    elif opt.backbone == 'ViT-B/16':
        x_hidden = 512
        m_hidden = 768
    X = np.zeros(shape=(episodes,test_way*query_sample, 49, x_hidden))
    M = np.zeros(shape=(episodes,test_way*query_sample, 49, m_hidden))
    y = np.zeros(shape=(episodes,test_way*query_sample))

    for i, batch in enumerate(tqdm(dloader_test(opt.seed)), 0):
        data_support, labels_support, real_labels_support, \
        data_query, labels_query, real_labels_query, k_all, _ = [
            x.cuda() for x in batch]

        test_n_query = data_query.shape[0]
#         print('data_query.shape', data_query.shape)

        with torch.no_grad():
            emb_query, new_emb_query, logit_scale, _ = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            print('emb_query.shape',emb_query.shape)
            emb_query = emb_query.reshape(1, test_n_query, 49,-1)
            if new_emb_query != None:
                new_emb_query = new_emb_query.reshape(1, test_n_query,49, -1)
            textemb_query = None

            # logits_query_img2text, logits_query_img2img, logits_query_img2fuse, logits_query_new_img2img, logits_query_new_img2text, logits_query_old_img2img,_ = cls_head(k_all, emb_query, new_emb_query, emb_support, new_emb_support, textemb_query,
            #                                                        textemb_support, labels_support, logit_scale, v_proj,v_bias,
            #                                                        opt.test_way, opt.val_shot, is_scale=True)
#         print('new_emb_query.shape', new_emb_query.shape)
        text_support = real_labels_query.reshape(-1).tolist()
#         print(text_support)
        X[i] = emb_query.squeeze().cpu().numpy().reshape(test_way*query_sample,49, x_hidden)
        M[i] = new_emb_query.squeeze().cpu().numpy().reshape(test_way*query_sample,49, m_hidden)
#         print(X[i].shape)
        yy = []
        sorted_categories = list(set(text_support))
        sorted_categories = sorted(sorted_categories)
        for t in text_support:
            yy.append(sorted_categories.index(t))
        y[i] = np.array(yy).squeeze()
#         print(text_support)

#         from sklearn import manifold
#         '''X是特征，不包含target;X_tsne是已经降维之后的特征'''
#         tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
#         xx = X[i].reshape(test_way * val_shot, -1)
#         yy = y[i].reshape(test_way * val_shot, -1).astype(np.uint8).squeeze()
#         xx_tsne = tsne.fit_transform(xx)
        
#         '''嵌入空间可视化'''
#         x_min, x_max = xx_tsne.min(0), xx_tsne.max(0)
#         xx_norm = (xx_tsne - x_min) / (x_max - x_min)  # 归一化
#         plt.figure()
        
#         scatter = plt.scatter(xx_norm[:, 0], xx_norm[:, 1], c=yy, cmap='gist_rainbow', s=2)
#         classes = [str(i) for i in range(test_way)]
#         plt.xticks([])
#         plt.yticks([])
#         plt.title('t-SNE embedding', fontsize=14)
#         plt.legend(handles=scatter.legend_elements()[0], labels=classes, fontsize=6)
#         plt.savefig("./vision_tSNE_k{}.jpg".format(str(i)))


    # from sklearn import manifold
    '''X是特征，不包含target;X_tsne是已经降维之后的特征'''
#     from cuml.manifold import TSNE
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, init='pca', random_state=501)
    X = X.reshape(test_way*query_sample*episodes,-1)
    M = M.reshape(test_way*query_sample*episodes,-1)
    # print(X)M
    # print(X.shape)
    y = y.reshape(test_way*query_sample*episodes,-1)
    Z = np.concatenate((X, y), axis=1)

    X = Z[:, :-1]
    # # X = X / np.linalg.norm(x=X, axis=-1, keepdims=True)
    y1 = Z[:, -1].astype(np.uint8)

    # print(X.shape)
#     print(y)
    print("Org data dimension is {}".format(X.shape))
    X_tsne = tsne.fit_transform(X)
    # print(X_tsne)
    print("Embedded data dimension is {}".format(X_tsne.shape))

    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(7.5, 3.5))

    plt.subplot(121)
    plt.axis('off')
    
    from matplotlib import cm
    # colors = cm.hsv(np.arange(test_way) / float(test_way))
    # print(colors)
    # cmap = get_cmap(test_way)
    # for i in range(X_norm.shape[0]):
    #     plt.scatter(X_norm[i, 0], X_norm[i, 1], c=plt.cm.tab20(int(y[i])))
    scatter = plt.scatter(X_norm[:,0], X_norm[:,1], c=y1, cmap ='tab10', s=1)
    classes = [str(i) for i in range(test_way)]
    plt.xticks([])
    plt.yticks([])
    plt.title('(a) Cross-modal features', fontsize=8, y=-0.1)
    plt.legend(handles=scatter.legend_elements()[0], labels=classes, fontsize=6, loc='upper left',borderaxespad=0,ncol=2)
#     plt.savefig("./vision_tSNE_v11_original.pdf",dpi=500)
    
    print('M.shape', M.shape)
    print('y.shape', y.shape)
    Z_m = np.concatenate((M, y), axis=1)
    M = Z_m[:, :-1]
    # # X = X / nMp.linalg.norm(x=X, axis=-1, keepdims=True)
    y2 = Z_m[:, -1].astype(np.uint8)

    # print(X.shape)
#     print(y)
    print("Org data dimension is {}".format(M.shape))
    X_tsne = tsne.fit_transform(M)
    # print(X_tsne)
    print("Embedded data dimension is {}".format(X_tsne.shape))

    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    
    plt.subplot(122)
    plt.axis('off')

    # colors = cm.hsv(np.arange(test_way) / float(test_way))
    # print(colors)
    # cmap = get_cmap(test_way)
    # for i in range(X_norm.shape[0]):
    #     plt.scatter(X_norm[i, 0], X_norm[i, 1], c=plt.cm.tab20(int(y[i])))
    scatter = plt.scatter(X_norm[:,0], X_norm[:,1], c=y2, cmap ='tab10', s=1)
    classes = [str(i) for i in range(test_way)]
    plt.xticks([])
    plt.yticks([])
    plt.title('(b) Vision-specific features', fontsize=8, y=-0.1)
    plt.legend(handles=scatter.legend_elements()[0], labels=classes, fontsize=6, loc='upper left',ncol=2, borderaxespad=0)
    plt.savefig("./Fig5_x4.pdf",dpi=500)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epoch', type=int, default=100,
                            help='number of training epochs')
    parser.add_argument('--save-epoch', type=int, default=10,
                            help='frequency of model saving')
    parser.add_argument('--train-shot', type=int, default=1,
                            help='number of support examples per training class')
    parser.add_argument('--val-shot', type=int, default=1,
                            help='number of support examples per validation class')
    parser.add_argument('--train-query', type=int, default=15,
                            help='number of query examples per training class')
    parser.add_argument('--val-episode', type=int, default=600,
                            help='number of episodes per validation')
    parser.add_argument('--val-query', type=int, default=15,
                            help='number of query examples per validation class')
    parser.add_argument('--train-way', type=int, default=10,
                            help='number of classes in one training episode')
    parser.add_argument('--test-way', type=int, default=10,
                            help='number of classes in one test (or validation) episode')
    parser.add_argument('--save-path', default='./experiments/meta_part_resnet12_mini')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--network', type=str, default='CLIP',
                            help='choose which embedding network to use. CLIP')
    parser.add_argument('--backbone', type=str, default='RN50',
                            help='choose which visual backbone network to use. RN50, RN101, ViT-B/16, ViT-B/32')
    parser.add_argument('--n-ctx', type=int, default=16,
                        help='number of context tokens')
    parser.add_argument('--ctx-init', type=str, default="",
                        help='initialization words')
    parser.add_argument('--prec', type=str, default='fp16',
                        help='choose which precision to use. fp16, fp32, amp')
    parser.add_argument('--class-token-position', type=str, default='end',
                        help='choose which precision to use. middle or end or front')
    parser.add_argument('--input-size', type=int, default=224,
                        help='image input size * image input size')
    parser.add_argument('--csc', type=bool, default=False,
                        help='class-specific context (False or True)')
    parser.add_argument('--init-weights', type=str, default="",
                        help='INIT_WEIGHTS')
    parser.add_argument('--class-positions', type=str, default="end",
                        help='end, middle, front')
    parser.add_argument('--head', type=str, default='Vision-Text',
                            help='choose which classification head to use. Vision-Text, Vision, FuseCLS')
    parser.add_argument('--loss', type=str, default='i2t',
                        help='choose which classification head to use. i2i, i2t, t2i, t2t, fuse2, i2t+t2t, t2i+i2i, i2t+i2i, newi2i+i2t, newi2i+newi2t+i2t, newi2i+kd+i2t, newi2i+kd2+i2t, newi2i+newi2t+kd+i2t, fuse2+t2t, fuse2+i2i, fuse2+i2i+t2t, apart_loss, apart_loss_i2i, apart_loss_kl, apart_loss_kl_oldi2i, i2t+i2i+kl')
    parser.add_argument('--test-head', type=str, default='Fuse_af',
                            help='choose which classification head to use. Vision, Text, Fuse_af, Fuse_be'),
    parser.add_argument('--para-update', type=str, default='prompt',
                        help='choose which para-update module to use. prompt, t_adapter, v_adapter, prompt+v_adapter')
    parser.add_argument('--adapter-dim', type=str, default='',
                        help='choose which dim of adapter to use. _dim_x4, _dim_x2')
    parser.add_argument('--visual-pre', type=str, default='T',
                        help='choose whether to use visual-pre feature,T,'',T_proj')
    parser.add_argument('--n-classes', type=int, default=1000,
                        help='number of classes')
    parser.add_argument('--lr-decay', type=str, default='step',
                        help='choose which lr decay to use. cosine, expo, step')
    parser.add_argument('--temp', type=int, default=5,
                        help='choose which lr decay to use. 5,10,20')
    parser.add_argument('--stage', type=str, default='',
                        help='choose which training stage to use. second')
    parser.add_argument('--proj', type=str, default='',
                        help="'',projup")
    parser.add_argument('--fl', type=bool, default=False,
                        help="whether focal loss to use")
    parser.add_argument('--warm-up', type=bool, default=False,
                        help="whether warm up lr to use")
    parser.add_argument('--pre_head', type=str, default='LinearNet',
                            help='choose which classification head to use. ProtoNet, Ridge, R2D2, SVM')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                            help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')
    parser.add_argument('--episodes-per-batch', type=int, default=1,
                            help='number of episodes per batch')
    parser.add_argument('--eps', type=float, default=0.0,
                            help='epsilon of label smoothing')
    parser.add_argument('--phase', type=str, default='metatest',
                        help='metatrain, metatest, tSNE')
    parser.add_argument('--use_trainval', type=str, default='False',
                        help='frequency of model saving')
    parser.add_argument('--subsample', type=str, default='all',
                        help='subsample of all, base, new')
    parser.add_argument('--seed', type=int, default=42,
                        help='number of episodes per batch')
    

    opt = parser.parse_args()

    (dataset_train, dataset_train_notfm, dataset_val, dataset_test, data_loader) = get_dataset(opt)

    # if opt.phase == 'pretrain':
    #     pre_train(opt, dataset_train, dataset_val, dataset_test, data_loader)
    if opt.phase == 'metatrain':
        seed_torch(opt.seed)
        meta_train(opt, dataset_train, dataset_val, dataset_test, data_loader)
#         meta_test(opt, dataset_train, dataset_val, dataset_test, data_loader)
        test_acc_avg_max = 0
        best_lamda = 0
#         for lamda in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#             acc, _, _ = meta_test(opt, dataset_train, dataset_val, dataset_val, data_loader, 50, lamda=lamda)
#             if test_acc_avg_max <= acc:
#                 test_acc_avg_max = acc
#                 print('lamda:', lamda)
#                 best_lamda = lamda
#         print('test_acc_avg_max:', test_acc_avg_max)
#         print('best_lamda:', best_lamda)

       
        if not os.path.exists(os.path.join(opt.save_path,'logits_val_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.subsample, opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim,opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay,opt.visual_pre))):
            _,X,y=meta_test(opt, None, dataset_val, None, data_loader, opt.val_episode, lamda=0.5)
            np.savez(os.path.join(opt.save_path, 'logits_val_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.subsample, opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre)),X,y)
        if not os.path.exists(os.path.join(opt.save_path, 'logits_test_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.subsample, opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre))):
            _,tX,ty=meta_test(opt, None, None, dataset_test, data_loader, opt.val_episode, lamda=0.5)
            np.savez(os.path.join(opt.save_path, 'logits_test_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.subsample, opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre)),tX,ty)
            
    elif opt.phase == 'metatest':
 
#         #grid search
#         test_acc_avg_max = 0
#         best_lamda = 0
#         for lamda in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#             acc = meta_test(opt, dataset_train, dataset_val, dataset_val, data_loader, 50, lamda=lamda)
#             if test_acc_avg_max <= acc:
#                 test_acc_avg_max = acc
#                 print('lamda:', lamda)
#                 best_lamda = lamda
#         print('test_acc_avg_max:', test_acc_avg_max)
#         print('best_lamda:', best_lamda)
#         meta_test(opt, dataset_train, dataset_val, dataset_test, data_loader, opt.val_episode, lamda=best_lamda)
        
#         #bayesian search
#         def eval_fn(lamda):
#             acc,_,_ = meta_test(opt, dataset_train, dataset_val, dataset_val, data_loader, 20, lamda)
#             return acc
#         bayes = BayesianOptimization(eval_fn, {'lamda':(0, 1)})
#         bayes.maximize(n_iter=20, init_points=3)
#         print('Final result: ', bayes.max)#{'lamda': 0.49321092200918537} {'lamda': 0.32924862052608117}
#         meta_test(opt, dataset_train, dataset_val, dataset_test, data_loader, opt.val_episode, lamda=bayes.max['params']['lamda'])

# #         ##bayesian mix
#         val_episodes=opt.n_classes//opt.test_way
#         from sklearn.naive_bayes import GaussianNB,MultinomialNB,ComplementNB
#         from sklearn.preprocessing import StandardScaler
#         from bayes_opt import BayesianOptimization
# #         from sklearn.ensemble import StackingClassifier,AdaBoostClassifier,GradientBoostingClassifier,VotingClassifier
# #         from sklearn.preprocessing import StandardScaler
# #         from sklearn.pipeline import make_pipeline
# #         from sklearn.model_selection import train_test_split
#         if not os.path.exists(os.path.join(opt.save_path, 'logits_val_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.subsample,opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre))):
#             _,X,y=meta_test(opt, None, dataset_val, None, data_loader, val_episodes, lamda=0.5)
#             np.savez(os.path.join(opt.save_path, 'logits_val_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.subsample,opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre)),X,y)
#         else:
#             npzfile = np.load(os.path.join(opt.save_path, 'logits_val_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.subsample,opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre)))
#             X = npzfile['arr_0']
#             y = npzfile['arr_1']

#         softmax_X = np.exp(X) / (np.expand_dims(np.sum(np.exp(X),axis=-1),-1).repeat(opt.n_classes,axis=-1))

#         X = X.reshape(-1,opt.n_classes*2)
#         softmax_X = softmax_X.reshape(-1,opt.n_classes*2)
#         print(X[X<=0])
#         X[X < 0] = 0

#         clf1 = MultinomialNB()
#         clf1.fit(X, y)
#         clf2 = MultinomialNB()
#         clf2.fit(softmax_X, y)
        
#         if np.sum(clf1.predict(X)==y) > np.sum(clf2.predict(softmax_X)==y):
#             clf = clf1
#             flag = 'X'
#         else:
#             clf = clf2
#             flag = 'softmax_X'
        
        
#         if not os.path.exists(os.path.join(opt.save_path, 'logits_test_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.subsample,opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre))):
#             _,tX,ty=meta_test(opt, None, None, dataset_test, data_loader, val_episodes, lamda=0.5)
#             np.savez(os.path.join(opt.save_path, 'logits_test_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.subsample,opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre)),tX,ty)
#         else:
#             npzfile = np.load(os.path.join(opt.save_path, 'logits_test_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.subsample,opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre)))
#             tX = npzfile['arr_0']
#             ty = npzfile['arr_1']
#         if flag == 'softmax_X':
#             tX = np.exp(tX) / (np.expand_dims(np.sum(np.exp(tX),axis=-1),-1).repeat(opt.n_classes,axis=-1))

#         tX = tX.reshape(-1,opt.n_classes*2)
#         pred = clf.predict(tX)
#         accuracy = 100 * np.sum(pred==ty)/ty.shape[0]
#         print('bayesian fusion result:', accuracy)
#         acces=[]
#         k=opt.test_way*opt.val_query
#         for i in range(val_episodes):
#             acce = 100 * np.sum(pred[i*k:i*k+k]==ty[i*k:i*k+k])/ty[i*k:i*k+k].shape[0]
#             acces.append(acce)
#         print('acces:', acces[:10], acces[-10:])
#         acc_ci95 = 1.96 * np.std(np.array(acces)) / np.sqrt(val_episodes)
# #         print('ty.shape[0]:', ty.shape[0])
#         print('bayesian fusion +-:', acc_ci95)


        if not 'ImageNet' in opt.dataset:
            # alpha learning
            val_episodes=opt.n_classes//opt.test_way
            import torch.nn as nn
            class Net(torch.nn.Module):
                def __init__(self,):
                    super(Net, self).__init__()
                    self.alpha = nn.Parameter(torch.FloatTensor([0.5]))
                def forward(self, x1, x2):
                    out = (torch.FloatTensor([1.0]) - self.alpha) * x1 + self.alpha * x2
    #                 print('self.alpha', self.alpha)
                    return out
            net = Net()
            optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
            loss_func = torch.nn.CrossEntropyLoss()

            if not os.path.exists(os.path.join(opt.save_path, 'logits_val_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.subsample,opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre))):
                _,X,y=meta_test(opt, None, dataset_val, None, data_loader, val_episodes, lamda=0.5)
                np.savez(os.path.join(opt.save_path, 'logits_val_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.subsample,opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre)),X,y)
            else:
                npzfile = np.load(os.path.join(opt.save_path, 'logits_val_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.subsample,opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre)))
                X = npzfile['arr_0']
                y = npzfile['arr_1']
                
                tx1 = torch.tensor(X[:,0,:])
                tx2 = torch.tensor(X[:,1,:])

#                 pred = np.argmax(tx1, axis=1).reshape(-1)
#                 label = y.reshape(-1)
#                 accuracy = np.array(pred==y).reshape(-1)
#                 accuracy_sum = [accuracy[i*50:i*50+50].mean()  for i in range(opt.n_classes)]
#                 print(accuracy_sum)
                
            import random
            batch_size = 200
            data_len = len(y.reshape(-1).tolist())
            for i in range(100):
                rand_arr = np.arange(X.shape[0])
                np.random.shuffle(rand_arr)
    #             print(rand_arr[0:5])
                x1 = torch.tensor(X[rand_arr[:batch_size],0,:])
                x2 = torch.tensor(X[rand_arr[:batch_size],1,:])
                y0 = torch.LongTensor(y.reshape(-1)[rand_arr[:batch_size]])
                out = net(x1, x2)
                loss = loss_func(out, y0)
    #             print('loss',loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if not os.path.exists(os.path.join(opt.save_path, 'logits_test_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.subsample,opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre))):
                _,tX,ty=meta_test(opt, None, None, dataset_test, data_loader, val_episodes, lamda=0.5)
                np.savez(os.path.join(opt.save_path, 'logits_test_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.subsample,opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre)),tX,ty)
            else:
                npzfile = np.load(os.path.join(opt.save_path, 'logits_test_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.subsample,opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre)))
                tX = npzfile['arr_0']
                ty = npzfile['arr_1']

            tx1 = torch.tensor(tX[:,0,:])
            tx2 = torch.tensor(tX[:,1,:])
            result = net(tx1, tx2)

            pred = torch.argmax(result, dim=1).view(-1).detach().numpy()
            accuracy = 100 * np.sum(pred==ty)/ty.shape[0]
            print('alpha fusion result:', accuracy)
#             print('net.alpha', net.alpha)
            acces=[]
#             k=50
#             for i in  range(opt.n_classes):
#                 acce = 100 * np.sum(pred[i*k:i*k+k]==ty[i*k:i*k+k])/ty[i*k:i*k+k].shape[0]
#                 acces.append(acce)
#             print('acces:', acces)
#             acc_ci95 = 1.96 * np.std(np.array(acces)) / np.sqrt(val_episodes)
    #         print('ty.shape[0]:', ty.shape[0])
#             print('alpha fusion +-:', acc_ci95)


         # random search
        val_episodes=opt.n_classes//opt.test_way
        import torch.nn as nn
        import random
        if not 'ImageNet' in opt.dataset:
            if not os.path.exists(os.path.join(opt.save_path, 'logits_val_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.subsample,opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre))):
                _,X,y=meta_test(opt, None, dataset_val, None, data_loader, val_episodes, lamda=0.5)
                np.savez(os.path.join(opt.save_path, 'logits_val_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.subsample,opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre)),X,y)
            else:
                npzfile = np.load(os.path.join(opt.save_path, 'logits_val_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.subsample,opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre)))
                X = npzfile['arr_0']
                y = npzfile['arr_1']

            MAX_ITER = 10
            best_score = 0
            best_alpha=0.5
            x1 = X[:,0,:]
            x2 = X[:,1,:]
            for i in range(MAX_ITER):
                alpha = random.uniform(0.0,1.0)
                result = alpha*x1+(1-alpha)*x2
                pred = np.argmax(result, axis=1)
                accuracy = 100 * np.sum(pred==y)/y.shape[0]
    #             print('alpha', alpha)
    #             print('accuracy',accuracy)
                if accuracy > best_score:
                    best_alpha =alpha
                    best_score = accuracy
            print('best_alpha', best_alpha)

        if not os.path.exists(os.path.join(opt.save_path, 'logits_test_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.subsample,opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre))):
            _,tX,ty=meta_test(opt, None, None, dataset_test, data_loader, val_episodes, lamda=0.5)
            np.savez(os.path.join(opt.save_path, 'logits_test_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.subsample,opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre)),tX,ty)
        else:
            npzfile = np.load(os.path.join(opt.save_path, 'logits_test_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.subsample,opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre)))
            tX = npzfile['arr_0']
            ty = npzfile['arr_1']
        
        tx1 = tX[:,0,:]
        tx2 = tX[:,1,:]
#         print('tx2',tx2[0:100:,:])
#         print('ty',ty)
#         best_alpha = 0.4
        if opt.dataset == 'ImageNet':
            if opt.train_shot == 1:
                best_alpha = 0.05
            elif opt.train_shot == 2:
                best_alpha = 0.075
            elif opt.train_shot == 4:
                best_alpha = 0.1
            elif opt.train_shot == 8:
                best_alpha = 0.22
            else:
                best_alpha = 0.27
        result = best_alpha*tx1 +(1-best_alpha)*tx2
        pred = np.argmax(result, axis=1)
        accuracy = 100 * np.sum(pred==ty)/ty.shape[0]
        print('random research result:', accuracy)
        acces=[]
        k=opt.test_way*opt.val_query
        for i in  range(val_episodes):
            acce = 100 * np.sum(pred[i*k:i*k+k]==ty[i*k:i*k+k])/ty[i*k:i*k+k].shape[0]
            acces.append(acce)
#         print('acces:', acces[:10], acces[-10:])
#         print('acces:', acces)
        acc_ci95 = 1.96 * np.std(np.array(acces)) / np.sqrt(val_episodes)
#         print('ty.shape[0]:', ty.shape[0])
        print('random research result +-:', acc_ci95)


# #         ##bayesian mix2
#         val_episodes=opt.n_classes//opt.test_way
#         from sklearn.naive_bayes import GaussianNB,MultinomialNB,ComplementNB
#         from sklearn.preprocessing import StandardScaler
#         from bayes_opt import BayesianOptimization
#         rng = np.random.RandomState(42)
#         if not os.path.exists(os.path.join(opt.save_path, 'logits_test_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.subsample,opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre))):
#             _,tX,ty=meta_test(opt, None, None, dataset_test, data_loader, val_episodes, lamda=0.5)
#             np.savez(os.path.join(opt.save_path, 'logits_test_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.subsample,opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre)),tX,ty)
#         else:
#             npzfile = np.load(os.path.join(opt.save_path, 'logits_test_{}_shot_{}_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.subsample,opt.train_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre)))
#             tX = npzfile['arr_0']
#             ty = npzfile['arr_1']
#         softmax_tX = np.exp(tX) / (np.expand_dims(np.sum(np.exp(tX),axis=-1),-1).repeat(opt.n_classes,axis=-1))
#         tX = tX.reshape(-1,opt.n_classes*2)
#         softmax_tX = softmax_tX.reshape(-1,opt.n_classes*2)
#         ty.reshape(-1)
#         print(tX[tX<=0])
#         tX[tX < 0] = 0
#         clf1 = MultinomialNB()
#         clf1.fit(tX, ty)
#         clf2 = MultinomialNB()
#         clf2.fit(softmax_tX, ty)
#         if np.sum(clf1.predict(tX)==ty) > np.sum(clf2.predict(softmax_tX)==ty):
#             clf = clf1
#             flag = 'tX'
#         else:
#             clf = clf2
#             flag = 'softmax_tX'
#         if flag == 'softmax_tX':
#             tX = softmax_tX
#         pred = clf.predict(tX)
#         accuracy = 100 * np.sum(pred==ty)/ty.shape[0]
#         print('bayesian fusion result:', accuracy)
#         acces=[]
#         k=opt.test_way*opt.val_query
#         for i in  range(val_episodes):
#             acce = 100 * np.sum(pred[i*k:i*k+k]==ty[i*k:i*k+k])/ty[i*k:i*k+k].shape[0]
#             acces.append(acce)
#         print('acces:', acces[:10], acces[-10:])
#         acc_ci95 = 1.96 * np.std(np.array(acces)) / np.sqrt(val_episodes)
# #         print('ty.shape[0]:', ty.shape[0])
#         print('bayesian fusion +-:', acc_ci95)


#         #dcs
#         from deslib.dcs.lca import LCA
#         from deslib.des.knora_u import KNORAU
#         from sklearn.ensemble import RandomForestClassifier
#         from deslib.des.meta_des import METADES
#         from deslib.des.knora_e import KNORAE
#         from deslib.dcs.mcb import MCB
#         from deslib.des.des_p import DESP
#         from sklearn.linear_model import LogisticRegression
#         from sklearn.ensemble import BaggingClassifier
#         from sklearn.tree import DecisionTreeClassifier
#         from deslib.static.stacked import StackedClassifier
#         from sklearn.model_selection import train_test_split
#         rng = np.random.RandomState(42)
#         val_episodes=opt.val_episode
# #         model = LCA()
#         if not os.path.exists(os.path.join(opt.save_path, 'logits_val_{}w_{}s_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.test_way, opt.val_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre))):
#             _,X,y=meta_test(opt, dataset_train, dataset_val, dataset_val, data_loader, val_episodes, lamda=0.5)
#             np.savez(os.path.join(opt.save_path, 'logits_val_{}w_{}s_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.test_way, opt.val_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre)),X,y)
#         else:
#             npzfile = np.load(os.path.join(opt.save_path, 'logits_val_{}w_{}s_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.test_way, opt.val_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre)))
#             X = npzfile['arr_0']
#             y = npzfile['arr_1']
#         X = np.exp(X) / (np.expand_dims(np.sum(np.exp(X),axis=-1),-1).repeat(opt.test_way,axis=-1))
#         X = X.reshape(-1,opt.test_way*2)

#         X_train, X_dsel, y_train, y_dsel = train_test_split(X, y,
#                                                     test_size=0.50,
#                                                     random_state=rng)

# #         pool_classifiers = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
# #                                      n_estimators=100,
# #                                      random_state=rng)

# #         pool_classifiers.fit(X_train, y_train)

#         RF = RandomForestClassifier(random_state=rng, n_estimators=100)
#         RF.fit(X_train, y_train)

#         # fit the model on the whole dataset
# #         model = KNORAU(pool_classifiers, random_state=rng)
#         model = KNORAU(RF, k=3, random_state=rng)
#         model.fit(X_dsel, y_dsel)

#         # make a single prediction
#         if not os.path.exists(os.path.join(opt.save_path, 'logits_test_{}w_{}s_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.test_way, opt.val_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre))):
#             _,tX,ty=meta_test(opt, dataset_train, dataset_val, dataset_test, data_loader, val_episodes, lamda=0.5)
#             np.savez(os.path.join(opt.save_path, 'logits_test_{}w_{}s_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.test_way, opt.val_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre)),tX,ty)
#         else:
#             npzfile = np.load(os.path.join(opt.save_path, 'logits_test_{}w_{}s_loss_{}_{}{}_{}_{}{}{}{}_{}.npz'.format(opt.test_way, opt.val_shot, opt.loss, opt.para_update, opt.adapter_dim, opt.head, opt.ctx_init, opt.n_ctx, opt.stage, opt.lr_decay, opt.visual_pre)))
#             tX = npzfile['arr_0']
#             ty = npzfile['arr_1']
            
#         tX = np.exp(tX) / (np.expand_dims(np.sum(np.exp(tX),axis=-1),-1).repeat(opt.test_way,axis=-1))
#         tX = tX.reshape(-1,opt.test_way*2)
#         pred = model.predict(tX)
#         accuracy = 100 * np.sum(pred==ty)/ty.shape[0]
#         print('dynamic selection result:', accuracy)
#         acces=[]
#         k=opt.test_way*opt.val_query
#         for i in  range(val_episodes):
#             acce = 100 * np.sum(pred[i*k:i*k+k]==ty[i*k:i*k+k])/ty[i*k:i*k+k].shape[0]
#             acces.append(acce)
#         print('acces:', acces[:10], acces[-10:])
#         acc_ci95 = 1.96 * np.std(np.array(acces)) / np.sqrt(val_episodes)
#         print('ty.shape[0]:', ty.shape[0])
#         print('dynamic selection result +-:', acc_ci95)




#         #softmax voting
#         val_episodes=opt.val_episode
#         _,tX,ty=meta_test(opt, dataset_train, dataset_val, dataset_test, data_loader, val_episodes, lamda=0.5)
#         tX = np.exp(tX) / (np.expand_dims(np.sum(np.exp(tX),axis=-1),-1).repeat(opt.test_way,axis=-1) + 1e-5)
#         X_sofmax = np.empty(shape=[tX.shape[0],tX.shape[-1]])
#         for i in range(tX.shape[0]):
#             if tX[i,0,:].max(axis=-1)>tX[i,1,:].max(axis=-1):
#                 X_sofmax[i,:] = tX[i,0,:]
#             else:
#                 X_sofmax[i, :] = tX[i, 1, :]
#         pred = X_sofmax.argmax(axis=-1)
#         accuracy = 100 * np.sum(pred==ty)/ty.shape[0]
#         print('max selection result:', accuracy)
#         acces=[]
#         k=opt.test_way*opt.val_query
#         for i in  range(val_episodes):
#             acce = 100 * np.sum(pred[i*k:i*k+k]==ty[i*k:i*k+k])/ty[i*k:i*k+k].shape[0]
#             acces.append(acce)
#         print('acces:', acces)
#         acc_ci95 = 1.96 * np.std(np.array(acces)) / np.sqrt(val_episodes)
#         print('ty.shape[0]:', ty.shape[0])
#         print('max selection +-:', acc_ci95)

# #         Dempster fusion
#         val_episodes=10#opt.val_episode

#         _,tX,ty=meta_test(opt, dataset_train, dataset_val, dataset_test, data_loader, val_episodes, lamda=0.5)
#         tX = np.exp(tX) / (np.expand_dims(np.sum(np.exp(tX),axis=-1),-1).repeat(opt.test_way,axis=-1) + 1e-5)
#         m1 = tX[:,0,:]
#         m2 = tX[:,1,:]
#         P = ['100', '010', '001', '110', '111']
#         mp = np.empty(shape=[m1.shape[0],m1.shape[-1]])
#         for i in range(m1.shape[0]):
# #             p, k = dempster(np.expand_dims(m1[i,:], axis=0), np.expand_dims(m2[i,:], axis=0), P)
#             tx=[]
#             tx.append(P)
#             tx.append(m1[i,:].tolist())
#             tx.append(m2[i,:].tolist())
#             tx = list(map(list, zip(*tx)))
                
#             mp[i] = np.array(DS_fusions(tx))
        
# #         print('m1=',m1)
# #         print('m2=',m2)
# #         print('mp=',mp)
#         pred = np.argmax(mp, axis=-1)
#         accuracy = 100 * np.sum(pred==ty)/ty.shape[0]
#         print('Dempster fusion result:', accuracy)
#         acces=[]
#         k=opt.test_way*opt.val_query
#         for i in  range(val_episodes):
#             acce = 100 * np.sum(pred[i*k:i*k+k]==ty[i*k:i*k+k])/ty[i*k:i*k+k].shape[0]
#             acces.append(acce)
#         print('acces:', acces[:10], acces[-10:])
#         acc_ci95 = 1.96 * np.std(np.array(acces)) / np.sqrt(val_episodes)
#         print('Dempster fusion +-:', acc_ci95)
        
#         #Softmax mean fusion
#         val_episodes=opt.val_episode

#         _,tX,ty=meta_test(opt, dataset_train, dataset_val, dataset_test, data_loader, val_episodes, lamda=0.5)
#         tX = np.exp(tX) / (np.expand_dims(np.sum(np.exp(tX),axis=-1),-1).repeat(opt.test_way,axis=-1) + 1e-5)
#         mp = (tX[:,0,:] + tX[:,1,:]) /2
#         pred = np.argmax(mp, axis=-1)
        
#         accuracy = 100 * np.sum(pred==ty)/ty.shape[0]
#         print('Softmax mean fusion result:', accuracy)
#         acces=[]
#         k=opt.test_way*opt.val_query
#         for i in  range(val_episodes):
#             acce = 100 * np.sum(pred[i*k:i*k+k]==ty[i*k:i*k+k])/ty[i*k:i*k+k].shape[0]
#             acces.append(acce)
#         print('acces:', acces[:10], acces[-10:])
#         acc_ci95 = 1.96 * np.std(np.array(acces)) / np.sqrt(val_episodes)
#         print('Softmax mean fusion +-:', acc_ci95)
        

    elif opt.phase == 'tSNE':
        tSNE(opt, dataset_train_notfm, dataset_val, dataset_test, data_loader)
