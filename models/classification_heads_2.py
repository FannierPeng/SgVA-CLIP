import os
import sys

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

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

def CosineNetHead(k_all, meta_part_infer, query, support, support_labels, n_way, n_shot, is_scale=False, normalize=True):
    """
    Constructs the prototype representation of each class(=mean of support vectors of each class) and
    returns the classification score (=L2 distance to each class prototype) on the query set.

    This model is the classification head described in:
    Prototypical Networks for Few-shot Learning
    (Snell et al., NIPS 2017).

    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """

    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)
    d = query.size(2)

    assert (query.dim() == 3)
    assert (support.dim() == 3)
    assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert (n_support == n_way * n_shot)  # n_support must equal to n_way * n_shot

    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)

    # From:
    # https://github.com/gidariss/FewShotWithoutForgetting/blob/master/architectures/PrototypicalNetworksHead.py
    # ************************* Compute Prototypes **************************
    labels_train_transposed = support_labels_one_hot.transpose(1, 2)

    prototypes = torch.bmm(labels_train_transposed, support)
    # Divide with the number of examples per novel category.
    prototypes = prototypes.div(
        labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes)
    )

    boost_prototypes, _ = meta_part_infer(prototypes.reshape(-1, d), k_all.reshape(-1), is_infer=is_scale)
    boost_prototypes = boost_prototypes.reshape(tasks_per_batch, n_way, d)
    # + Fusion & No Fusion
    # prototypes = prototypes * 0.5 + boost_prototypes * 0.5
    prototypes = prototypes

    # Distance Matrix Vectorization Trick
    logits = torch.nn.functional.cosine_similarity(query.unsqueeze(2).expand(-1, -1, prototypes.shape[1], -1),
                                                   prototypes.unsqueeze(1).expand(-1, query.shape[1], -1, -1), dim=-1)

    return logits

def ViTeCLSHead(k_all, query, new_query, support, new_support, image_prototypes, new_image_prototypes, textemb_support, textemb_query, support_labels, logit_scale, v_proj, v_bias, n_way, n_shot, n_classes, query_unique_id, is_scale=False, normalize=True):
    """
    Constructs the prototype representation of each class(=mean of support vectors of each class) and
    returns the classification score (=L2 distance to each class prototype) on the query set.

    This model is the classification head described in:
    Prototypical Networks for Few-shot Learning
    (Snell et al., NIPS 2017).

    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    tasks_per_batch = query.size(0)

    if image_prototypes ==None:
        support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_way * n_shot), n_way)
        support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_way * n_shot,
                                                             n_way)  # [episodes,n*k,1000]
        labels_train_transposed = support_labels_one_hot.transpose(1, 2).half()
#         print('labels_train_transposed.type',labels_train_transposed.type)
#         print('support.type',support.type)
        image_prototypes = torch.bmm(labels_train_transposed, support)
        image_prototypes = image_prototypes.div(
            labels_train_transposed.sum(dim=2, keepdim=True).expand_as(image_prototypes)).half()

        text_prototypes = textemb_support

        if new_support != None:
            # print('new_support: ', new_support.shape)
            new_image_prototypes = torch.bmm(labels_train_transposed, new_support)
            new_image_prototypes = new_image_prototypes.div(
                labels_train_transposed.sum(dim=2, keepdim=True).expand_as(new_image_prototypes)
        ).half()
        else:
            new_image_prototypes = None

    else:
        text_prototypes = textemb_support

    # # print('image_prototypes', image_prototypes[:, :10, :10])
    # # print('new_image_prototypes', new_image_prototypes[:, :20, :5])
    # if support_labels != None:
    #     support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_way* n_shot), n_classes)
    #     support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_way* n_shot, n_classes) #[episodes,n*k,1000]
    #     labels_train_transposed = support_labels_one_hot.transpose(1, 2)
    #     # print('labels_train_transposed.shape',labels_train_transposed.shape)
    #     prototypes = torch.bmm(labels_train_transposed, support)
    #     # Divide with the number of examples per novel category.
    #     prototypes = prototypes / n_shot #[episodes, 1000, emb]
    #
    #     new_prototypes = torch.bmm(labels_train_transposed, new_support)
    #     # Divide with the number of examples per novel category.
    #     new_prototypes = new_prototypes / n_shot  # [episodes, 1000, emb]
    #
    #     class_ids = support_labels.reshape(-1).tolist()
    #     class_ids = sorted(list(set(class_ids)))
    #     # print('class_ids:', class_ids)
    #     class_labels = torch.tensor(class_ids).cuda()
    #     support_labels_one_hot = one_hot(class_labels.view(n_way), n_classes)
    #     support_labels_one_hot = support_labels_one_hot.expand(tasks_per_batch, n_way, n_classes)
    #     support_labels_one_hot = support_labels_one_hot.transpose(1, 2)
    #     textemb_query = torch.bmm(support_labels_one_hot, textemb_query) #[episodes, 1000, emb]
    #     mask = torch.sum(support_labels_one_hot,dim=-1).view(n_classes,1).expand(n_classes,textemb_query.shape[-1])
    #     mask2 = torch.sum(support_labels_one_hot,dim=-1).view(n_classes,1).expand(n_classes,new_prototypes.shape[-1])
    #     text_prototypes = textemb_support.expand_as(image_prototypes)*(1-mask) + textemb_query
    #     new_image_prototypes = new_image_prototypes*(1-mask2) + new_prototypes
    #     image_prototypes = image_prototypes*(1-mask) + prototypes
    #
    #     # print('text_prototypes', text_prototypes[:, :5, :5])
    #     # print('new_prototypes', new_prototypes[:, :20, :5])
    #     # print('new_image_prototypes', new_image_prototypes[:, :20, :5])
    #
    # # print('text_prototypes = ', text_prototypes)
    # # print('textemb_query = ', textemb_query)


    fuse_prototype = None #0.5*image_prototypes + 0.5*text_prototypes
    # Distance Matrix Vectorization Trick query=[episodes_perbatch, n*q , emb_m]; prototypes=[episodes_perbatch, n, emb_m] ==> [episodes_perbatch, n*q, n, emb_m]
    if query_unique_id != None:
        query_unique = query.index_select(1, query_unique_id).view(n_shot, n_way, -1)
        # new_query_unique = new_query.index_select(1, query_unique_id)
        logits_img2text_unique = logit_scale * torch.bmm(query_unique, text_prototypes.expand_as(query_unique).transpose(1, 2))
    else:
        logits_img2text_unique = None
    logits_img2text = logit_scale * torch.bmm(query, text_prototypes.transpose(1, 2))
    logits_img2img = logit_scale * torch.bmm(query, image_prototypes.transpose(1, 2))
    logits_img2fuse = None #logit_scale * torch.bmm(query, fuse_prototype.transpose(1, 2))
    if new_query != None:
        logits_new_img2img = logit_scale * torch.bmm(new_query, new_image_prototypes.transpose(1, 2))
        if new_query.shape[-1] != text_prototypes.shape[-1]:
            if v_bias == None:
                logits_new_img2text = logit_scale * torch.bmm(new_query@v_proj, text_prototypes.transpose(1, 2))
                logits_old_img2img = None
            else:
                logits_new_img2text = logit_scale * torch.bmm(new_query@v_proj.t()+v_bias, text_prototypes.transpose(1, 2))
                logits_old_img2img = None
        else:
            logits_new_img2text = logit_scale * torch.bmm(new_query, text_prototypes.transpose(1, 2))
            logits_old_img2img = logit_scale * torch.bmm(query, new_image_prototypes.transpose(1, 2))
    else:
        logits_new_img2img = None
        logits_new_img2text = None
        logits_old_img2img =None

    if textemb_query==None:
        return logits_img2text, logits_img2img, logits_img2fuse, logits_new_img2img, logits_new_img2text, logits_old_img2img, logits_img2text_unique

    logits_text2img = None
    logits_text2text = None
    # print('logits_img2text', logits_img2text[:, :5, :5])
    # print('logits_new_img2img',logits_new_img2img[:, :5, :5])
    return logits_img2text, logits_text2img, logits_img2img, logits_img2fuse, logits_text2text, logits_new_img2img, logits_new_img2text, logits_old_img2img, logits_img2text_unique


def FuseCosineNetHead(k_all, meta_part_infer, query, support, support_labels, n_way, n_shot, is_scale=False, normalize=True):
    """
    Constructs the prototype representation of each class(=mean of support vectors of each class) and
    returns the classification score (=L2 distance to each class prototype) on the query set.

    This model is the classification head described in:
    Prototypical Networks for Few-shot Learning
    (Snell et al., NIPS 2017).

    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    scale = 10
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)
    d = query.size(2)

    assert (query.dim() == 3)
    assert (support.dim() == 3)
    assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert (n_support == n_way * n_shot)  # n_support must equal to n_way * n_shot

    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)

    # From:
    # https://github.com/gidariss/FewShotWithoutForgetting/blob/master/architectures/PrototypicalNetworksHead.py
    # ************************* Compute Prototypes **************************
    labels_train_transposed = support_labels_one_hot.transpose(1, 2)

    prototypes = torch.bmm(labels_train_transposed, support)
    # Divide with the number of examples per novel category.
    prototypes = prototypes.div(
        labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes)
    )
    if is_scale:
        boost_prototypes, _ = meta_part_infer(prototypes.reshape(-1, d), k_all.reshape(-1), use_scale=is_scale, is_infer=is_scale)
        boost_prototypes = boost_prototypes.reshape(tasks_per_batch, n_way, d)
    else:
        boost_prototypes = meta_part_infer(prototypes.reshape(-1, d), k_all.reshape(-1), use_scale=is_scale, is_infer=is_scale)
        boost_prototypes = boost_prototypes[0].reshape(tasks_per_batch, n_way, d)

    logits = torch.nn.functional.cosine_similarity(query.unsqueeze(2).expand(-1, -1, prototypes.shape[1], -1),
                                                   prototypes.unsqueeze(1).expand(-1, query.shape[1], -1, -1), dim=-1)
    assign_1 = F.softmax(logits * scale, dim=-1)
    assign_1 = torch.cat([support_labels_one_hot, assign_1], dim=1)
    assign_1_transposed = assign_1.transpose(1, 2)
    emb = torch.cat([support, query], dim=1)
    mean_1 = torch.bmm(assign_1_transposed, emb)
    mean_1 = mean_1.div(
        assign_1_transposed.sum(dim=2, keepdim=True).expand_as(mean_1)
    )
    diff = torch.pow(emb.unsqueeze(1).expand(-1, n_way, -1, -1) - mean_1.unsqueeze(2).expand(-1, -1, emb.shape[1], -1), 2)
    std_1 = (assign_1_transposed.unsqueeze(-1).expand_as(diff) * diff).sum(dim=2) / assign_1_transposed.unsqueeze(-1).expand_as(diff).sum(dim=2)

    logits = torch.nn.functional.cosine_similarity(query.unsqueeze(2).expand(-1, -1, boost_prototypes.shape[1], -1),
                                                   boost_prototypes.unsqueeze(1).expand(-1, query.shape[1], -1, -1), dim=-1)
    assign_2 = F.softmax(logits * scale, dim=-1)
    assign_2 = torch.cat([support_labels_one_hot, assign_2], dim=1)
    assign_2_transposed = assign_2.transpose(1, 2)
    emb = torch.cat([support, query], dim=1)
    mean_2 = torch.bmm(assign_2_transposed, emb)
    mean_2 = mean_2.div(
        assign_2_transposed.sum(dim=2, keepdim=True).expand_as(mean_2)
    )
    diff = torch.pow(emb.unsqueeze(1).expand(-1, n_way, -1, -1) - mean_2.unsqueeze(2).expand(-1, -1, emb.shape[1], -1), 2)
    std_2 = (assign_2_transposed.unsqueeze(-1).expand_as(diff) * diff).sum(dim=2) / assign_2_transposed.unsqueeze(-1).expand_as(diff).sum(dim=2)

    prototypes = (mean_1 * std_2 + mean_2 * std_1) / (std_2 + std_1)
    logits = torch.nn.functional.cosine_similarity(query.unsqueeze(2).expand(-1, -1, prototypes.shape[1], -1),
                                                   prototypes.unsqueeze(1).expand(-1, query.shape[1], -1, -1), dim=-1)
    # Distance Matrix Vectorization Trick
    return logits

class ClassificationHead(nn.Module):
    def __init__(self, base_learner='MetaOptNet', enable_scale=False):
        super(ClassificationHead, self).__init__()
        if ('Cosine' in base_learner):
            self.head = CosineNetHead
        elif ('FuseCos' in base_learner):
            self.head = FuseCosineNetHead
        elif ('VisionCLS' in base_learner):
            self.head = ViTeCLSHead
        elif ('Vision-TextCLS' in base_learner):
            self.head = ViTeCLSHead
        elif ('FuseCLS' in base_learner):
            self.head = ViTeCLSHead
        else:
            print ("Cannot recognize the base learner type")
            assert(False)
        
        # Add a learnable scale
        self.enable_scale = enable_scale
        self.scale = nn.Parameter(torch.FloatTensor([1.0]))
        
    def forward(self, k_all, query, new_query, support, new_support, image_prototypes, new_image_prototypes, textemb_support, textemb_query, support_labels, logit_scale, v_proj, v_bias, n_way, n_shot, n_classes, query_unique_id, **kwargs):
        if self.enable_scale:
            return self.scale * self.head(k_all, query, new_query, support, new_support, image_prototypes, new_image_prototypes, textemb_support, textemb_query, support_labels, logit_scale,  v_proj, v_bias, n_way, n_shot, n_classes, query_unique_id, **kwargs)
        else:
            return self.head(k_all, query, new_query, support, new_support, image_prototypes, new_image_prototypes, textemb_support, textemb_query, support_labels, logit_scale, v_proj, v_bias, n_way, n_shot, n_classes, query_unique_id, **kwargs)