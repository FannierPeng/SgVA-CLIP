from __future__ import print_function

import os
import os.path
import numpy as np
import random
import json
import pickle
import math
from collections import defaultdict
import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchnet as tnt
import h5py
from collections import OrderedDict
from PIL import Image
from PIL import ImageEnhance
from scipy.io import loadmat
import sys
sys.path.append("..")
from utils import Datum
from data.oxford_flowers import buildLabelIndex, read_json, read_json, convert, save_split, read_split, listdir_nohidden, read_and_split_data, split_trainval
# Set the appropriate paths of the datasets here.

_IMAGE_DATASET_DIR = '/userhome/CLIP/data/stanford_cars/'
# split_path = os.path.join(_IMAGE_DATASET_DIR, "split_zhou_StanfordCars.json")
# data = read_json(split_path)
# val = {}
# for item in data['val']:
#     if item[2] not in val.keys():
#         val[int(item[1])] = item[2]
# classnames=[]
# for k in  sorted(val.keys()):
#     print(k)
#     classnames.append(val[k])
# print('classnames:', classnames)
#

class StanfordCars(data.Dataset):
    def __init__(self, phase='train', do_not_use_random_transf=False, subsample='all', num_shots=1):

        assert (phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.name = 'StanfordCars_' + phase
        self.split_fewshot_dir = os.path.join(_IMAGE_DATASET_DIR, "split_fewshot")
        self.split_path = os.path.join(_IMAGE_DATASET_DIR, "split_zhou_StanfordCars.json")
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

        if not os.path.exists(self.split_fewshot_dir):
            os.mkdir(self.split_fewshot_dir)
        if os.path.exists(self.split_path):
            if not os.path.exists(preprocessed):
                train, val, test = read_split(self.split_path, _IMAGE_DATASET_DIR)
            else:
                test = read_split(self.split_path, _IMAGE_DATASET_DIR, self.phase)
        else:
            trainval_file = os.path.join(_IMAGE_DATASET_DIR, "devkit", "cars_train_annos.mat")
            test_file = os.path.join(_IMAGE_DATASET_DIR, "cars_test_annos_withlabels.mat")
            meta_file = os.path.join(_IMAGE_DATASET_DIR, "devkit", "cars_meta.mat")
            trainval = self.read_data("cars_train", trainval_file, meta_file)
            test = self.read_data("cars_test", test_file, meta_file)
            train, val = split_trainval(trainval)
            save_split(train, val, test, self.split_path, _IMAGE_DATASET_DIR)

        if num_shots >= 1:
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(self, train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(self, val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        train, val = self.subsample_classes(train, val, subsample=subsample)
        if self.phase == 'train':
            # During training phase we only load the training phase images
            # of the training categories (aka base categories).
            self.labels = [item.label for item in train]
            self.impaths = [item.impath for item in train]
            self.classnames = [item.classname for item in train]

            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
            self.labelIds_base = self.labelIds
            self.num_cats_base = len(self.labelIds_base)
        #             print('Loading Flowers102 dataset - phase {0}'.format(phase))

            # print('classnames =', self.classnames)

        elif self.phase == 'val' or self.phase == 'test':
            if self.phase == 'test':
                # load data that will be used for evaluating the recognition
                # accuracy of the base categories.
                train, val, test = self.subsample_classes(train, val, test, subsample=subsample)
                self.labels = [item.label for item in test]
                self.impaths = [item.impath for item in test]
                self.classnames = [item.classname for item in test]
                print('Loading Flowers102 dataset - phase {0}'.format(phase))

            else:  # phase=='val'
                # load data that will be used for evaluating the recognition
                # accuracy of the base categories.
                self.labels = [item.label for item in val]
                self.impaths = [item.impath for item in val]
                self.classnames = [item.classname for item in val]

            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
            self.labelIds_base = self.labelIds
            self.num_cats_base = len(self.labelIds_base)
            self.labelIds_novel = self.labelIds
            self.num_cats_novel = len(self.labelIds_novel)

        else:
            raise ValueError('Not valid phase {0}'.format(self.phase))

        mean_pix = [0.48145466, 0.4578275, 0.40821073]  # CLIP
        std_pix = [0.26862954, 0.26130258, 0.27577711]  # CLIP
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)
        if (self.phase == 'test' or self.phase == 'val') or (do_not_use_random_transf == True):
            self.transform = transforms.Compose([
                transforms.Resize(224, interpolation=Image.BICUBIC),
                transforms.CenterCrop(224),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(224, interpolation=Image.BICUBIC),
                transforms.RandomCrop(224),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        impath, label = self.impaths[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(impath)
        img = img.convert('RGB')
        img = self.transform(img)
        #         print('img.shape',img.shape)
        return img, label

    def __len__(self):
        return len(self.impaths)

    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """Divide classes into two groups. The first group
        represents base classes while the second group represents
        new classes.
        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args

        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]  # take the first half
        else:
            selected = labels[m:]  # take the second half
        relabeler = {y: y_new for y_new, y in enumerate(selected)}

        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = Datum(
                    impath=item.impath,
                    label=relabeler[item.label],
                    classname=item.classname
                )
                dataset_new.append(item_new)
            output.append(dataset_new)

        return output

    @staticmethod
    def generate_fewshot_dataset(
            self, data_sources, num_shots=-1, repeat=False):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a few number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed (default: False).
        """

        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f"Creating a {num_shots}-shot dataset")

        output = []

        print('len(data_sources)', len(data_sources))

        tracker = self.split_dataset_by_label(self, data_sources)

        dataset = []

        for label, items in tracker.items():
            if len(items) >= num_shots:
                sampled_items = random.sample(items, num_shots)  # 随机采样了该类中的k个shot

            else:
                if repeat:
                    sampled_items = random.choices(items, k=num_shots)
                else:
                    sampled_items = items
            dataset.extend(sampled_items)

        output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output

    @staticmethod
    def split_dataset_by_label(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.label].append(item)

        return output

    def read_data(self, image_dir, anno_file, meta_file):
        anno_file = loadmat(anno_file)["annotations"][0]
        meta_file = loadmat(meta_file)["class_names"][0]
        items = []

        for i in range(len(anno_file)):
            imname = anno_file[i]["fname"][0]
            impath = os.path.join(self.dataset_dir, image_dir, imname)
            label = anno_file[i]["class"][0, 0]
            label = int(label) - 1  # convert to 0-based index
            classname = meta_file[label][0]
            names = classname.split(" ")
            year = names.pop(-1)
            names.insert(0, year)
            classname = " ".join(names)
            item = Datum(impath=impath, label=label, classname=classname)
            items.append(item)

        return items

class FewShotDataloader():
    def __init__(self,
                 dataset,
                 nKnovel=5,  # number of novel categories.
                 nKbase=-1,  # number of base categories.
                 nExemplars=1,  # number of training examples per novel category.
                 nTestNovel=15 * 5,  # number of test examples for all the novel categories.
                 nTestBase=15 * 5,  # number of test examples for all the base categories.
                 batch_size=1,  # number of training episodes per batch.
                 num_workers=4,
                 epoch_size=2000,  # number of batches per epoch.
                 sample_categeries=None,
                 ):

        self.dataset = dataset
        self.phase = self.dataset.phase
        self.sample_categeries = sample_categeries

        # train时 max_possible_nKnovel=64，nKnovel=novel 类数目=5， nKbase=0
        # val时 max_possible_nKnovel=16，nKnovel=novel 类数目=5， nKbase=0
        max_possible_nKnovel = (self.dataset.num_cats_base if self.phase == 'train'
                                else self.dataset.num_cats_novel)
        assert (nKnovel >= 0 and nKnovel <= max_possible_nKnovel)

        self.nKnovel = nKnovel

        max_possible_nKbase = self.dataset.num_cats_base
        nKbase = nKbase if nKbase >= 0 else max_possible_nKbase
        if self.phase == 'train' and nKbase > 0:
            nKbase -= self.nKnovel  # train时nKbase需要减去nKnovel
            max_possible_nKbase -= self.nKnovel

        assert (nKbase >= 0 and nKbase <= max_possible_nKbase)
        self.nKbase = nKbase

        self.nExemplars = nExemplars
        self.nTestNovel = nTestNovel
        self.nTestBase = nTestBase
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.num_workers = num_workers
        self.is_eval_mode = (self.phase == 'test') or (self.phase == 'val')

    def sampleImageIdsFrom(self, cat_id, sample_size=1):
        """
        Samples `sample_size` number of unique image ids picked from the
        category `cat_id` (i.e., self.dataset.label2ind[cat_id]).

        Args:
            cat_id: a scalar with the id of the category from which images will
                be sampled.
            sample_size: number of images that will be sampled.

        Returns:
            image_ids: a list of length `sample_size` with unique image ids.
        """
        assert (cat_id in self.dataset.label2ind)
        #         print(self.dataset.label2ind[cat_id])
        #         print(sample_size)
        assert (len(self.dataset.label2ind[cat_id]) >= sample_size)
        # Note: random.sample samples elements without replacement.
        return random.sample(self.dataset.label2ind[cat_id], sample_size)

    def sampleCategories(self, cat_set, sample_size=1):
        """
        Samples `sample_size` number of unique categories picked from the
        `cat_set` set of categories. `cat_set` can be either 'base' or 'novel'.

        Args:
            cat_set: string that specifies the set of categories from which
                categories will be sampled.
            sample_size: number of categories that will be sampled.

        Returns:
            cat_ids: a list of length `sample_size` with unique category ids.
        """
        if cat_set == 'base':
            labelIds = self.dataset.labelIds_base
        elif cat_set == 'novel':
            labelIds = self.dataset.labelIds_novel
        else:
            raise ValueError('Not recognized category set {}'.format(cat_set))

        assert (len(labelIds) >= sample_size)
        # return sample_size unique categories chosen from labelIds set of
        # categories (that can be either self.labelIds_base or self.labelIds_novel)
        # Note: random.sample samples elements without replacement.
        if self.sample_categeries != None:
            return self.sample_categeries
        else:
            return random.sample(labelIds, sample_size)

    def sample_base_and_novel_categories(self, nKbase, nKnovel):
        """
        Samples `nKbase` number of base categories and `nKnovel` number of novel
        categories.

        Args:
            nKbase: number of base categories
            nKnovel: number of novel categories

        Returns:
            Kbase: a list of length 'nKbase' with the ids of the sampled base
                categories.
            Knovel: a list of lenght 'nKnovel' with the ids of the sampled novel
                categories.
        """
        if self.is_eval_mode:
            assert (nKnovel <= self.dataset.num_cats_novel)
            # sample from the set of base categories 'nKbase' number of base
            # categories.
            Kbase = sorted(self.sampleCategories('base', nKbase))
            # sample from the set of novel categories 'nKnovel' number of novel
            # categories.
            Knovel = sorted(self.sampleCategories('novel', nKnovel))
        else:
            # sample from the set of base categories 'nKnovel' + 'nKbase' number
            # of categories.
            cats_ids = self.sampleCategories('base', nKnovel + nKbase)
            assert (len(cats_ids) == (nKnovel + nKbase))
            # Randomly pick 'nKnovel' number of fake novel categories and keep
            # the rest as base categories.
            random.shuffle(cats_ids)
            Knovel = sorted(cats_ids[:nKnovel])
            Kbase = sorted(cats_ids[nKnovel:])

        return Kbase, Knovel

    def sample_test_examples_for_base_categories(self, Kbase, nTestBase):
        """
        Sample `nTestBase` number of images from the `Kbase` categories.

        Args:
            Kbase: a list of length `nKbase` with the ids of the categories from
                where the images will be sampled.
            nTestBase: the total number of images that will be sampled.

        Returns:
            Tbase: a list of length `nTestBase` with 2-element tuples. The 1st
                element of each tuple is the image id that was sampled and the
                2nd elemend is its category label (which is in the range
                [0, len(Kbase)-1]).
        """
        Tbase = []
        if len(Kbase) > 0:
            # Sample for each base category a number images such that the total
            # number sampled images of all categories to be equal to `nTestBase`.
            KbaseIndices = np.random.choice(
                np.arange(len(Kbase)), size=nTestBase, replace=True)
            KbaseIndices, NumImagesPerCategory = np.unique(
                KbaseIndices, return_counts=True)

            for Kbase_idx, NumImages in zip(KbaseIndices, NumImagesPerCategory):
                imd_ids = self.sampleImageIdsFrom(
                    Kbase[Kbase_idx], sample_size=NumImages)
                Tbase += [(img_id, Kbase_idx) for img_id in imd_ids]

        assert (len(Tbase) == nTestBase)

        return Tbase

    def sample_train_and_test_examples_for_novel_categories(
            self, Knovel, nTestNovel, nExemplars, nKbase):
        """Samples train and test examples of the novel categories.

        Args:
            Knovel: a list with the ids of the novel categories.
            nTestNovel: the total number of test images that will be sampled
                from all the novel categories.
            nExemplars: the number of training examples per novel category that
                will be sampled.
            nKbase: the number of base categories. It is used as offset of the
                category index of each sampled image.

        Returns:
            Tnovel: a list of length `nTestNovel` with 2-element tuples. The
                1st element of each tuple is the image id that was sampled and
                the 2nd element is its category label (which is in the range
                [nKbase, nKbase + len(Knovel) - 1]).
            Exemplars: a list of length len(Knovel) * nExemplars of 2-element
                tuples. The 1st element of each tuple is the image id that was
                sampled and the 2nd element is its category label (which is in
                the range [nKbase, nKbase + len(Knovel) - 1]).
        """

        if len(Knovel) == 0:
            return [], []

        nKnovel = len(Knovel)
        Tnovel = []
        Exemplars = []
        assert ((nTestNovel % nKnovel) == 0)
        nEvalExamplesPerClass = int(nTestNovel / nKnovel)

        for Knovel_idx in range(len(Knovel)):
            imd_ids = self.sampleImageIdsFrom(
                Knovel[Knovel_idx],
                sample_size=(nEvalExamplesPerClass + nExemplars))

            imds_tnovel = imd_ids[:nEvalExamplesPerClass]
            imds_ememplars = imd_ids[nEvalExamplesPerClass:]

            Tnovel += [(img_id, nKbase + Knovel_idx) for img_id in imds_tnovel]
            Exemplars += [(img_id, nKbase + Knovel_idx) for img_id in imds_ememplars]
        # assert(len(Tnovel) == nTestNovel)
        # assert(len(Exemplars) == len(Knovel) * nExemplars)

        random.shuffle(Exemplars)

        return Tnovel, Exemplars

    def sample_episode(self):
        """Samples a training episode."""
        nKnovel = self.nKnovel
        nKbase = self.nKbase
        nTestNovel = self.nTestNovel
        nTestBase = self.nTestBase
        nExemplars = self.nExemplars

        Kbase, Knovel = self.sample_base_and_novel_categories(nKbase, nKnovel)
        # 从base类中采样test样本()--0
        Tbase = self.sample_test_examples_for_base_categories(Kbase, nTestBase)
        # 从novel类中采样train 和 test样本
        Tnovel, Exemplars = self.sample_train_and_test_examples_for_novel_categories(
            Knovel, nTestNovel, nExemplars, nKbase)

        # concatenate the base and novel category examples.
        Test = Tbase + Tnovel
        random.shuffle(Test)
        Kall = Kbase + Knovel

        return Exemplars, Test, Kall, nKbase

    def createExamplesTensorData(self, examples):
        """
        Creates the examples image and label tensor data.

        Args:
            examples: a list of 2-element tuples, each representing a
                train or test example. The 1st element of each tuple
                is the image id of the example and 2nd element is the
                category label of the example, which is in the range
                [0, nK - 1], where nK is the total number of categories
                (both novel and base).

        Returns:
            images: a tensor of shape [nExamples, Height, Width, 3] with the
                example images, where nExamples is the number of examples
                (i.e., nExamples = len(examples)).
            labels: a tensor of shape [nExamples] with the category label
                of each example.
        """
        #         print(self.dataset[img_idx].shape)
        #         print(self.dataset[img_idx][0].shape)
        images = torch.stack(
            [self.dataset[img_idx][0] for img_idx, _ in examples], dim=0)
        labels = torch.LongTensor([label for _, label in examples])
        # print('labels = ', labels)

        real_labels = torch.tensor([int(self.dataset[img_idx][1]) for img_idx, _ in examples])
        #         print('real_labels = ', real_labels)

        return images, labels, real_labels

    def get_iterator(self, epoch=0):
        rand_seed = epoch + 21
        random.seed(rand_seed)
        np.random.seed(rand_seed)

        def load_function(iter_idx):
            #
            Exemplars, Test, Kall, nKbase = self.sample_episode()
            if len(Test) > 0:
                Xt, Yt, RYt = self.createExamplesTensorData(Test)
            Kall = torch.LongTensor(Kall)
            if len(Exemplars) > 0:
                Xe, Ye, RYe = self.createExamplesTensorData(Exemplars)
                return Xe, Ye, RYe, Xe, Ye, RYe, Kall, nKbase
            else:
                return Xt, Yt, RYt, Xt, Yt, RYt, Kall, nKbase

        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(self.epoch_size), load=load_function)
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=(0 if self.is_eval_mode else self.num_workers),
            shuffle=(False if self.is_eval_mode else True))

        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return (self.epoch_size / self.batch_size)