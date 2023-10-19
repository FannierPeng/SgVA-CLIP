# Dataloader of Gidaris & Komodakis, CVPR 2018
# Adapted from:
# https://github.com/gidariss/FewShotWithoutForgetting/blob/master/dataloader.py
from __future__ import print_function

import os
import os.path
import numpy as np
import random
import pickle
import json
import math

import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchnet as tnt
import cv2
import pickle as pkl
import h5py

from PIL import Image
from PIL import ImageEnhance

from pdb import set_trace as breakpoint


# Set the appropriate paths of the datasets here.
_TIERED_IMAGENET_DATASET_DIR = '/userhome/CLIP/data/tiered-imagenet/'
class_idxes = {}
with open(_TIERED_IMAGENET_DATASET_DIR+'/class_names.txt', "r") as f:
    for i, line in enumerate(f.readlines()):
        line = line.strip('\n')  # 去掉列表中每一个元素的换行符
        # print(line)
        class_idxes[line] = i

label_idx_specific_train = ['Yorkshire terrier', 'space shuttle', 'drake', "plane, carpenter's plane, woodworking plane", 'mosquito net', 'sax, saxophone', 'container ship, containership, container vessel', 'patas, hussar monkey, Erythrocebus patas', 'cheetah, chetah, Acinonyx jubatus', 'submarine, pigboat, sub, U-boat', 'prison, prison house', 'can opener, tin opener', 'syringe', 'odometer, hodometer, mileometer, milometer', 'bassoon', 'Kerry blue terrier', 'scale, weighing machine', 'baseball', 'cassette player', 'shield, buckler', 'goldfinch, Carduelis carduelis', 'cornet, horn, trumpet, trump', 'flute, transverse flute', 'stopwatch, stop watch', 'basketball', 'brassiere, bra, bandeau', 'bulbul', 'steel drum', 'bolo tie, bolo, bola tie, bola', 'planetarium', 'stethoscope', 'proboscis monkey, Nasalis larvatus', 'guillotine', 'Scottish deerhound, deerhound', 'ocarina, sweet potato', 'Border terrier', 'capuchin, ringtail, Cebus capucinus', 'magnetic compass', 'alligator lizard', 'baboon', 'sundial', 'gibbon, Hylobates lar', 'grand piano, grand', 'Arabian camel, dromedary, Camelus dromedarius', 'basset, basset hound', 'corkscrew, bottle screw', 'miniskirt, mini', 'missile', 'hatchet', 'acoustic guitar', 'impala, Aepyceros melampus', 'parking meter', 'greenhouse, nursery, glasshouse', 'home theater, home theatre', 'hartebeest', 'hippopotamus, hippo, river horse, Hippopotamus amphibius', 'warplane, military plane', 'albatross, mollymawk', 'umbrella', 'shoe shop, shoe-shop, shoe store', 'suit, suit of clothes', 'pickelhaube', 'soccer ball', 'yawl', 'screwdriver', 'Madagascar cat, ring-tailed lemur, Lemur catta', 'garter snake, grass snake', 'bustard', 'tabby, tabby cat', 'airliner', 'tobacco shop, tobacconist shop, tobacconist', 'Italian greyhound', 'projector', 'bittern', 'rifle', 'pay-phone, pay-station', 'house finch, linnet, Carpodacus mexicanus', 'monastery', 'lens cap, lens cover', 'maillot, tank suit', 'canoe', 'letter opener, paper knife, paperknife', 'nail', 'guenon, guenon monkey', 'CD player', 'safety pin', 'harp', 'disk brake, disc brake', 'otterhound, otter hound', 'green mamba', 'violin, fiddle', 'American coot, marsh hen, mud hen, water hen, Fulica americana', 'ram, tup', 'jay', 'trench coat', 'Indian cobra, Naja naja', 'projectile, missile', 'schooner', 'magpie', 'Norwich terrier', 'cairn, cairn terrier', 'crossword puzzle, crossword', 'snow leopard, ounce, Panthera uncia', 'gong, tam-tam', 'library', 'swimming trunks, bathing trunks', 'Staffordshire bullterrier, Staffordshire bull terrier', 'Lakeland terrier', 'black stork, Ciconia nigra', 'king penguin, Aptenodytes patagonica', 'water ouzel, dipper', 'macaque', 'lynx, catamount', 'ping-pong ball', 'standard schnauzer', 'Australian terrier', 'stupa, tope', 'white stork, Ciconia ciconia', 'king snake, kingsnake', 'Airedale, Airedale terrier', 'banjo', 'Windsor tie', 'abaya', 'stole', 'vine snake', 'Bedlington terrier', 'langur', 'catamaran', 'sarong', 'spoonbill', 'boa constrictor, Constrictor constrictor', 'ruddy turnstone, Arenaria interpres', 'hognose snake, puff adder, sand viper', 'American chameleon, anole, Anolis carolinensis', 'rugby ball', 'black swan, Cygnus atratus', 'frilled lizard, Chlamydosaurus kingi', 'oscilloscope, scope, cathode-ray oscilloscope, CRO', 'ski mask', 'marmoset', 'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis', 'accordion, piano accordion, squeeze box', 'horned viper, cerastes, sand viper, horned asp, Cerastes cornutus', 'bookshop, bookstore, bookstall', 'Boston bull, Boston terrier', 'crane', 'junco, snowbird', 'silky terrier, Sydney silky', 'Egyptian cat', 'Irish terrier', 'leopard, Panthera pardus', 'sea snake', 'hog, pig, grunter, squealer, Sus scrofa', 'colobus, colobus monkey', 'chickadee', 'Scotch terrier, Scottish terrier, Scottie', 'digital watch', 'analog clock', 'zebra', 'American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier', 'European gallinule, Porphyrio porphyrio', 'lampshade, lamp shade', 'holster', 'jaguar, panther, Panthera onca, Felis onca', 'cleaver, meat cleaver, chopper', 'brambling, Fringilla montifringilla', 'orangutan, orang, orangutang, Pongo pygmaeus', 'combination lock', 'tile roof', 'borzoi, Russian wolfhound', 'water snake', 'knot', 'window shade', 'mosque', 'Walker hound, Walker foxhound', 'cardigan', 'warthog', 'whiptail, whiptail lizard', 'plow, plough', 'bluetick', 'poncho', 'shovel', 'sidewinder, horned rattlesnake, Crotalus cerastes', 'croquet ball', 'sorrel', 'airship, dirigible', 'goose', 'church, church building', 'titi, titi monkey', 'butcher shop, meat market', 'diamondback, diamondback rattlesnake, Crotalus adamanteus', 'common iguana, iguana, Iguana iguana', 'Saluki, gazelle hound', 'monitor', 'sunglasses, dark glasses, shades', 'flamingo', 'seat belt, seatbelt', 'Persian cat', 'gorilla, Gorilla gorilla', 'banded gecko', 'thatch, thatched roof', 'beagle', 'limpkin, Aramus pictus', 'jigsaw puzzle', 'rule, ruler', 'hammer', 'cello, violoncello', 'lab coat, laboratory coat', 'indri, indris, Indri indri, Indri brevicaudatus', 'vault', 'cellular telephone, cellular phone, cellphone, cell, mobile phone', 'whippet', 'siamang, Hylobates syndactylus, Symphalangus syndactylus', "loupe, jeweler's loupe", 'modem', 'lifeboat', 'dial telephone, dial phone', 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor', 'thimble', 'ibex, Capra ibex', 'lawn mower, mower', 'bell cote, bell cot', 'chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour', 'hair slide', 'apiary, bee house', 'harmonica, mouth organ, harp, mouth harp', 'green snake, grass snake', 'howler monkey, howler', 'digital clock', 'restaurant, eating house, eating place, eatery', 'miniature schnauzer', 'panpipe, pandean pipe, syrinx', 'pirate, pirate ship', 'window screen', 'binoculars, field glasses, opera glasses', 'Afghan hound, Afghan', 'cinema, movie theater, movie theatre, movie house, picture palace', 'liner, ocean liner', 'ringneck snake, ring-necked snake, ring snake', 'redshank, Tringa totanus', 'Siamese cat, Siamese', 'thunder snake, worm snake, Carphophis amoenus', 'boathouse', 'jersey, T-shirt, tee shirt', 'soft-coated wheaten terrier', 'scabbard', 'muzzle', 'Ibizan hound, Ibizan Podenco', 'tennis ball', 'padlock', 'kimono', 'redbone', 'wild boar, boar, Sus scrofa', 'dowitcher', 'oboe, hautboy, hautbois', 'electric guitar', 'trimaran', 'barometer', 'llama', 'robin, American robin, Turdus migratorius', 'maraca', 'feather boa, boa', 'Dandie Dinmont, Dandie Dinmont terrier', 'Lhasa, Lhasa apso', 'bow', 'punching bag, punch bag, punching ball, punchball', 'volleyball', 'Norfolk terrier', 'Gila monster, Heloderma suspectum', 'fire screen, fireguard', 'hourglass', 'chimpanzee, chimp, Pan troglodytes', 'birdhouse', 'Sealyham terrier, Sealyham', 'Tibetan terrier, chrysanthemum dog', 'palace', 'wreck', 'overskirt', 'pelican', 'French horn, horn', 'tiger cat', 'barbershop', 'revolver, six-gun, six-shooter', 'Irish wolfhound', 'lion, king of beasts, Panthera leo', 'fur coat', 'ox', 'cuirass', 'grocery store, grocery, food market, market', 'hoopskirt, crinoline', 'spider monkey, Ateles geoffroyi', 'tiger, Panthera tigris', 'bloodhound, sleuthhound', 'red-backed sandpiper, dunlin, Erolia alpina', 'drum, membranophone, tympan', 'radio telescope, radio reflector', 'West Highland white terrier', 'bow tie, bow-tie, bowtie', 'golf ball', 'barn', 'binder, ring-binder', 'English foxhound', 'bison', 'screw', 'assault rifle, assault gun', 'diaper, nappy, napkin', 'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis', 'Weimaraner', 'computer keyboard, keypad', 'black-and-tan coonhound', 'little blue heron, Egretta caerulea', 'breastplate, aegis, egis', 'gasmask, respirator, gas helmet', 'aircraft carrier, carrier, flattop, attack aircraft carrier', 'iPod', 'organ, pipe organ', 'wall clock', 'rock python, rock snake, Python sebae', 'squirrel monkey, Saimiri sciureus', 'bikini, two-piece', 'water buffalo, water ox, Asiatic buffalo, Bubalus bubalis', 'upright, upright piano', 'chime, bell, gong', 'confectionery, confectionary, candy store', 'indigo bunting, indigo finch, indigo bird, Passerina cyanea', 'green lizard, Lacerta viridis', 'Norwegian elkhound, elkhound', 'dome', 'buckle', 'giant schnauzer', 'jean, blue jean, denim', 'wire-haired fox terrier', 'African chameleon, Chamaeleo chamaeleon', 'trombone', 'oystercatcher, oyster catcher', 'sweatshirt', 'American egret, great white heron, Egretta albus', 'marimba, xylophone', 'gazelle', 'red-breasted merganser, Mergus serrator', 'tape player', 'speedboat', 'gondola', 'night snake, Hypsiglena torquata', 'cannon', "plunger, plumber's helper", 'balloon', 'toyshop', 'agama', 'fireboat', 'bakery, bakeshop, bakehouse']
label_idx_specific_val = ['cab, hack, taxi, taxicab', 'jeep, landrover', 'English setter', 'flat-coated retriever', 'bassinet', 'sports car, sport car', 'golfcart, golf cart', 'clumber, clumber spaniel', 'puck, hockey puck', 'reel', 'Welsh springer spaniel', 'car wheel', 'wardrobe, closet, press', 'go-kart', 'switch, electric switch, electrical switch', 'crib, cot', 'laptop, laptop computer', 'thresher, thrasher, threshing machine', 'web site, website, internet site, site', 'English springer, English springer spaniel', 'iron, smoothing iron', 'Gordon setter', 'Labrador retriever', 'Irish water spaniel', 'amphibian, amphibious vehicle', 'file, file cabinet, filing cabinet', 'harvester, reaper', 'convertible', 'paddlewheel, paddle wheel', 'microwave, microwave oven', 'swing', 'chiffonier, commode', 'desktop computer', 'gas pump, gasoline pump, petrol pump, island dispenser', 'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon', 'carousel, carrousel, merry-go-round, roundabout, whirligig', "potter's wheel", 'folding chair', 'fire engine, fire truck', 'slide rule, slipstick', 'vizsla, Hungarian pointer', 'waffle iron', 'trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi', 'toilet seat', 'medicine chest, medicine cabinet', 'Brittany spaniel', 'Chesapeake Bay retriever', 'cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM', 'moped', 'Model T', 'bookcase', 'ambulance', 'German short-haired pointer', 'dining table, board', 'minivan', 'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria', 'entertainment center', 'throne', 'desk', 'notebook, notebook computer', 'snowplow, snowplough', 'cradle', 'abacus', 'hand-held computer, hand-held microcomputer', 'Dutch oven', 'toaster', 'barber chair', 'vending machine', 'four-poster', 'rotisserie', 'hook, claw', 'vacuum, vacuum cleaner', 'pickup, pickup truck', 'table lamp', 'rocking chair, rocker', 'prayer rug, prayer mat', 'moving van', 'studio couch, day bed', 'racer, race car, racing car', 'park bench', 'Irish setter, red setter', 'refrigerator, icebox', 'china cabinet, china closet', 'cocker spaniel, English cocker spaniel, cocker', 'radiator', 'Sussex spaniel', 'hand blower, blow dryer, blow drier, hair dryer, hair drier', 'slot, one-armed bandit', 'golden retriever', 'curly-coated retriever', 'limousine, limo', 'washer, automatic washer, washing machine', 'garbage truck, dustcart', 'dishwasher, dish washer, dishwashing machine', 'pinwheel', 'espresso maker', 'tow truck, tow car, wrecker']
label_idx_specific_test = ['Siberian husky', 'dung beetle', 'jackfruit, jak, jack', 'miniature pinscher', 'tiger shark, Galeocerdo cuvieri', 'weevil', 'goldfish, Carassius auratus', 'schipperke', 'Tibetan mastiff', 'orange', 'whiskey jug', 'hammerhead, hammerhead shark', 'bull mastiff', 'eggnog', 'bee', 'tench, Tinca tinca', 'chocolate sauce, chocolate syrup', "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk", 'zucchini, courgette', 'kelpie', 'stone wall', 'butternut squash', 'mushroom', 'Old English sheepdog, bobtail', 'dam, dike, dyke', 'picket fence, paling', 'espresso', 'beer bottle', 'plate', 'dough', 'sandbar, sand bar', 'boxer', 'bathtub, bathing tub, bath, tub', 'beaker', 'bucket, pail', 'Border collie', 'sturgeon', 'worm fence, snake fence, snake-rail fence, Virginia fence', 'seashore, coast, seacoast, sea-coast', 'long-horned beetle, longicorn, longicorn beetle', 'turnstile', 'groenendael', 'vase', 'teapot', 'water tower', 'strawberry', 'burrito', 'cauliflower', 'volcano', 'valley, vale', 'head cabbage', 'tub, vat', 'lacewing, lacewing fly', 'coral reef', 'hot pot, hotpot', 'custard apple', 'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus', 'cricket', 'pill bottle', 'walking stick, walkingstick, stick insect', 'promontory, headland, head, foreland', 'malinois', 'pizza, pizza pie', 'malamute, malemute, Alaskan malamute', 'kuvasz', 'trifle', 'fig', 'komondor', 'ant, emmet, pismire', 'electric ray, crampfish, numbfish, torpedo', 'Granny Smith', 'cockroach, roach', 'stingray', 'red wine', 'Saint Bernard, St Bernard', 'ice lolly, lolly, lollipop, popsicle', 'bell pepper', 'cup', 'pomegranate', 'Appenzeller', 'hay', 'EntleBucher', 'sulphur butterfly, sulfur butterfly', 'mantis, mantid', 'Bernese mountain dog', 'banana', 'water jug', 'cicada, cicala', 'barracouta, snoek', 'washbasin, handbasin, washbowl, lavabo, wash-hand basin', 'wine bottle', 'Rottweiler', 'briard', 'puffer, pufferfish, blowfish, globefish', 'ground beetle, carabid beetle', 'Bouvier des Flandres, Bouviers des Flandres', 'chainlink fence', 'damselfly', 'grasshopper, hopper', 'carbonara', 'German shepherd, German shepherd dog, German police dog, alsatian', 'guacamole', 'leaf beetle, chrysomelid', 'caldron, cauldron', 'fly', 'bannister, banister, balustrade, balusters, handrail', 'spaghetti squash', 'coffee mug', 'gar, garfish, garpike, billfish, Lepisosteus osseus', 'barrel, cask', 'eel', 'rain barrel', 'coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch', 'water bottle', 'menu', 'tiger beetle', 'Great Dane', 'rock beauty, Holocanthus tricolor', 'anemone fish', 'mortar', 'Eskimo dog, husky', 'affenpinscher, monkey pinscher, monkey dog', 'breakwater, groin, groyne, mole, bulwark, seawall, jetty', 'artichoke, globe artichoke', 'broccoli', 'French bulldog', 'coffeepot', 'cliff, drop, drop-off', 'ladle', 'sliding door', 'leafhopper', 'collie', 'Doberman, Doberman pinscher', 'pitcher, ewer', 'admiral', 'cabbage butterfly', 'geyser', 'cheeseburger', 'grille, radiator grille', 'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle', 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias', 'pineapple, ananas', 'cardoon', 'pop bottle, soda bottle', 'lionfish', 'cucumber, cuke', 'face powder', 'Shetland sheepdog, Shetland sheep dog, Shetland', 'ringlet, ringlet butterfly', 'Greater Swiss Mountain dog', 'alp', 'consomme', 'potpie', 'acorn squash', 'ice cream, icecream', 'lakeside, lakeshore', 'hotdog, hot dog, red hot', 'rhinoceros beetle', 'lycaenid, lycaenid butterfly', 'lemon']
label_idx_specific = label_idx_specific_train+label_idx_specific_val+label_idx_specific_test

def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


def load_data(file):
    try:
        with open(file, 'rb') as fo:
            data = pickle.load(fo)
        return data
    except:
        with open(file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        return data

class tieredImageNet(data.Dataset):
    def __init__(self, phase='train', do_not_use_random_transf=False, use_base=True):

        assert(phase=='train' or phase=='val' or phase=='test')
        self.phase = phase
        self.name = 'tieredImageNet_' + phase

        print('Loading tiered ImageNet dataset - phase {0}'.format(phase))
        file_train_categories_train_phase = os.path.join(
            _TIERED_IMAGENET_DATASET_DIR,
            'train_images.npz')
        label_train_categories_train_phase = os.path.join(
            _TIERED_IMAGENET_DATASET_DIR,
            'train_labels.pkl')
        file_train_categories_val_phase = os.path.join(
            _TIERED_IMAGENET_DATASET_DIR,
            'train_images.npz')
        label_train_categories_val_phase = os.path.join(
            _TIERED_IMAGENET_DATASET_DIR,
            'train_labels.pkl')
        file_train_categories_test_phase = os.path.join(
            _TIERED_IMAGENET_DATASET_DIR,
            'train_images.npz')
        label_train_categories_test_phase = os.path.join(
            _TIERED_IMAGENET_DATASET_DIR,
            'train_labels.pkl')

        file_val_categories_val_phase = os.path.join(
            _TIERED_IMAGENET_DATASET_DIR,
            'val_images.npz')
        label_val_categories_val_phase = os.path.join(
            _TIERED_IMAGENET_DATASET_DIR,
            'val_labels.pkl')
        file_test_categories_test_phase = os.path.join(
            _TIERED_IMAGENET_DATASET_DIR,
            'test_images.npz')
        label_test_categories_test_phase = os.path.join(
            _TIERED_IMAGENET_DATASET_DIR,
            'test_labels.pkl')
        
        if self.phase=='train':
            # During training phase we only load the training phase images
            # of the training categories (aka base categories).
            data_train = load_data(label_train_categories_train_phase)
            #self.data = data_train['data']
            self.labels = data_train['labels']
            print('train_labels=', self.labels)
            self.data = np.load(file_train_categories_train_phase)['images']#np.array(load_data(file_train_categories_train_phase))
            #self.labels = load_data(file_train_categories_train_phase)#data_train['labels']

            self.label2ind = buildLabelIndex(self.labels)
            print('train_label2ind=', self.label2ind)
            self.labelIds = sorted(self.label2ind.keys())
            print('train_labelIds=', self.labelIds)
            self.num_cats = len(self.labelIds)
            print('train_num_cats=', self.num_cats)
            self.labelIds_base = self.labelIds
            print('train_labelIds_base=', self.labelIds_base)
            self.num_cats_base = len(self.labelIds_base)
            print('train_num_cats_base=', self.num_cats_base)

        elif self.phase=='val' or self.phase=='test':
            if self.phase=='test':
                # load data that will be used for evaluating the recognition
                # accuracy of the base categories.
                data_base = load_data(label_train_categories_test_phase)
                data_base_images = np.load(file_train_categories_test_phase)['images']
                
                # load data that will be use for evaluating the few-shot recogniton
                # accuracy on the novel categories.
                data_novel = load_data(label_test_categories_test_phase)
                data_novel_images = np.load(file_test_categories_test_phase)['images']
            else: # phase=='val'
                # load data that will be used for evaluating the recognition
                # accuracy of the base categories.
                data_base = load_data(label_train_categories_val_phase)
                data_base_images = np.load(file_train_categories_val_phase)['images']
                #print (data_base_images)
                #print (data_base_images.shape)
                # load data that will be use for evaluating the few-shot recogniton
                # accuracy on the novel categories.
                data_novel = load_data(label_val_categories_val_phase)
                data_novel_images = np.load(file_val_categories_val_phase)['images']
            if use_base:
                self.data = np.concatenate(
                    [data_base_images, data_novel_images], axis=0)
                self.labels = data_base['labels'] + data_novel['labels']
            else:
                self.data = data_novel_images
                self.labels = data_novel['labels']

            print('test_labels=', self.labels)
            self.label2ind = buildLabelIndex(self.labels)
            print('test_label2ind=', self.label2ind)
            self.labelIds = sorted(self.label2ind.keys())
            print('test_labelIds=', self.labelIds)
            self.num_cats = len(self.labelIds)
            print('test_num_cats=', self.num_cats)

            self.labelIds_base = buildLabelIndex(data_base['labels']).keys()
            self.labelIds_novel = buildLabelIndex(data_novel['labels']).keys()
            self.num_cats_base = len(self.labelIds_base)
            self.num_cats_novel = len(self.labelIds_novel)
            print('test_labelIds_base=', self.labelIds_base)
            print('test_labelIds_novel=', self.labelIds_novel)
            print('test_num_cats_base=', self.num_cats_base)
            print('test_num_cats_novel=', self.num_cats_novel)
            intersection = set(self.labelIds_base) & set(self.labelIds_novel)
            print('intersection=',intersection)
            assert(len(intersection) == 0)
        else:
            raise ValueError('Not valid phase {0}'.format(self.phase))

        mean_pix = [x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]
        std_pix = [x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        if (self.phase=='test' or self.phase=='val') or (do_not_use_random_transf==True):
            self.transform = transforms.Compose([
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomCrop(84, padding=8),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


class tieredImageNetPP(data.Dataset):
    def __init__(self, phase='train', do_not_use_random_transf=False, use_base=True):

        assert (phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.name = 'tieredImageNet_' + phase

        print('Loading tiered ImageNet dataset - phase {0}'.format(phase))
        file_train_categories_train_phase = os.path.join(
            _TIERED_IMAGENET_DATASET_DIR,
            'train_images_png.pkl')
        label_train_categories_train_phase = os.path.join(
            _TIERED_IMAGENET_DATASET_DIR,
            'train_labels.pkl')
        file_train_categories_val_phase = os.path.join(
            _TIERED_IMAGENET_DATASET_DIR,
            'train_images_png.pkl')
        label_train_categories_val_phase = os.path.join(
            _TIERED_IMAGENET_DATASET_DIR,
            'train_labels.pkl')
        file_train_categories_test_phase = os.path.join(
            _TIERED_IMAGENET_DATASET_DIR,
            'train_images_png.pkl')
        label_train_categories_test_phase = os.path.join(
            _TIERED_IMAGENET_DATASET_DIR,
            'train_labels.pkl')

        file_val_categories_val_phase = os.path.join(
            _TIERED_IMAGENET_DATASET_DIR,
            'val_images_png.pkl')
        label_val_categories_val_phase = os.path.join(
            _TIERED_IMAGENET_DATASET_DIR,
            'val_labels.pkl')
        file_test_categories_test_phase = os.path.join(
            _TIERED_IMAGENET_DATASET_DIR,
            'test_images_png.pkl')
        label_test_categories_test_phase = os.path.join(
            _TIERED_IMAGENET_DATASET_DIR,
            'test_labels.pkl')

        if self.phase == 'train':
            # During training phase we only load the training phase images
            # of the training categories (aka base categories).

            with open(label_train_categories_train_phase, "rb") as f:
                data = pkl.load(f)
                label_specific = list(data["label_specific"])
                label_specific_str = data["label_specific_str"]

            self.labels = label_specific
            f_images = open(file_train_categories_train_phase, 'rb')

            self.data = pkl.load(f_images)
            # print(self.data[0].shape)
            self.label_specific_str = label_specific_str
            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
            self.labelIds_base = self.labelIds
            self.num_cats_base = len(self.labelIds_base)


            # print('label_specific_str =', self.label_specific_str)
            # print('train_label2ind=', self.label2ind)
            # print('train_labelIds=', self.labelIds)  #[0-350]
            # print('train_num_cats=', self.num_cats)  #351
            # print('train_labelIds_base=', self.labelIds_base)  #[0-350]
            # print('train_num_cats_base=', self.num_cats_base)  #351

        elif self.phase == 'val' or self.phase == 'test':
            if self.phase == 'test':
                # load data that will be used for evaluating the recognition
                # accuracy of the base categories.
                with open(label_train_categories_test_phase, "rb") as f:
                    data_base = pkl.load(f)
                f_images = open(file_train_categories_test_phase, 'rb')
                data_base_images = pkl.load(f_images)
                # load data that will be use for evaluating the few-shot recogniton
                # accuracy on the novel categories.
                with open(label_test_categories_test_phase, "rb") as f:
                    data_novel = pkl.load(f)
                f_images = open(file_test_categories_test_phase, 'rb')
                data_novel_images = pkl.load(f_images)

            else:  # phase=='val'
                # load data that will be used for evaluating the recognition
                # accuracy of the base categories.
                with open(label_train_categories_val_phase, "rb") as f:
                    data_base = pkl.load(f)
                f_images = open(file_train_categories_val_phase, 'rb')
                data_base_images = pkl.load(f_images)
                # print (data_base_images)
                # print (data_base_images.shape)
                # load data that will be use for evaluating the few-shot recogniton
                # accuracy on the novel categories.
                with open(label_val_categories_val_phase, "rb") as f:
                    data_novel = pkl.load(f)
                f_images = open(file_val_categories_val_phase, 'rb')
                data_novel_images = pkl.load(f_images)

            if use_base:
                self.data = np.concatenate(
                    [data_base_images, data_novel_images], axis=0)
                if self.phase == 'test':
                    self.labels = list(data_base["label_specific"]) + list(map(lambda x:x+448, list(data_novel["label_specific"])))
                else:
                    self.labels = list(data_base["label_specific"]) + list(map(lambda x:x+351, list(data_novel["label_specific"])))
                self.label_specific_str = data_base["label_specific_str"] + data_novel["label_specific_str"]
            else:
                self.data = data_novel_images
                if self.phase == 'test':
                    self.labels = list(map(lambda x:x+448, list(data_novel["label_specific"])))
                else:
                    self.labels = list(map(lambda x:x+351, list(data_novel["label_specific"])))
                self.label_specific_str = data_novel["label_specific_str"]

            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
            # print('test_labels=', self.labels)
            # print('test_label2ind=', self.label2ind) #img_id号，6位数
            # print('test_labelIds=', self.labelIds) #test_labelIds=[448-607] val:[351-447]
            # print('test_num_cats=', self.num_cats) #test 160  val 97


            self.labelIds_base = buildLabelIndex(list(data_base["label_specific"])).keys()
            if self.phase == 'test':
                self.labelIds_novel = buildLabelIndex(list(map(lambda x:x+448, list(data_novel["label_specific"])))).keys()
            else:
                self.labelIds_novel = buildLabelIndex(list(map(lambda x:x+351, list(data_novel["label_specific"])))).keys()
            self.num_cats_base = len(self.labelIds_base)
            self.num_cats_novel = len(self.labelIds_novel)
            intersection = set(self.labelIds_base) & set(self.labelIds_novel)

            # print('test_labelIds_base=', self.labelIds_base)  #test_labelIds_base=[0-350]  val:[0-350]
            # print('test_labelIds_novel=', self.labelIds_novel) #test_labelIds_novel=[448-607]  val:[351-447]
            # print('test_num_cats_base=', self.num_cats_base) #test 351    #val 351
            # print('test_num_cats_novel=', self.num_cats_novel) #test 160   #val 97
            # print('intersection=', intersection)
            assert (len(intersection) == 0)
        else:
            raise ValueError('Not valid phase {0}'.format(self.phase))

        # mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        # std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        mean_pix = [0.48145466, 0.4578275, 0.40821073]  # CLIP
        std_pix = [0.26862954, 0.26130258, 0.27577711]  # CLIP
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        if (self.phase == 'test' or self.phase == 'val') or (do_not_use_random_transf == True):
            self.transform = transforms.Compose([
                transforms.Resize(224, interpolation=Image.BICUBIC),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(224, interpolation=Image.BICUBIC),
                transforms.RandomCrop(224, padding=8),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        im, label = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        im = cv2.imdecode(im, 1)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im.transpose(2, 0, 1)
        img = Image.fromarray(im)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


class FewShotDataloader():
    def __init__(self,
                 dataset,
                 nKnovel=5, # number of novel categories.
                 nKbase=-1, # number of base categories.
                 nExemplars=1, # number of training examples per novel category.
                 nTestNovel=15*5, # number of test examples for all the novel categories.
                 nTestBase=15*5, # number of test examples for all the base categories.
                 batch_size=1, # number of training episodes per batch.
                 num_workers=4,
                 epoch_size=2000, # number of batches per epoch.
                 sample_categeries = None,
                 ):

        self.dataset = dataset
        self.phase = self.dataset.phase
        self.sample_categeries = sample_categeries

        #train时 max_possible_nKnovel=64，nKnovel=novel类数目=5， nKbase=0
        #val时 max_possible_nKnovel=16，nKnovel=novel类数目=5， nKbase=0
        max_possible_nKnovel = (self.dataset.num_cats_base if self.phase=='train'
                                else self.dataset.num_cats_novel)
        assert(nKnovel >= 0 and nKnovel <= max_possible_nKnovel)
        self.nKnovel = nKnovel

        max_possible_nKbase = self.dataset.num_cats_base
        nKbase = nKbase if nKbase >= 0 else max_possible_nKbase
        if self.phase=='train' and nKbase > 0:
            nKbase -= self.nKnovel   #train时nKbase需要减去nKnovel
            max_possible_nKbase -= self.nKnovel

        assert(nKbase >= 0 and nKbase <= max_possible_nKbase)
        self.nKbase = nKbase

        self.nExemplars = nExemplars
        self.nTestNovel = nTestNovel
        self.nTestBase = nTestBase
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.num_workers = num_workers
        self.is_eval_mode = (self.phase=='test') or (self.phase=='val')

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
        assert(cat_id in self.dataset.label2ind)
        assert(len(self.dataset.label2ind[cat_id]) >= sample_size)
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
        if cat_set=='base':
            labelIds = self.dataset.labelIds_base
        elif cat_set=='novel':
            labelIds = self.dataset.labelIds_novel
        else:
            raise ValueError('Not recognized category set {}'.format(cat_set))

        assert(len(labelIds) >= sample_size)
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
            assert(nKnovel <= self.dataset.num_cats_novel)
            # sample from the set of base categories 'nKbase' number of base
            # categories.
            Kbase = sorted(self.sampleCategories('base', nKbase))
            # sample from the set of novel categories 'nKnovel' number of novel
            # categories.
            Knovel = sorted(self.sampleCategories('novel', nKnovel))
        else:
            # sample from the set of base categories 'nKnovel' + 'nKbase' number
            # of categories.
            cats_ids = self.sampleCategories('base', nKnovel+nKbase)
            assert(len(cats_ids) == (nKnovel+nKbase))
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

        assert(len(Tbase) == nTestBase)

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
        assert((nTestNovel % nKnovel) == 0)
        nEvalExamplesPerClass = int(nTestNovel / nKnovel)

        for Knovel_idx in range(len(Knovel)):
            imd_ids = self.sampleImageIdsFrom(
                Knovel[Knovel_idx],
                sample_size=(nEvalExamplesPerClass + nExemplars))

            imds_tnovel = imd_ids[:nEvalExamplesPerClass]
            imds_ememplars = imd_ids[nEvalExamplesPerClass:]

            Tnovel += [(img_id, nKbase+Knovel_idx) for img_id in imds_tnovel]
            Exemplars += [(img_id, nKbase+Knovel_idx) for img_id in imds_ememplars]
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
        images = torch.stack(
            [self.dataset[img_idx][0] for img_idx, _ in examples], dim=0)
        labels = torch.LongTensor([label for _, label in examples])
        # print('labels = ', labels)

        real_labels = torch.tensor([class_idxes[label_idx_specific[int(self.dataset[img_idx][1])]] for img_idx, _ in examples])


        return images, labels, real_labels

    def get_iterator(self, epoch=0):
        rand_seed = epoch+21
        random.seed(rand_seed)
        np.random.seed(rand_seed)
        def load_function(iter_idx):
            #
            Exemplars, Test, Kall, nKbase = self.sample_episode()
            Xt, Yt, RYt = self.createExamplesTensorData(Test)
            Kall = torch.LongTensor(Kall)
            if len(Exemplars) > 0:
                Xe, Ye, RYe = self.createExamplesTensorData(Exemplars)
                return Xe, Ye, RYe, Xt, Yt, RYt, Kall, nKbase
            else:
                return Xt, Yt, RYt, Kall, nKbase

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