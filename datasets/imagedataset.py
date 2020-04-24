import os

import numpy as np
from csl_common.utils import ds_utils, transforms as csl_tf, geometry
from csl_common.utils.image_loader import CachedCropLoader
from torch.utils import data as td
from torchvision import transforms as tf
import torchvision.datasets as tdv


class ImageDataset(tdv.VisionDataset):

    def __init__(self, root, fullsize_img_dir, image_size, base_folder='', cache_root=None, train=True,
                 transform=None, crop_type='tight', color=True, start=None, max_samples=None, deterministic=None, use_cache=True,
                 with_occlusions=False, test_split='fullset', crop_source='bb_ground_truth', daug=0,
                 crop_border_mode='black', crop_dir='crops', median_blur_crop=False, **kwargs):

        print("Setting up dataset {}...".format(self.__class__.__name__))

        self.image_size = image_size
        self.crop_size = ds_utils.get_crop_size(image_size)
        self.crop_dir = crop_dir
        self.margin = self.crop_size - self.image_size

        self.test_split = test_split
        self.split = 'train' if train else self.test_split
        self.train = train
        self.use_cache = use_cache
        self.crop_source = crop_source
        self.crop_type = crop_type
        self.start = start
        self.max_samples = max_samples
        self.daug = daug
        self.with_occlusions = with_occlusions

        self.deterministic = deterministic
        if self.deterministic is None:
            self.deterministic = not self.train

        self.fullsize_img_dir = fullsize_img_dir

        self.root = root
        self.cache_root = cache_root if cache_root is not None else self.root
        self.base_folder = base_folder

        self.color = color

        self.annotations = self._load_annotations(self.split)

        self._init()
        self._select_index_range()

        transforms = [csl_tf.CenterCrop(image_size)]
        transforms += [csl_tf.ToTensor()]
        transforms += [csl_tf.Normalize([0.518, 0.418, 0.361], [1, 1, 1])]  # VGGFace(2) means
        self.crop_to_tensor = tf.Compose(transforms)

        self.image_loader = CachedCropLoader(fullsize_img_dir,
                                             self.cropped_img_dir,
                                             img_size=self.image_size,
                                             margin=self.margin,
                                             use_cache=self.use_cache,
                                             crop_type=crop_type,
                                             border_mode=crop_border_mode,
                                             median_blur_crop=median_blur_crop)

        super().__init__(root,
                         # transform=ds_utils.build_transform(deterministic, color, daug),
                         transform=transform,
                         target_transform=self.build_target_transform())

    @property
    def feature_dir(self):
        return os.path.join(self.cache_root, self.base_folder, 'features')

    @property
    def cropped_img_dir(self):
        return os.path.join(self.cache_root, self.crop_dir, self.crop_source)

    def build_target_transform(self):
        transform = None
        if self.with_occlusions:
            transform = tf.Compose([csl_tf.RandomOcclusion()])
        return transform

    def _init(self):
        pass

    def _load_annotations(self, split):
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        raise NotImplementedError

    def filter_labels(self, label_dict):
        import collections
        print("Applying filter to labels: {}".format(label_dict))
        for k, v in label_dict.items():
            if isinstance(v, collections.Sequence):
                selected_rows = self.annotations[k].isin(v)
            else:
                selected_rows = self.annotations[k] == v
            self.annotations = self.annotations[selected_rows]
        print("  Number of images: {}".format(len(self.annotations)))

    def _select_index_range(self):
        st,nd = 0, None
        if self.start is not None:
            st = self.start
        if self.max_samples is not None:
            nd = st + self.max_samples
        self.annotations = self.annotations[st:nd]

    @property
    def labels(self):
        return NotImplemented

    def get_crop_extend_factors(self):
        return 0, 0

    def get_adjusted_bounding_box(self, l, t, w, h):
        r, b = l + w, t + h

        # enlarge bounding box
        if t > b:
            t, b = b, t
        h = b-t
        assert(h >= 0)
        extend_top, extend_bottom = self.get_crop_extend_factors()
        t_new, b_new = int(t - extend_top * h), int(b + extend_bottom * h)

        h_new = b_new - t_new
        # set width of bbox same as height
        size = w if w > h_new else h_new
        cx = (r + l) / 2
        cy = (t_new + b_new) / 2
        l_new, r_new = cx - size/2, cx + size/2
        t_new, b_new = cy - size/2, cy + size/2

        # in case right eye is actually left of right eye...
        if l_new > r_new:
            l_new, r_new = r_new, l_new

        # extend area by crop border margins
        bbox = np.array([l_new, t_new, r_new, b_new], dtype=np.float32)
        scalef = self.crop_size / self.image_size
        bbox_crop = geometry.scaleBB(bbox, scalef, scalef, typeBB=2)
        return bbox_crop


    def get_sample(self, filename, bb=None, landmarks_for_crop=None, id=None):

        try:
            image  = self.image_loader.load_crop(filename, bb=bb, id=id)
        except:
            print('Could not load image {}'.format(filename))
            raise

        sample = self.transform(image)
        target = self.target_transform(sample.copy()) if self.target_transform else None

        if self.crop_type != 'fullsize':
            sample = self.crop_to_tensor(sample)
            if target is not None:
                target = self.crop_to_tensor(target)

        sample = ({ 'image': sample,
                    'fnames': filename,
                    'bb': bb if bb is not None else [0,0,0,0]})

        if target is not None:
            sample['target'] = target

        return sample