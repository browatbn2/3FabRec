import os
import numpy as np

import csl_common
from csl_common.utils import geometry, ds_utils
import csl_common.utils.transforms as csl_tf
from csl_common.utils.image_loader import CachedCropLoader

from constants import *
import torch.utils.data as td
from torchvision import transforms as tf
from landmarks import lmutils


class ImageDataset(td.Dataset):

    def __init__(self, root, fullsize_img_dir, image_size, base_folder='', cache_root=None, train=True,
                 crop_type='tight', color=True, start=None, max_samples=None, deterministic=None, use_cache=True,
                 with_occlusions=False, test_split='fullset', crop_source='bb_ground_truth', daug=0,
                 crop_border_mode='black', crop_dir='crops', median_blur_crop=False, **kwargs):

        print("Setting up dataset {}".format(self.__class__.__name__))

        self.image_size = image_size
        self.crop_size = ds_utils.get_crop_size(image_size)
        self.crop_dir = crop_dir
        self.margin = self.crop_size - self.image_size

        self.test_split = test_split
        self.split = 'train' if train else self.test_split
        self.train = train
        self.mode = TRAIN if train else VAL
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

        self.transform = ds_utils.build_transform(self.deterministic, self.color, daug)
        self.target_transform = self.build_target_transform()

        self.annotations = self._load_annotations(self.split)

        self.init()
        self.limit_sample_count()

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

        # print("  Number of images: {}".format(len(self.annotations)))

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

    def init(self):
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

    def limit_sample_count(self):
        # print("Limiting number of samples...")
        st,nd = 0, None
        if self.start is not None:
            st = self.start
        if self.max_samples is not None:
            nd = st + self.max_samples
        self.annotations = self.annotations[st:nd]
        # print("  Number of images: {}".format(len(self.annotations)))

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



class FaceDataset(ImageDataset):
    NUM_LANDMARKS = 68
    LANDMARKS_ONLY_OUTLINE = list(range(17))
    LANDMARKS_NO_OUTLINE = list(range(17,NUM_LANDMARKS))
    ALL_LANDMARKS =  LANDMARKS_ONLY_OUTLINE + LANDMARKS_NO_OUTLINE

    def __init__(self, return_landmark_heatmaps=False, landmark_sigma=9, align_face_orientation=False, **kwargs):
        super().__init__(**kwargs)
        self.return_landmark_heatmaps = return_landmark_heatmaps
        self.landmark_sigma = landmark_sigma
        self.empty_landmarks = np.zeros((self.NUM_LANDMARKS, 2), dtype=np.float32)
        self.align_face_orientation = align_face_orientation

    @staticmethod
    def _get_expression(sample):
        return np.array([[0,0,0]], dtype=np.float32)

    @staticmethod
    def _get_identity(sample):
        return -1

    def _crop_landmarks(self, lms):
         return self.image_loader._cropper.apply_to_landmarks(lms)[0]

    def get_sample(self, filename, bb=None, landmarks_for_crop=None, id=None, landmarks_to_return=None):
        try:
            crop_mode = 'landmarks' if landmarks_for_crop is not None else 'bounding_box'
            crop_params = {'landmarks': landmarks_for_crop,
                           'bb': bb,
                           'id': id,
                           'aligned': self.align_face_orientation,
                           'mode': crop_mode}
            image = self.image_loader.load_crop(filename, **crop_params)
        except:
            print('Could not load image {}'.format(filename))
            raise

        relative_landmarks = self._crop_landmarks(landmarks_to_return) \
            if landmarks_to_return is not None else self.empty_landmarks

        # self.show_landmarks(image, landmarks)

        sample = {'image': image,
                  'landmarks': relative_landmarks,
                  'pose': np.zeros(3, dtype=np.float32)}

        sample = self.transform(sample)
        target = self.target_transform(sample) if self.target_transform else None

        # self.show_landmarks(sample['image'], sample['landmarks'])

        if self.crop_type != 'fullsize':
            sample = self.crop_to_tensor(sample)
            if target is not None:
                target = self.crop_to_tensor(target)

        sample.update({
            'fnames': filename,
            'bb': bb if bb is not None else [0,0,0,0],
            # 'expression':self._get_expression(sample),
            # 'id': self._get_identity(sample),
        })

        if target is not None:
            sample['target'] = target

        if self.return_landmark_heatmaps and self.crop_type != 'fullsize':
            from landmarks import lmconfig as lmcfg
            heatmap_size = lmcfg.HEATMAP_SIZE
            scaled_landmarks = sample['landmarks'] * (heatmap_size / self.image_size)
            sample['lm_heatmaps'] = lmutils.create_landmark_heatmaps(scaled_landmarks, self.landmark_sigma,
                                                                     self.ALL_LANDMARKS, heatmap_size)
        return sample


    def show_landmarks(self, img, landmarks):
        import cv2
        for lm in landmarks:
            lm_x, lm_y = lm[0], lm[1]
            cv2.circle(img, (int(lm_x), int(lm_y)), 3, (0, 0, 255), -1)
        cv2.imshow('landmarks', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)