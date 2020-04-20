import os
from skimage import io
import numpy as np
import cv2
from csl_common.utils import cropping
from csl_common.utils.io_utils import makedirs
import numbers

FILE_EXT_CROPS = '.jpg'


class ImageLoader():
    def __init__(self, fullsize_img_dir, img_size, margin, border_mode='black'):
        assert border_mode in ['black', 'edge', 'mirror']
        self.fullsize_img_dir = fullsize_img_dir
        self.border_mode = border_mode
        if isinstance(img_size, numbers.Number):
            img_size = (img_size, img_size)
        self.input_size = img_size
        assert len(img_size) == 2
        self.size = img_size[0] + margin
        if isinstance(self.size, numbers.Number):
            self.size = (self.size, self.size)

    def load_image(self, filename):
        """ Load original image from dataset """
        img_path = os.path.join(self.fullsize_img_dir, filename)
        try:
            img = io.imread(img_path)
        except:
            raise IOError("\tError: Could not load image {}".format(img_path))
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape[2] == 4:
            print(filename, "converting RGBA to RGB...")
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        assert img.shape[2] == 3, "{}, invalid format: {}".format(img_path, img.shape)
        return img


class CachedCropLoader(ImageLoader):
    def __init__(self, fullsize_img_dir, cropped_img_root, crop_type,
                 use_cache=True, median_blur_crop=False, **kwargs):
        self.cropped_img_root = cropped_img_root
        self.median_blur_crop = median_blur_crop
        self.crop_type = crop_type
        self.use_cache = use_cache
        super().__init__(fullsize_img_dir, **kwargs)

    def _cache_filepath(self, filename, id, aligned):
        imgsize_dirname = str(int(self.input_size[0]))
        crop_dir = self.crop_type
        if not aligned:
            crop_dir += '_noalign'
        filename_noext = os.path.splitext(filename)[0]
        # adding an id is necessary in case there is more than one crop in an image
        if id is not None:
            filename_noext += '.{:07d}'.format(id)
        return os.path.join(self.cropped_img_root, imgsize_dirname, crop_dir, filename_noext + FILE_EXT_CROPS)

    def _load_cached_image(self, filename, id=None, aligned=False):
        is_cached = False
        if self.crop_type=='fullsize':
            img = self.load_image(filename)
        else:
            cache_filepath = self._cache_filepath(filename, id, aligned)
            if self.use_cache and os.path.isfile(cache_filepath):
                try:
                    img = io.imread(cache_filepath)
                except:
                    print("\tError: Could load not cropped image {}!".format(cache_filepath))
                    print("\tDeleting file and loading fullsize image.")
                    os.remove(cache_filepath)
                    img = self.load_image(filename)
                is_cached = True
            else:
                img = self.load_image(filename)

        assert isinstance(img, np.ndarray)
        return img, is_cached

    def load_crop(self, filename, bb=None, landmarks=None, id=None, aligned=False, mode='bounding_box'):
        assert mode in ['bounding_box', 'landmarks']
        assert mode != 'landmarks' or landmarks is not None
        assert mode == 'landmarks' or not aligned

        img, is_cached_crop = self._load_cached_image(filename, id, aligned)

        if self.crop_type == 'fullsize':
            return img

        if mode == 'bounding_box':
            self._cropper = cropping.FaceCrop(img, output_size=self.size, bbox=bb,
                                              img_already_cropped=is_cached_crop)
        else:
            self._cropper = cropping.FaceCrop(img, output_size=self.size, landmarks=landmarks,
                                              align_face_orientation=aligned,
                                              img_already_cropped=is_cached_crop)

        try:
            crop = self._cropper.apply_to_image(border_mode=self.border_mode)
            if self.use_cache and not is_cached_crop:
                cache_filepath = self._cache_filepath(filename, id, aligned)
                makedirs(cache_filepath)
                io.imsave(cache_filepath, crop)
        except cv2.error:
            print('Could not crop image {}.'.format(filename))
            crop = img  # fallback to fullsize image

        if self.median_blur_crop:
            crop = cv2.medianBlur(crop, ksize=3)
        return np.clip(crop, a_min=0, a_max=255)
