import os
# import socket

# COGSYS_SERVER_NAME = 'cogsys-dualGPU'
# LAPTOP_NAME = 'browatbn-HP-ENVY'
# LAB_PC_NAME = 'browatbn-HP-Z620'

# PROJECT_DIR = '/home/browatbn/dev/csl/faces'

# DATA_DIR = '/media/browatbn/073dbe00-d671-49dd-9ebc-c794352523ba/dev/data'
# DATASET_ROOT = '/media/browatbn/073dbe00-d671-49dd-9ebc-c794352523ba/datasets'
# DATASET_ROOT_LOCAL = '/home/browatbn/dev/datasets'
#
# if socket.gethostname() == LAPTOP_NAME:
#     DATA_DIR = os.path.join('/media/browatbn/Samsung_T5/dev', 'data')
# elif socket.gethostname() == COGSYS_SERVER_NAME:
#     DATA_DIR = os.path.join('/home/browatbn/dev', 'data')
# elif not socket.gethostname() == LAB_PC_NAME:
#     DATA_DIR = '/opt/notebooks/dev/data'
#     PROJECT_DIR = '/opt/notebooks/dev/expressions'
#     DATASET_ROOT = '/opt/notebooks/dev/datasets'
#     DATASET_ROOT_LOCAL = '/opt/notebooks/dev/datasets'
#
#
# MODEL_DIR = os.path.join(DATA_DIR, 'models')
# SNAPSHOT_DIR = os.path.join(MODEL_DIR, 'snapshots')
#
# OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs')
# RESULT_DIR = os.path.join(OUTPUT_DIR, 'results')
# REPORT_DIR = os.path.join(OUTPUT_DIR, 'reports')

# EMOTIW2018_ROOT_LOCAL = os.path.join(DATASET_ROOT_LOCAL, 'EmotiW2018')
# EMOTIW2018_ROOT = os.path.join(DATASET_ROOT_LOCAL, 'EmotiW2018')
#
# FER2013_ROOT = os.path.join(DATASET_ROOT_LOCAL, 'fer2013')
#
# UMDFACES_ROOT = os.path.join(DATASET_ROOT, 'UMDFaces')
# UMDFACES_ROOT_LOCAL = os.path.join(DATASET_ROOT_LOCAL, 'UMDFaces')
#
# OULU_ROOT = os.path.join(DATASET_ROOT, 'Oulu')
# OULU_ROOT_LOCAL = os.path.join(DATASET_ROOT_LOCAL, 'Oulu')
#
# VOXCELEB_ROOT = os.path.join(DATASET_ROOT, 'VoxCeleb1')
# VOXCELEB_ROOT_LOCAL = os.path.join(DATASET_ROOT_LOCAL, 'VoxCeleb1')
#
# LFW_ROOT = os.path.join(DATASET_ROOT, 'LFW')
#
# CELEBA_ROOT = os.path.join(DATASET_ROOT, 'CelebA')
# CELEBA_ROOT_LOCAL = os.path.join(DATASET_ROOT_LOCAL, 'CelebA')
#
# RAF_ROOT = os.path.join(DATASET_ROOT, 'RAF/basic')
# RAF_ROOT_LOCAL = os.path.join(DATASET_ROOT_LOCAL, 'RAF/basic')
#
#
# WIDER_ROOT = os.path.join(DATASET_ROOT, 'WIDER')
# WIDER_ROOT_LOCAL = os.path.join(DATASET_ROOT_LOCAL, 'WIDER')
#
# MULTIPIE_ROOT = os.path.join(DATASET_ROOT, 'Multi-Pie')
# MULTIPIE_ROOT_LOCAL = os.path.join(DATASET_ROOT_LOCAL, 'Multi-Pie')
#
# CFPW_ROOT = os.path.join(DATASET_ROOT, 'CFPW')
# CFPW_ROOT_LOCAL = os.path.join(DATASET_ROOT, 'CFPW')
#
# DEEPFASHION_ROOT = os.path.join(DATASET_ROOT, 'DeepFashion')
# DEEPFASHION_ROOT_LOCAL = os.path.join(DATASET_ROOT_LOCAL, 'DeepFashion')
#
# MTFL_ROOT = os.path.join(DATASET_ROOT_LOCAL, 'MTFL')
# MTFL_ROOT_LOCAL = os.path.join(DATASET_ROOT_LOCAL, 'MTFL')
#
# COMPCARS_ROOT = os.path.join(DATASET_ROOT, 'CompCars/data')
# COMPCARS_ROOT_LOCAL = os.path.join(DATASET_ROOT_LOCAL, 'CompCars')



ENCODER_LAYER_NORMALIZATION = 'batch'
DECODER_LAYER_NORMALIZATION = 'batch'
DECODER_SPECTRAL_NORMALIZATION = False
DECODER_PLANES_PER_BLOCK = 1

# from datasets import affectnet, vggface2, multi, w300, wflw, aflw
# from datasets import deepfashion, celeba, mtfl, compcars, ffhq
#
# AFFECTNET_RAW_DIR = os.path.join(DATASET_ROOT_LOCAL, 'AffectNet')
# AFFECTNET_ROOT = os.path.join(DATASET_ROOT_LOCAL, 'AffectNet')
# AFFECTNET_ROOT_LOCAL = os.path.join(DATASET_ROOT_LOCAL, 'AffectNet')
#
# VGGFACE2_ROOT = os.path.join(DATASET_ROOT, 'VGGFace2')
# VGGFACE2_ROOT_LOCAL = os.path.join(DATASET_ROOT_LOCAL, 'VGGFace2')
#
# W300_ROOT = os.path.join(DATASET_ROOT, '300W')
# W300_ROOT_LOCAL = os.path.join(DATASET_ROOT_LOCAL, '300W')
#
# WFLW_ROOT = os.path.join(DATASET_ROOT, 'WFLW')
# WFLW_ROOT_LOCAL = os.path.join(DATASET_ROOT_LOCAL, 'WFLW')
#
# AFLW_ROOT = os.path.join(DATASET_ROOT, 'AFLW/aflw')
# AFLW_ROOT_LOCAL = os.path.join(DATASET_ROOT_LOCAL, 'AFLW')



# DATASETS = {
#     'multi': multi.MultiFaceDataset,
#     'affectnet': affectnet.AffectNet,
#     'vggface2': vggface2.VggFace2,
#     '300w': w300.W300,
#     'aflw': aflw.AFLW,
#     'wflw': wflw.WFLW,
    # 'deepfashion': deepfashion.DeepFashion,
    # 'mafl': celeba.MAFL,
    # 'celeba': celeba.CelebA,
    # 'mtfl': mtfl.MTFL,
    # 'compcars': compcars.CompCars,
    # 'ffhq': ffhq.FFHQ,
# }


# DATASET_ROOT = {
#     'affectnet': AFFECTNET_ROOT,
#     'vggface2': VGGFACE2_ROOT
# }
#
# DATASET_ROOT_LOCAL = {
#     'affectnet': AFFECTNET_ROOT_LOCAL,
#     'vggface2': VGGFACE2_ROOT_LOCAL
# }
#
# def get_dataset_root(ds_str, local=False):
#     if local:
#         return DATASET_ROOT_LOCAL[ds_str]
#     else:
#         return DATASET_ROOT[ds_str]

datasets = {}
def register_dataset(cls):
    datasets[cls.__name__.lower()] = cls


# def _load_paths_from_config(parser):
#     def add_dataset(cls):
#         dsname = cls.__name__.lower()
#         parser.add_argument(f'--{dsname}', default='./', type=str, metavar='PATH',
#                             help=f"root directory of dataset {dsname}")
#         parser.add_argument(f'--{dsname}_local', default='./', type=str, metavar='PATH',
#                             help='path to directory that will be used to store cached data (e.g. crops)')
#
#     parser.add_argument('-c',  default='local_config.ini', required=False, is_config_file=True, help='config file path')
#
#     # datasets for pre-training
#     from datasets import affectnet, vggface2
#     add_dataset(affectnet.AffectNet)
#     add_dataset(vggface2.VggFace2)
#
#     # datasets for landmark detection
#     from datasets import w300, wflw, aflw
#     add_dataset(w300.W300)
#     add_dataset(aflw.AFLW)
#     add_dataset(wflw.WFLW)


_default_config_files = [
    './local_config.ini',
    '../local_config.ini'
]

def get_dataset_paths(dsname):
    import configargparse
    _parser = configargparse.ArgParser(default_config_files=_default_config_files)
    # _parser.add_argument('-c',  default='local_config.ini', required=False, is_config_file=True, help='config file path')
    def add_dataset(dsname):
        _parser.add_argument(f'--{dsname}', default='./', type=str, metavar='PATH',
                            help=f"root directory of dataset {dsname}")
        _parser.add_argument(f'--{dsname}_local', default='', type=str, metavar='PATH',
                            help='path to directory that will be used to store cached data (e.g. crops)')
    add_dataset(dsname)
    _paths = _parser.parse_known_args()[0]
    assert hasattr(_paths, dsname)
    assert hasattr(_paths, dsname+'_local')

    def check_paths(path, path_local):
        """return 'path' if 'path_local' is not defined"""
        if not os.path.exists(path):
            raise IOError(f"Could not find datset {dsname}. Invalid path '{path}'.")
        if not path_local:
            path_local = path
        if not os.path.exists(path_local):
            raise IOError(f"Could not set up dataset {dsname}. Invalid path '{path_local}'.")
        return path, path_local

    return check_paths(_paths.__getattribute__(dsname), _paths.__getattribute__(dsname+'_local'))


def read_local_config():
    import configargparse
    _parser = configargparse.ArgParser(default_config_files=_default_config_files)
    # _parser.add_argument('-c',  default='local_config.ini', required=False, is_config_file=True, help='config file path')
    _parser.add_argument(f'--data', default='./data', type=str, metavar='PATH')
    _parser.add_argument(f'--outputs', default='./outputs', type=str, metavar='PATH')

    _paths = _parser.parse_known_args()[0]
    return _paths

_paths = read_local_config()
DATA_DIR = _paths.data
OUTPUT_DIR = _paths.outputs

MODEL_DIR = os.path.join(DATA_DIR, 'models')
SNAPSHOT_DIR = os.path.join(MODEL_DIR, 'snapshots')
RESULT_DIR = os.path.join(OUTPUT_DIR, 'results')
REPORT_DIR = os.path.join(OUTPUT_DIR, 'reports')
