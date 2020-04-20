from datasets.ds_config import DATASETS
from landmarks.train_aae_landmarks import AAELandmarkTraining
from aae_training import AAETraining

TRAIN = 'train'

class FabrecEval(AAELandmarkTraining):
    def __init__(self, datasets, args, session_name, **kwargs):
        args.reset = False  # just to make sure we don't reset the discriminator by accident
        ds = datasets[VAL]
        self.num_landmarks = ds.NUM_LANDMARKS
        self.all_landmarks = ds.ALL_LANDMARKS
        self.landmarks_no_outline = ds.LANDMARKS_NO_OUTLINE
        self.landmarks_only_outline = ds.LANDMARKS_ONLY_OUTLINE
        AAETraining.__init__(self,datasets, args, session_name, **kwargs)

def run(args):

    if args.seed is not None:
        from utils.common import init_random
        init_random(args.seed)
    # log.info(json.dumps(vars(args), indent=4))

    datasets = {}
    dsname = args.dataset_val[0]
    datasets[VAL] = DATASETS[dsname](train=False,
                                     test_split=args.test_split,
                                     max_samples=args.val_count,
                                     start=args.st,
                                     use_cache=args.use_cache,
                                     align_face_orientation=args.align,
                                     crop_source=args.crop_source,
                                     return_landmark_heatmaps=True,
                                     landmark_sigma=args.sigma,
                                     image_size=args.input_size)
    print(datasets[VAL])


    fntr = FabrecEval(datasets, args, args.sessionname, workers=args.workers, wait=args.wait)

    import torch
    torch.backends.cudnn.benchmark = True
    fntr.eval_epoch()

if __name__ == '__main__':
    # Disable traceback on Ctrl+c
    import sys
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    import configargparse
    import numpy as np
    np.set_printoptions(linewidth=np.inf)

    import aae_training
    from datasets import wflw
    from constants import VAL
    from utils import log
    from landmarks import lmconfig

    parser = configargparse.ArgParser()
    aae_training.add_arguments(parser)

    # parser.add_argument('-c', '--conf', required=False, is_config_file=True, help='config file path')

    # Dataset
    parser.add_argument('--dataset', default=['300w'], type=str, choices=DATASETS, nargs='+',
                        help='dataset for training and testing')
    parser.add_argument('--test-split', default='full', type=str, help='test set split for 300W/AFLW/WFLW',
                        choices=['challenging', 'common', '300w', 'full', 'frontal']+wflw.SUBSETS)

    # Landmarks
    parser.add_argument('--sigma', default=7, type=float, help='size of landmarks in heatmap')
    parser.add_argument('--ocular-norm', default=lmconfig.LANDMARK_OCULAR_NORM, type=str,
                        help='how to normalize landmark errors', choices=['pupil', 'outer', 'none'])

    args = parser.parse_args()

    args.dataset_train = args.dataset
    args.dataset_val = args.dataset

        # log.info('Switching to evaluation mode...')
    args.eval = True
    args.batchsize_eval = 10
    args.wait = 0
    args.workers = 0
    args.print_freq_eval = 1
    args.epochs = 1

    if args.benchmark:
        log.info('Switching to benchmark mode...')
        args.batchsize_eval = 50
        args.wait = 10
        args.workers = 4
        args.print_freq_eval = 20
        args.epochs = 1
        args.val_count = None

    if args.sessionname is None:
        if args.resume:
            import os
            modelname = os.path.split(args.resume)[0]
            args.sessionname = modelname
        else:
            args.sessionname = 'debug'
    # print(args)
    # exit()
    run(args)
