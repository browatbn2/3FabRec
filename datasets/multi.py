import numpy as np
import torch.utils.data as td
import sklearn.utils

class MultiFaceDataset(td.Dataset):
    def __init__(self, datasets, train=True, shuffle=True, max_samples=None):
        self.datasets = datasets
        self.idx = [0] * len(self.datasets)

        inds = []
        for id, ds in enumerate(self.datasets):
            inds.append(np.array([[id]*len(ds), range(len(ds))]))
            print(ds.__class__.__name__, len(ds))

        self.joint_idx = np.hstack(inds).transpose()
        if max_samples is not None:
            self.joint_idx = sklearn.utils.shuffle(self.joint_idx)
            self.joint_idx = self.joint_idx[:max_samples]
        # print(self.joint_idx.shape)

    def __len__(self):
        # return sum(map(len, self.datasets))
        return len(self.joint_idx)

    def __getitem__(self, idx):
        ds, sample_idx = self.joint_idx[idx]
        return self.datasets[ds][sample_idx]

    def __repr__(self):
        return '\n'.join([ds.__repr__() for ds in self.datasets])

    def print_stats(self):
        for ds in self.datasets:
            ds.print_stats()

    def get_class_sizes(self):
        sizes = []
        for ds in self.datasets:
            sizes.append(ds.get_class_sizes())
        return np.sum(sizes, axis=0)

    @property
    def heights(self):
        sizes = []
        for ds in self.datasets:
            sizes.append(ds.heights)
        return np.concatenate(sizes)

    @property
    def widths(self):
        sizes = []
        for ds in self.datasets:
            sizes.append(ds.widths)
        return np.concatenate(sizes)


if __name__ == '__main__':
    import torch
    from utils import vis
    from utils.nn import Batch
    from datasets import ds_utils
    from datasets import affectnet

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    train = True
    datasets = [
        affectnet.AffectNet(train=train, max_samples=1000),
        # vggface2.VggFace2(train=train, max_samples=1000),
    ]
    multi_ds = MultiFaceDataset(datasets, train=True, max_samples=5000)
    print(multi_ds)
    dl = td.DataLoader(multi_ds, batch_size=40, shuffle=False, num_workers=0)
    for data in dl:
        batch = Batch(data, gpu=False)
        inputs = batch.images.clone()
        ds_utils.denormalize(inputs)
        imgs = vis.add_landmarks_to_images(inputs.numpy(), batch.landmarks.numpy())
        # imgs = vis.add_pose_to_images(inputs.numpy(), batch.poses.numpy())
        # imgs = vis.add_emotion_to_images(imgs, batch.emotions.numpy())
        vis.vis_square(imgs, nCols=20, fx=0.6, fy=0.6, normalize=False)
