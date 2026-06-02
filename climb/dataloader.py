import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from datasets.market1501 import Market1501
from datasets.msmt17 import MSMT17
from datasets.ilidsvid import iLIDSVID
from datasets.video_loader import VideoDataset, VideoTransform
from .preprocessing import RandomErasing
from .dataset import ImageDataset, IterLoader
from .sampler import RandomIdentitySampler, RandomMultipleGallerySampler

FACTORY = {
    'market1501': Market1501,
    'msmt17': MSMT17,
    'ilidsvid': iLIDSVID,
}


def train_collate_fn(batch):
    imgs, pids, camids, viewids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids


def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths


def make_CLIMB_dataloader(cfg, all_iters=False):
    """
    PCL dataloader. It returns 3 dataloaders: training loader, cluster loader and validation loader.
    """

    dataset_name = cfg.DATASETS.NAMES
    split_id = getattr(cfg.DATASETS, 'SPLIT', 0)
    dataset = FACTORY[dataset_name](root=cfg.DATASETS.ROOT_DIR, split_id=split_id)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    # Check if video dataset
    is_video = dataset_name in ['ilidsvid', 'mars', 'lsvid']

    if is_video:
        seq_len = getattr(cfg.INPUT, 'SEQ_LEN', 8)

        # train transforms with temporal consistency (shared random params across frames)
        video_train_transform = VideoTransform(
            size_train=cfg.INPUT.SIZE_TRAIN,
            prob=cfg.INPUT.PROB,
            padding=cfg.INPUT.PADDING,
            pixel_mean=cfg.INPUT.PIXEL_MEAN,
            pixel_std=cfg.INPUT.PIXEL_STD,
            re_prob=cfg.INPUT.RE_PROB,
        )

        # val/test transforms
        val_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST, interpolation=3),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ])

        # train loader with rrs_train sampling
        train_set = VideoDataset(dataset.train, seq_len=seq_len, sample='rrs_train', video_transform=video_train_transform)
        sampler = RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=train_collate_fn,
        )
        train_loader = IterLoader(train_loader, cfg.SOLVER.ITERS if not all_iters else None)

        # val loader with dense sampling for evaluation
        val_set = VideoDataset(dataset.query + dataset.gallery, seq_len=seq_len, sample='dense', transform=val_transforms)
        val_loader = DataLoader(
            val_set,
            batch_size=1,  # dense sampling requires batch_size=1
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=val_collate_fn,
        )
        num_queries = len(dataset.query)

        # cluster loader for memory bank (use rrs_test for consistent feature extraction)
        cluster_set = VideoDataset(dataset.train, seq_len=seq_len, sample='rrs_test', transform=val_transforms)
        cluster_loader = DataLoader(
            cluster_set,
            batch_size=cfg.TEST.IMS_PER_BATCH,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=train_collate_fn,
        )

    else:
        # Image dataset (original logic)
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
        train_set = ImageDataset(dataset.train, train_transforms)
        sampler = RandomMultipleGallerySampler(dataset.train, cfg.DATALOADER.NUM_INSTANCE)
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=sampler,
            num_workers=num_workers,
        )
        train_loader = IterLoader(train_loader, cfg.SOLVER.ITERS if not all_iters else None)

        val_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST, interpolation=3),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ])
        val_set = ImageDataset(dataset.query+dataset.gallery, val_transforms, return_path=True)
        num_queries = len(dataset.query)
        eval_batch = cfg.TEST.IMS_PER_BATCH
        val_loader = DataLoader(
            val_set, batch_size=eval_batch, shuffle=False, num_workers=num_workers
        )

        cluster_set = ImageDataset(dataset.train, val_transforms)
        cluster_loader = DataLoader(
            cluster_set, batch_size=eval_batch, shuffle=False, num_workers=num_workers
        )

    return train_loader, val_loader, cluster_loader, num_queries, num_classes, cam_num, view_num
