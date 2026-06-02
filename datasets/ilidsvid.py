from __future__ import absolute_import
import os
import os.path as osp
import glob
import scipy.io as sio
from datasets.bases import BaseDataset


class iLIDSVID(BaseDataset):
    """
    iLIDS-VID Dataset for Video Person Re-ID.

    Dataset statistics:
    # identities: 300
    # cameras: 2
    """

    def __init__(self, root='/home/data/users/cfdeng/tdh/datasets/ilids-vid', split_id=0, min_seq_len=0):
        self.root = root
        self.split_id = split_id
        self.min_seq_len = min_seq_len
        self._check_before_run()

        # Load person identities
        identities = self._load_identities()

        # Load splits
        split_mat = sio.loadmat(osp.join(root, 'train-test people splits', 'train_test_splits_ilidsvid.mat'))
        person_list = split_mat['ls_set']

        pids = (person_list[split_id] - 1).squeeze().tolist()
        num = len(identities)
        trainval_pids = sorted(pids[:num // 2])
        test_pids = sorted(pids[num // 2:])

        # Build tracklets
        self.train, self.num_train_tracklets, self.num_train_pids, self.num_train_imgs = \
            self._process_data(identities, trainval_pids, relabel=True)

        self.query, self.num_query_tracklets, self.num_query_pids, self.num_query_imgs = \
            self._process_data(identities, test_pids, relabel=False, cam_id=0)

        self.gallery, self.num_gallery_tracklets, self.num_gallery_pids, self.num_gallery_imgs = \
            self._process_data(identities, test_pids, relabel=False, cam_id=1)

        print("=> iLIDS-VID loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(self.num_train_pids, self.num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(self.num_query_pids, self.num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(self.num_gallery_pids, self.num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(
            self.num_train_pids + self.num_query_pids,
            self.num_train_tracklets + self.num_query_tracklets + self.num_gallery_tracklets))
        print("  ------------------------------")

        self.num_train_cams = 2
        self.num_train_vids = 1

    def _check_before_run(self):
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(osp.join(self.root, 'i-LIDS-VID')):
            raise RuntimeError("'{}' is not available".format(osp.join(self.root, 'i-LIDS-VID')))
        if not osp.exists(osp.join(self.root, 'train-test people splits', 'train_test_splits_ilidsvid.mat')):
            raise RuntimeError("Split file not found")

    def _load_identities(self):
        """Load all identities with their frame paths for each camera."""
        identities = []
        seq_dir = osp.join(self.root, 'i-LIDS-VID', 'sequences')

        for pid in range(1, 320):  # person001 to person319
            cam_frames = []
            valid = True
            for camid in range(1, 3):
                person_dir = osp.join(seq_dir, 'cam{}'.format(camid), 'person{:03d}'.format(pid))
                if osp.exists(person_dir):
                    frames = sorted(glob.glob(osp.join(person_dir, '*.png')))
                    if len(frames) > 0:
                        cam_frames.append(frames)
                    else:
                        valid = False
                        break
                else:
                    valid = False
                    break

            if valid and len(cam_frames) == 2:
                identities.append(cam_frames)

        return identities

    def _process_data(self, identities, indices, relabel=False, cam_id=None):
        """Process data into tracklets format."""
        tracklets = []
        num_imgs_per_tracklet = []

        pid2label = {pid: label for label, pid in enumerate(indices)} if relabel else {}

        for pid in indices:
            if pid >= len(identities):
                continue
            pid_images = identities[pid]

            if cam_id is not None:
                # For query/gallery: only one camera
                cam_images = pid_images[cam_id]
                if len(cam_images) >= self.min_seq_len:
                    tracklets.append((tuple(cam_images), pid2label.get(pid, pid), cam_id, 0))
                    num_imgs_per_tracklet.append(len(cam_images))
            else:
                # For training: both cameras
                for camid, cam_images in enumerate(pid_images):
                    if len(cam_images) >= self.min_seq_len:
                        tracklets.append((tuple(cam_images), pid2label.get(pid, pid), camid, 0))
                        num_imgs_per_tracklet.append(len(cam_images))

        num_tracklets = len(tracklets)
        num_pids = len(set([t[1] for t in tracklets]))
        num_imgs = sum(num_imgs_per_tracklet)

        return tracklets, num_tracklets, num_pids, num_imgs
