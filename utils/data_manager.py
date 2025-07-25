import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR224, iImageNetR, iImageNetA, CUB, omnibenchmark, vtab, cars, core50, cddb, domainnet

class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment,use_input_norm=False):
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed,use_input_norm)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(self, indices, source, mode, appendent=None, ret_data=False):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed,use_input_norm):
        idata = _get_idata(dataset_name,use_input_norm)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        if 'domainnet' not in dataset_name :
            # Order
            order = [i for i in range(len(np.unique(self._train_targets)))]
            if shuffle:
                np.random.seed(seed)
                order = np.random.permutation(len(order)).tolist()
            else:
                order = idata.class_order
            self._class_order = order
            logging.info("Class Order: ["+",".join([str(x) for x in self._class_order])+"]")

            # Map indices
            self._train_targets = _map_new_class_index(
                self._train_targets, self._class_order
            )
            self._test_targets = _map_new_class_index(self._test_targets, self._class_order)
        else:
            np.random.seed(seed)
            self._class_order =np.arange(345).tolist()
            logging.info("Class Order: ["+",".join([str(x) for x in self._class_order])+"]")

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        # return idx, image, label
        return image, label
        # return np.array(image).astype(np.float32), np.array(label).astype(np.int32)


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name,use_input_norm):
    name = dataset_name.lower()
    if name== "cifar224":
        return iCIFAR224(use_input_norm)
    elif name== "imagenetr":
        return iImageNetR(use_input_norm)
    elif name=="imageneta":
        return iImageNetA(use_input_norm)
    elif name=="cub":
        return CUB(use_input_norm)
    elif name=="omnibenchmark":
        return omnibenchmark(use_input_norm)
    elif name=="vtab":
        return vtab(use_input_norm)
    elif name=="cars":
        return cars(use_input_norm)
    elif "core50" in name:
        logging.info('Starting next DIL task: '+name)
        return core50(name[7::],use_input_norm)
    elif "cddb" in name:
        logging.info('Starting next DIL task: '+name)
        return cddb(name[5::],use_input_norm)
    elif "domainnet" in name:
        logging.info('Starting next DIL task: '+name)
        return domainnet(name[10::],use_input_norm)

    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
