import jittor as jt
import numpy as np


def count_parameters(model, trainable=False):
    """统计模型参数量（与PyTorch兼容）
    Args:
        model: 待统计的模型
        trainable: 是否只统计可训练参数
    """
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def val2numpy(x):
    """Jittor张量转NumPy数组（自动处理设备）"""
    return x.numpy()  # Jittor张量无论CPU/GPU都直接用.numpy()


def target2onehot(targets, n_classes):
    """标签转one-hot编码
    Args:
        targets: 形状为[N]的整数标签张量
        n_classes: 总类别数
    """
    # Jittor的scatter_用法与PyTorch相同
    onehot = jt.zeros((targets.shape[0], n_classes), dtype=jt.float32)
    # onehot.scatter_(1, targets.unsqueeze(-1).long().view(-1, 1), 1.0)
    src = jt.ones((targets.shape[0], n_classes), dtype=jt.float32)
    onehot.scatter_(1, targets.unsqueeze(-1).long().view(-1, 1), src)
    # jt.misc.scatter(onehot, 1, targets.long().view(-1, 1), src)
    return onehot


def accuracy(y_pred, y_true, nb_old, class_increments):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    acc_total = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # Grouped accuracy
    for classes in class_increments:
        idxes = np.where(
            np.logical_and(y_true >= classes[0], y_true <= classes[1])
        )[0]
        label = "{}-{}".format(
            str(classes[0]).rjust(2, "0"), str(classes[1]).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    return acc_total,all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)
