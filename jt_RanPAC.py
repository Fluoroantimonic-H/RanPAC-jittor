from jt_toolkit import *
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
import jittor.nn as nn
import copy
from jt_inc_net import SimpleVitNet
import math

import matplotlib.pyplot as plt

num_workers = 8


class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self.class_increments = []
        self._network = None

        self._device = args["device"][0] if isinstance(args["device"], list) else args["device"]
        self._multiple_gpus = args["device"]

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        acc_total, grouped = self._evaluate(y_pred, y_true)
        return acc_total, grouped, y_pred[:, 0], y_true

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []

        for _, (inputs, targets) in enumerate(loader):
            inputs = jt.array(inputs.numpy())
            targets = jt.array(targets.numpy())

            with jt.no_grad():  # 替换 torch.no_grad()
                outputs = self._network(inputs)["logits"]

            # Jittor 的 topk 用法
            predicts = jt.topk(outputs, k=1, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.numpy())  # 直接 .numpy() 无需 .cpu()
            y_true.append(targets.numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)

    def _evaluate(self, y_pred, y_true):
        acc_total, grouped = accuracy(y_pred.T[0], y_true, self._known_classes, self.class_increments)
        return acc_total, grouped

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0

        for i, (inputs, targets) in enumerate(loader):
            inputs = jt.array(inputs.numpy())
            targets = jt.array(targets.numpy())

            with jt.no_grad():
                outputs = model(inputs)["logits"]

            # Jittor 的 argmax 操作
            predicts = jt.argmax(outputs, dim=1)[0]  # Jittor 返回 (indices, values)
            # correct += (predicts == targets[:, 0]).sum().numpy()  # 同train_acc
            correct += (predicts == targets).sum().numpy()  # 同train_acc
            total += len(targets)

        return np.around(correct * 100 / total, decimals=2)

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if args["model_name"] != 'ncm':
            if args["model_name"] == 'adapter' and '_adapter' not in args["convnet_type"]:
                raise NotImplementedError('Adapter requires Adapter backbone')
            if args["model_name"] == 'ssf' and '_ssf' not in args["convnet_type"]:
                raise NotImplementedError('SSF requires SSF backbone')
            if args["model_name"] == 'vpt' and '_vpt' not in args["convnet_type"]:
                raise NotImplementedError('VPT requires VPT backbone')

            if 'resnet' in args['convnet_type']:
                raise NotImplementedError('No jittor implementation of ResNetCosineIncrementalNet')
            else:
                self._network = SimpleVitNet(args, True)
                self._batch_size = args["batch_size"]

            self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
            self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        else:
            self._network = SimpleVitNet(args, True)
            self._batch_size = args["batch_size"]
        self.args = args

        self.W_rand = jt.randn(768, self.args['M'])  # for test

        self.Q = None
        self.G = None

    def after_task(self):
        self._known_classes = self._classes_seen_so_far

    def replace_fc(self, trainloader):
        self._network = self._network.eval()

        if self.args['use_RP']:
            self._network.fc.use_RP = True
            if self.args['M'] > 0:
                self._network.fc.W_rand = self.W_rand
            else:
                self._network.fc.W_rand = None

        Features_f = []
        label_list = []
        with jt.no_grad():
            for i, (data, label) in enumerate(trainloader):

                data = jt.array(data.numpy())
                label = jt.array(label.numpy())

                embedding = self._network.convnet(data)
                Features_f.append(embedding.numpy())
                label_list.append(label)

        Features_f = jt.concat(Features_f, dim=0)
        label_list = jt.concat(label_list, dim=0)

        # one-hot编码
        Y = target2onehot(label_list, self.total_classnum)

        if self.args['use_RP']:
            if self.args['M'] > 0:
                Features_h = nn.relu(Features_f @ self._network.fc.W_rand)
            else:
                Features_h = Features_f

            self.Q = self.Q + Features_h.transpose() @ Y
            self.G = self.G + Features_h.transpose() @ Features_h

            ridge = self.optimise_ridge_parameter(Features_h, Y)

            Wo = np.linalg.solve(self.G + ridge * jt.init.eye(self.G.shape[0]), self.Q).transpose()
            Wo = jt.array(Wo)

            self._network.fc.weight.assign(Wo[0:self._network.fc.weight.shape[0], :])
        else:
            # 原型计算（DIL/CIL模式）
            for class_index in np.unique(self.train_dataset.labels):
                data_index = (label_list == class_index).nonzero()[:, 0]  # Jittor的nonzero返回格式不同
                if self.is_dil:
                    class_prototype = Features_f[data_index].sum(0)
                    self._network.fc.weight.data[class_index] += class_prototype  # DIL模式更新所有权重
                else:
                    class_prototype = Features_f[data_index].mean(0)
                    self._network.fc.weight.data[class_index] = class_prototype  # CIL模式仅更新新类

    def optimise_ridge_parameter(self, Features, Y):
        ridges = 10.0 ** np.arange(-8, 9)
        num_val_samples = int(Features.shape[0] * 0.8)
        losses = []

        Q_val = Features[0:num_val_samples, :].transpose() @ Y[0:num_val_samples, :]
        G_val = Features[0:num_val_samples, :].transpose() @ Features[0:num_val_samples, :]

        for ridge in ridges:
            # Wo = jt.linalg.solve(G_val + ridge * jt.init.eye(G_val.shape[0]), Q_val).transpose()
            Wo = np.linalg.solve(G_val + ridge * jt.init.eye(G_val.shape[0]), Q_val).transpose()
            Wo = jt.array(Wo)

            Y_train_pred = Features[num_val_samples:, :] @ Wo.transpose()
            # losses.append(nn.mse_loss(Y_train_pred, Y[num_val_samples:, :]).numpy())  # 转换为numpy标量
            losses.append(nn.mse_loss(Y_train_pred, Y[num_val_samples:, :]))

        ridge = ridges[np.argmin(np.array(losses))]
        # ridge = ridges[jt.argmin(jt.array(losses))[0]]
        logging.info("Optimal lambda: " + str(ridge))
        return ridge

    def incremental_train(self, data_manager):
        self.total_classnum = data_manager.get_total_classnum()
        self._cur_task += 1
        self._classes_seen_so_far = self._known_classes + data_manager.get_task_size(self._cur_task)

        if self.args['use_RP']:
            self._network.fc = None

        self._network.update_fc(self._classes_seen_so_far)

        if not self.is_dil:
            logging.info("Starting CIL Task {}".format(self._cur_task + 1))
        logging.info("Learning on classes {}-{}".format(self._known_classes, self._classes_seen_so_far - 1))

        self.class_increments.append([self._known_classes, self._classes_seen_so_far - 1])

        self.train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._classes_seen_so_far),
            source="train", mode="train"
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        train_dataset_for_CPs = data_manager.get_dataset(
            np.arange(self._known_classes, self._classes_seen_so_far),
            source="train", mode="test"
        )
        self.train_loader_for_CPs = DataLoader(
            train_dataset_for_CPs,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        self.train_loader_for_CPs = self.train_loader  # for test

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._classes_seen_so_far),
            source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader, self.train_loader_for_CPs)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def freeze_backbone(self, is_first_session=False):
        # 检查是否是ViT结构
        is_vit = 'vit' in self.args['convnet_type']

        if isinstance(self._network.convnet, nn.Module):
            for name, param in self._network.convnet.named_parameters():
                if is_first_session:
                    if is_vit:
                        if "head." not in name and "ssf_scale" not in name and "ssf_shift_" not in name:
                            param.requires_grad = False
                    else:
                        if "ssf_scale" not in name and "ssf_shift_" not in name:
                            param.requires_grad = False
                else:
                    param.requires_grad = False

    def show_num_params(self, verbose=False):
        # 总参数量统计
        total_params = sum(p.numel() for p in self._network.parameters())
        logging.info(f'{total_params:,} total parameters.')

        # 可训练参数量统计
        total_trainable_params = sum(
            p.numel() for p in self._network.parameters()
            if p.requires_grad
        )
        for name, p in self._network.named_parameters():
            # print(name + ": " + str(p.requires_grad))
            if p.requires_grad:
                print(name)
        logging.info(f'{total_trainable_params:,} training parameters.')

        if total_params != total_trainable_params and verbose:
            for name, param in self._network.named_parameters():
                if param.requires_grad:
                    print(f"{name}: {param.numel():,}")

    def _train(self, train_loader, test_loader, train_loader_for_CPs):
        if self._cur_task == 0 and self.args["model_name"] in ['ncm', 'joint_linear']:
            self.freeze_backbone()

        if self.args["model_name"] in ['joint_linear', 'joint_full']:
            # 联合训练分支（使用SGD优化所有任务）
            if self.args["model_name"] == 'joint_linear':
                assert self.args['body_lr'] == 0.0

            self.show_num_params()

            # Jittor 优化器配置
            optimizer = jt.optim.SGD([
                {'params': self._network.convnet.parameters()},
                {'params': self._network.fc.parameters(), 'lr': self.args['head_lr']}
            ], momentum=0.9, lr=self.args['body_lr'], weight_decay=self.weight_decay)

            # Jittor 学习率调度器
            scheduler = jt.optim.MultiStepLR(optimizer, milestones=[100000])

            logging.info("Starting joint training on all data using " + self.args["model_name"] + " method")
            self._init_train(train_loader, test_loader, optimizer, scheduler)
            self.show_num_params()
        else:
            # CP更新或PETL方法训练分支
            if self._cur_task == 0 and not self.dil_init:# and False:
                if 'ssf' in self.args['convnet_type']:
                    self.freeze_backbone(is_first_session=True)

                if self.args["model_name"] != 'ncm':
                    # PETL方法训练
                    self.show_num_params()
                    optimizer = jt.optim.SGD(
                        self._network.parameters(),
                        momentum=0.9,
                        lr=self.args['body_lr'] * 100,
                        weight_decay=self.weight_decay
                    )
 
                    scheduler = jt.optim.LambdaLR(
                        optimizer,
                        lambda step: self.min_lr + 0.5 * (self.args['body_lr'] - self.min_lr) * (
                                    1 + math.cos(math.pi * step / self.args['tuned_epoch']))
                    )
                    # scheduler = None

                    logging.info("Starting PETL training on first task using " + self.args["model_name"] + " method")
                    self._init_train(train_loader, test_loader, optimizer, scheduler)
                    self.freeze_backbone()

                if self.args['use_RP'] and not self.dil_init:
                    self.setup_RP()

            if self.is_dil and not self.dil_init:
                self.dil_init = True
                self._network.fc.weight.data.fill_(0.0)

            self.replace_fc(train_loader_for_CPs)
            self.show_num_params()

    def setup_RP(self):
        """初始化随机投影矩阵（Jittor版本）"""
        self.initiated_G = False
        self._network.fc.use_RP = True

        if self.args['M'] > 0:
            M = self.args['M']
            
            self._network.fc.weight = jt.random((self._network.fc.out_features, M), 'float32')
            self._network.fc.reset_parameters()

            self._network.fc.W_rand = jt.randn(self._network.fc.in_features, M)
            self.W_rand = copy.deepcopy(self._network.fc.W_rand)  

        else:
            M = self._network.fc.in_features

        # 初始化统计矩阵（Jittor的zeros自动处理设备）
        self.Q = jt.zeros((M, self.total_classnum))
        self.G = jt.zeros((M, M))

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))

        train_loss_list = []

        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs = jt.array(inputs.numpy())
                targets = jt.array(targets.numpy())

                logits = self._network(inputs)["logits"]

                loss = nn.cross_entropy_loss(logits, targets)

                # 优化步骤
                optimizer.step(loss)

                losses += float(loss.item())
                preds, _ = jt.argmax(logits, dim=1)  # Jittor的argmax返回(indices, values)

                correct += (preds == targets).sum().numpy()  # 避免correct计算出错
                total += len(targets)

            # 更新学习率
            scheduler.step()

            avg_loss = losses / len(train_loader)
            train_loss_list.append(avg_loss)

            # 计算准确率
            train_acc = np.around(correct * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)

            # 进度条显示
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {}, Test_accy {}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                str(train_acc),
                # test_acc,
                str(test_acc),      # python版本太低需要用str
            )
            prog_bar.set_description(info)

        logging.info(info)

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, marker='o', color='b', label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig("loss_curve.png")