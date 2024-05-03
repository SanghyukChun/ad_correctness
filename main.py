"""
AutoDiff Correctness Checker
Copyright (c) 2024-present Sanghyuk Chun.
MIT license
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from scipy.optimize import linprog

import os
import argparse

import time
import datetime

import activations
from dataloader import get_cifar_loader, get_mnist_loader
from models import vgg, resnet, fnn

from torch_receptive_field import receptive_field, receptive_field_for_unit


def build_model(args):
    if 'vgg' in args.model:
        getter = getattr(vgg, args.model)
    elif 'resnet' in args.model:
        getter = getattr(resnet, args.model)
    elif args.model == 'fnn':
        pass
    else:
        raise ValueError(f'{args.model} is not a valid model name')

    relu_fn = nn.ReLU

    if 'vgg' in args.model:
        return getter(relu_fn=relu_fn,
                      maxpool_fn=activations.NewMaxPool2d,
                      is_imagenet=False)
    elif 'resnet' in args.model:
        # Note that ResNet implementation used in this code uses stride=1,
        # therefore it is okay with using `NewMaxPool2d`.
        # However, if you need to use a standard ResNet for ImageNet,
        # then you should use `NewMaxPool2d_with_stride` instead.
        return getter(relu_fn=relu_fn,
                      maxpool_fn=activations.NewMaxPool2d_with_stride,
                      is_imagenet=False)
    elif args.model == 'fnn':
        if args.fnnactivation not in ['ReLU6', 'Hardsigmoid', 'Hardtanh']:
            raise ValueError(args.fnnactivation)
        return fnn.FNN(activation=args.fnnactivation)


def check_fnn(S, idx, cur_batch_idx, ad_log_fname):
    if S[idx]:
        rank = torch.linalg.matrix_rank(torch.stack(S[idx]))
        with open(ad_log_fname, 'a') as fout:
            fout.write(f'{cur_batch_idx},{idx},{rank},{len(S[idx])},{rank == len(S[idx])}\n')


class TrainEngine():
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def build(self):
        print('==> Preparing data..')
        self.build_data_loaders()

        print('==> Building model..')
        self.build_model()
        self.build_optimizer()
        self.build_scheduler()

    def build_data_loaders(self):
        if self.args.dataset == 'cifar10':
            self.trainloader, self.testloader = get_cifar_loader(self.args)
        elif self.args.dataset == 'mnist':
            self.trainloader, self.testloader = get_mnist_loader(self.args)
        else:
            raise ValueError(self.args.dataset)

    def build_model(self):
        self.net = build_model(self.args)
        self.net = self.net.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        if 'vgg' in args.model:
            self.receptive_field_dict = receptive_field(self.net, (3, 32, 32))

    def build_optimizer(self):
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr,
                                   momentum=0.9, weight_decay=self.args.wd)

    def build_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.n_epochs)

    def train(self):
        args = self.args

        self.mp_log_fname = os.path.join(args.log_dir, 'mp_log.txt')
        self.ad_log_fname = os.path.join(args.log_dir, 'ad_log.txt')
        self.loss_log_fname = os.path.join(args.log_dir, 'loss_log.txt')
        self.acc_log_fname = os.path.join(args.log_dir, 'acc_log.txt')

        with open(self.mp_log_fname, 'w') as fout:
            fout.write('')
        with open(self.ad_log_fname, 'w') as fout:
            fout.write('')
        with open(self.loss_log_fname, 'w') as fout:
            fout.write('')
        with open(self.acc_log_fname, 'w') as fout:
            fout.write('')

        elapsed_list = []
        print(f'[{datetime.datetime.now()}] Start training!')
        for epoch in range(args.n_epochs):
            tic = time.time()
            self.train_epoch(epoch)
            elapsed_list.append(time.time() - tic)

            self.test(epoch)
            self.scheduler.step()
            elapsed = sum(elapsed_list) / len(elapsed_list)
            print(f'average {elapsed=}')

    def get_lp_solver_conds(self, maxpool_module, module_idx, inputs, n_module_lp_conds):
        n_S = maxpool_module.n_violated
        n_T0, n_T, n_P = 0, 0, 0
        backwarded = False
        local_input = self.net.intermediate_features[module_idx].clone()
        local_output = self.net.intermediate_out_features[module_idx].clone()

        N, C_in, H, W = local_input.size()
        _, C_out, H_out, W_out = local_output.size()

        # XXX If bacbkone is changed, it should be changed as well
        kernel_size = 3
        stride = 1
        padding = 1

        unfolded_local_input = F.unfold(local_input, kernel_size=kernel_size, stride=stride, padding=padding)
        unfolded_local_input = unfolded_local_input.reshape(N, C_in, kernel_size * kernel_size, H // stride, W // stride)
        input_patches = unfolded_local_input.permute(1, 0, 3, 4, 2)
        input_patches = input_patches.reshape(C_in, N, H // stride * W // stride, kernel_size, kernel_size)

        unfolded_local_output = F.unfold(local_output, kernel_size=kernel_size, stride=stride, padding=padding)
        unfolded_local_output = unfolded_local_output.reshape(N, C_out, kernel_size * kernel_size, H_out // stride, W_out // stride)

        # Construct the condition polytope
        passed = None
        for n, c, out_h, out_w in maxpool_module.violated_patches:
            lp_solver_conds = []
            _max_vals = maxpool_module.is_max_val_element[n, c, out_h, out_w]
            # _hw_idx: tensor([ 2,  3, 34, 35])
            _hw_idx = maxpool_module.indices_of_windows[out_h, out_w]
            # _max_hw_idx: tensor([2, 3])
            _max_hw_idx = _hw_idx[_max_vals]
            # ij_others = []
            i_star, j_star = maxpool_module.patch_indices_to_hw[maxpool_module.maxpool_indices[n, c, out_h, out_w]]
            star_patch = unfolded_local_input[n, :, :, i_star, j_star]
            star_patch_out = unfolded_local_output[n, :, :, i_star, j_star]
            # for i_other, j_other in ij_others:
            for _idx in _max_hw_idx:
                i_other, j_other = maxpool_module.patch_indices_to_hw[_idx]
                if i_star == i_other and j_star == j_other:
                    continue
                other_patch = unfolded_local_input[n, :, :, i_other, j_other]

                # Algorithm L10 for the input layer
                # Skip if the input RGB patches are the same
                if module_idx == 0:
                    if ((star_patch - other_patch) ** 2).sum() == 0:
                        n_T0 += 1
                        continue

                # Algorithm L10 for the other layers
                if module_idx > 0:
                    num = str(int((module_idx - 1) / 2 * 5 + 5))
                    x_star_range, y_star_range = receptive_field_for_unit(self.receptive_field_dict, num, (i_star, j_star), verbose=False)
                    x_oth_range, y_oth_range = receptive_field_for_unit(self.receptive_field_dict, num, (i_other, j_other), verbose=False)
                    if all([_x is not None for _x in x_star_range + y_star_range + x_oth_range + y_oth_range]):
                        # check only if the receptive field is value
                        patch_star = inputs[n, :, int(x_star_range[0]):int(x_star_range[1]), int(y_star_range[0]):int(y_star_range[1])]
                        patch_other = inputs[n, :, int(x_oth_range[0]):int(x_oth_range[1]), int(y_oth_range[0]):int(y_oth_range[1])]

                        if torch.sum((patch_star - patch_other) ** 2) < 1e-8:
                            # check if i_star/j_star and i_other/j_other are
                            # calculated from the exactly same patch or not
                            # `star_patch` and `other_patch` belong to `T_j` (Algorithm L10)
                            n_T += 1
                            continue

                    # Compare weight difference between
                    # d (_output[n, c, i_star, j_star]) / d theta
                    # d (_output[n, c, i_other, j_other]) / d theta
                    # check equivalence by < 1e-8 due to float point errors
                    # https://docs.nvidia.com/cuda/floating-point/index.html
                    if torch.sum((star_patch - other_patch) ** 2) < 1e-8:
                        backwarded = True
                        self.optimizer.zero_grad()
                        maxpool_module.in_features[n, c, i_star, j_star].backward(retain_graph=True)
                        star_grads = []
                        for _, param in sorted(self.net.named_parameters()):
                            if param.grad is not None:
                                star_grads.append(param.grad.view(-1))
                        star_grads = torch.cat(star_grads)
                        self.optimizer.zero_grad()
                        maxpool_module.in_features[n, c, i_other, j_other].backward(retain_graph=True)
                        other_grads = []
                        for _, param in sorted(self.net.named_parameters()):
                            if param.grad is not None:
                                other_grads.append(param.grad.view(-1))
                        other_grads = torch.cat(other_grads)

                        if torch.sum((star_grads - other_grads) ** 2) < 1e-8:
                            # `star_patch` and `other_patch` belong to `T_j` (Algorithm L10)
                            n_T += 1
                            continue
                lp_solver_conds.append((star_patch - other_patch).reshape(-1))
                n_P += 1
            res = self.solve_lp(lp_solver_conds, module_idx=module_idx)
            n_conds = len(lp_solver_conds)
            if n_conds:
                n_module_lp_conds.append(n_conds)
            if res and res > 0:
                passed = True
            if res and res < 0:
                return False, n_S, n_T0, n_T, n_P, backwarded
        return passed, n_S, n_T0, n_T, n_P, backwarded
    
    def solve_lp(self, lp_solver_conds, module_idx=None):
        n_conds = len(lp_solver_conds)

        # No condition to be verified
        if not n_conds:
            return None

        # Start solver
        # 1. Check wheter a trivial solution can be happened (e.g., pi_star == pi_others)
        if len(lp_solver_conds) == 1:
            if torch.sum(torch.abs(lp_solver_conds[0])) == 0:
                print(f'Failed at: {module_idx=} {len(lp_solver_conds)=} due to the same pi_star & pi_others')
                return -1
            else:
                return 1

        if any([torch.sum(torch.abs(lp_solver_conds[0])) == 0 for c in lp_solver_conds]):
            print(f'trivially failed at: {module_idx=} {len(lp_solver_conds)=} due to the same pi_star & pi_others')
            return -1

        # 2. Build LP for solving the condition
        # c = [0, 0, ..., 0, -1]
        # min c^T x                          (=> max C)
        # s.t. -[[pi* - pi], -1] * x <= 0     (=> (pi* - pi) * x >= C)

        # min c^T x
        # s.t. Ax <= 0

        # c0 * x0 + c1 * x1 + ... + c_d * x_d + c_{d+1} * C
        # => ... -1 * C

        # min [0->d, -1->1]^T [x->d, c->1]
        # s.t. [pi->d, -1] * [x->d, c->1] >= 0

        lp_solver_conds = torch.stack(lp_solver_conds).detach().cpu().numpy()
        lp_solver_conds_tensor = torch.from_numpy(lp_solver_conds)

        # shape [n_conds, x_dim]
        is_all_same_sign = torch.logical_or(
            (torch.abs(lp_solver_conds_tensor) == lp_solver_conds_tensor),
            (-torch.abs(lp_solver_conds_tensor) == lp_solver_conds_tensor)
        )

        is_all_same_sign = torch.logical_and(
            (lp_solver_conds_tensor != 0), is_all_same_sign
        )

        if any([all(sign_cond) for sign_cond in is_all_same_sign.T]):
            # If there is ANY row that satisfies all conditions are the same sign,
            # a solution ALWAYS EXISTS by setting all other variables to zero,
            # and find any scalar value that satisfies the condition.
            return 1

        n_conds, x_dim = lp_solver_conds.shape

        if n_conds < x_dim:
            rank = torch.linalg.matrix_rank(lp_solver_conds_tensor)
            if rank == n_conds:
                # A solution ALWAYS EXISTS if all rows are linearly independent.
                return 1

        c = [0 for _ in range(x_dim)] + [-1]
        A = np.pad(lp_solver_conds, ((0, 0), (0, 1)), 'constant', constant_values=-1)
        b = [0 for _ in range(n_conds)]
        res = linprog(c, A_ub=-A, b_ub=b, bounds=[(-1000000, 1000000) for _ in range(x_dim + 1)])
        if res.x is not None:
            checker = res.x[-1]
            if checker <= 0:
                print(f'failed at the solver: {module_idx=} {n_conds=} {x_dim=} {checker=}')
                # P is empty
                return -1
            else:
                # P is not empty
                return 1
        else:
            print(f'failed to search a solution: {module_idx=} {n_conds=} {x_dim=}')
            return -2

    def check_fnn_ad_correctness(self, epoch, batch_idx):
        if args.fnnactivation == 'ReLU6':
            minval = 0.0
            maxval = 6.0
        elif args.fnnactivation == 'Hardtanh':
            minval = -1.0
            maxval = 1.0
        elif args.fnnactivation == 'Hardsigmoid':
            minval = -3.0
            maxval = 3.0
        S = {0: [], 1: []}

        if any([any(outs.reshape(-1) == minval) for outs in self.net.intermediate_out_features.values()]):
            for idx, outs in self.net.intermediate_out_features.items():
                if any(outs.reshape(-1) == minval):
                    for _batch_idx, _out in enumerate(outs):
                        if any(_out == minval):
                            S[idx].append(self.net.intermediate_out_features[idx][_batch_idx])
                    print(f'{args.fnnactivation} #{idx} touched {minval}: {outs.shape}')
        elif any([any(outs.reshape(-1) == maxval) for outs in self.net.intermediate_out_features.values()]):
            for idx, outs in self.net.intermediate_out_features.items():
                if any(outs.reshape(-1) == maxval):
                    for _batch_idx, _out in enumerate(outs):
                        if any(_out == maxval):
                            S[idx].append(self.net.intermediate_out_features[idx][_batch_idx])
                    print(f'{args.fnnactivation} #{idx} touched {maxval}: {outs.shape}')
        check_fnn(S, 0, batch_idx / len(self.trainloader) + epoch, self.ad_log_fname)
        check_fnn(S, 1, batch_idx / len(self.trainloader) + epoch, self.ad_log_fname)

    def check_cnn_ad_correctness(self, inputs, module_indices, batch_idx, epoch):
        # n_S: the cardinality of the `S_j`, i.e., the number of the "violated" patches (Algorithm L7)
        # n_T0: the cardinality of the `T_j` (Algorithm L10) at the 0-th layer (the input layer)
        # n_T: the cardinality of the `T_j` (Algorithm L10) except the 0-th layer
        # n_P: the cardinality of the `P_l` (Algorithm L11)
        n_S, n_T0, n_T, n_P = 0, 0, 0, 0
        n_lp_conds = {idx: [] for idx in module_indices}
        n_failed_solver_failed = 0

        with open(self.mp_log_fname, 'a') as fout:
            fout.write(f'{batch_idx / len(self.trainloader) + epoch},')

        backwarded = False
        for module_idx, maxpool_module in self.net.maxpool_modules.items():
            if args.skip_check:
                continue

            # module_idx => 0, 1, 3, 5, 7
            with open(self.mp_log_fname, 'a') as fout:
                fout.write(f'{maxpool_module.max_n}/{maxpool_module.n_violated},')

            if maxpool_module.n_violated:
                passed, n_S, n_T0, n_T, n_P, backwarded = self.get_lp_solver_conds(maxpool_module, module_idx, inputs, n_lp_conds[module_idx])
                if passed is not None:
                    if passed:
                        continue
                    else:
                        n_failed_solver_failed = 1
                        continue
                else:
                    continue

        with open(self.ad_log_fname, 'a') as fout:
            cond_log = '/'.join([f'{idx}_{len(rows)}_{np.mean(rows):.2f}' for idx, rows in n_lp_conds.items()])
            fout.write(f'{batch_idx / len(self.trainloader) + epoch},{n_S},{n_T0},{n_T},{n_P},{cond_log},{n_failed_solver_failed == 0}\n')
        with open(self.mp_log_fname, 'a') as fout:
            fout.write('\n')
        return n_failed_solver_failed, backwarded

    def train_epoch(self, epoch):
        print('\nEpoch: %d' % epoch, flush=True)
        self.net.train()
        train_loss = 0
        correct = 0
        n_total = 0
        
        n_total_failed_solver_failed = 0

        module_indices = sorted(list(self.net.maxpool_modules.keys()))

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)

            if args.model == 'fnn':
                self.check_fnn_ad_correctness(epoch, batch_idx)
            elif not args.skip_check:
                n_failed_solver_failed, backwarded = self.check_cnn_ad_correctness(inputs, module_indices, batch_idx, epoch)
                if n_failed_solver_failed and self.args.dump_failed:
                    torch.save(
                        {'inputs': inputs,
                            'targets': targets,
                            'net': self.net},
                        os.path.join(args.log_dir, f'failed_{batch_idx}_{epoch}.pth')
                    )
                    n_total_failed_solver_failed += 1
                    if n_total_failed_solver_failed > 5:
                        print("AD is incorrect. Finalizing the program...")
                        exit(-1)

                if backwarded:
                    self.optimizer.zero_grad()
                    outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            n_total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            _loss = train_loss / (batch_idx + 1)
            _acc = 100. * correct / n_total
            if (batch_idx + 1) % self.args.log_steps == 0:
                print(f'[{datetime.datetime.now()}] [{batch_idx} / {len(self.trainloader)}] Loss: {_loss:.3f} | Acc: {_acc:.3f}% ({correct}/{n_total})', flush=True)
                with open(self.loss_log_fname, 'a') as fout:
                    fout.write(f'{batch_idx},{len(self.trainloader)},{_loss:.3f},{_acc:.3f},{correct}/{n_total}\n')
        print(f'Loss: {_loss:.3f} | Acc: {_acc:.3f}%% ({correct}/{n_total})')
        return _acc, _loss

    def test(self, epoch):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        test_loss = test_loss / (batch_idx + 1)
        print(f'Test accuracy: {acc}')
        with open(self.acc_log_fname, 'a') as fout:
            fout.write(f'{epoch},{acc}\n')
        return acc, test_loss


def main(args):
    os.makedirs(args.log_dir, exist_ok=True)
    trainer = TrainEngine(args)
    trainer.build()
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

    # model
    parser.add_argument('--model', type=str, required=True, help='model name')
    parser.add_argument('--fnnactivation', type=str, default='ReLU6', help='activation for FNN ReLU6|Hardsigmoid|Hardtanh')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset name cifar10|mnist')

    # optimizer
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--n_epochs', default=20, type=int, help='number of training epochs')

    # data laoder
    parser.add_argument('--dataset_root', default='./data', type=str, help='path to download CIFAR')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--num_workers', default=4, type=int, help='num_works for data loaders')

    # logging
    parser.add_argument('--log_dir', type=str, required=True, help='report text file (if not specified, then no record)')
    parser.add_argument('--log_steps', default=1, type=int, help='number of steps for logging')

    # debugging options
    parser.add_argument('--skip_check', action='store_true', help='skip check logics')
    parser.add_argument('--dump_failed', action='store_true', help='dump failed state')
    parser.add_argument('--dumped_weight', type=str, help='path of dumped pth file (with dump_failed option)')

    args = parser.parse_args()
    main(args)
