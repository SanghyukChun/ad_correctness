"""
AutoDiff Correctness Checker
Copyright (c) 2024-present Sanghyuk Chun.
MIT license
"""
import torch
import torch.nn as nn
from torch import Tensor
from torch._jit_internal import BroadcastingList2
from typing import Optional
import torch.nn.functional as F


class NewMaxPool2d(nn.Module):
    def __init__(self, kernel_size: BroadcastingList2[int],
                 stride: Optional[BroadcastingList2[int]] = None,
                 padding: BroadcastingList2[int] = 0,
                 dilation: BroadcastingList2[int] = 1,
                 return_indices: bool = False,
                 ceil_mode: bool = False,
                 skip_check: bool = False) -> None:
        super().__init__()

        if stride != 1 or padding != 1:
            raise ValueError(f"stride/padding should be 1 but got {stride=} {padding=}. Use `NewMaxPool2d_with_stride` instead.")

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.pi_nc_indices = []
        self.pi_star_indices = []
        self.pi_others_indices = []

        self.in_features = []

        self.in_hdim = 0
        self.in_wdim = 0

        self.max_n = 0
        self.n_violated = 0

        self.skip_check = skip_check

    def forward(self, input: Tensor):
        N, C, H, W = input.size()
        # Perform the standard maxpool operation
        # output: N x C x pooled_H x pooled_W
        # indices: N x C x pooled_H x pooled_W
        # indices: [ ..., [ 128,  130,  132,  ...,  154,  188,  158], ...]
        output, maxpool_indices = F.max_pool2d(input, self.kernel_size, self.stride,
                                               self.padding, self.dilation, ceil_mode=self.ceil_mode,
                                               return_indices=True)
        if self.skip_check:
            return output
        _, _, pooled_H, pooled_W = maxpool_indices.size()

        self.in_hdim = H
        self.in_wdim = W

        # Implementation of MaxPool by Unfold operation
        # https://github.com/pytorch/pytorch/pull/1523#issue-119774673

        # input: N x C x H x W
        # input_windows: N x C x pooled_H x pooled_W x kernel_H x kernel_W
        input_windows = input.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        # input_windows: N x C x pooled_H x pooled_W x (kernel_H x kernel_W)
        input_windows = input_windows.contiguous().view(*input_windows.size()[:-2], -1)
        # MaxPool operation is equivalent to the following operation:
        max_val, _ = input_windows.max(4)
        if (max_val != output).sum() != 0:
            raise ValueError('UnFold MaxPool and Torch MaxPool return different values!')

        # ===========================================================================================
        # Assertion logic for checking if `max_idx` is equal to `maxpool_indices`
        # Extracting "MaxPool2D selected index (e.g., 992)" for each patch
        # indices_of_windows: pooled_H x pooled_W x (kernel_H x kernel_W)
        # e.g.,
        #    [ 0,  1, 32, 33],
        #    ....
        #    [ 988,  989, 1020, 1021],
        #    [ 990,  991, 1022, 1023]]])
        input_indices = torch.arange(H * W).reshape(H, W)
        indices_of_windows = input_indices.unfold(0, self.kernel_size, self.stride).unfold(1, self.kernel_size, self.stride)
        indices_of_windows = indices_of_windows.contiguous().view(*indices_of_windows.size()[:-2], -1)

        # ===========================================================================================
        # Find patches having multiple maximum (positive) values
        # is_max_val_element: N x C x pooled_H x pooled_W x (kernel_H x kernel_W)
        #   [[ False, False, True, False],
        #    [ True, False, False, False],
        #    ...,
        #    [False, False, False,  True],
        #    [False, False, True,  True],   # => it means that both 3rd and 4th elements are the maximum
        #    [False, False,  False, False]]]]])  # => it means that no positive maximum value
        is_max_val_element = torch.logical_and((input_windows - output.unsqueeze(-1) == 0), input_windows > 0)

        # Find violated_patches
        # violated_patches: number_of_collision_patches x 4 (denotes the coordinate of N, C, H, W of the collision patch)
        # ([[  0,   2,   2,  15],
        #   [  0,   6,   2,  15],
        #   [  0,   7,   2,  15],
        n_max_vals = is_max_val_element.sum(-1)
        n_pos_max_vals = torch.logical_and(n_max_vals > 1, output > 0)
        violated_patches = n_pos_max_vals.nonzero()
        self.n_violated = len(violated_patches)

        # star_indices: N x C x pooled_H x pooled_W x 2 (the final value denotes the original coordinate)
        # e.g., star_indices[0, 0, 0, 0, :] = [1, 1] means that the first pooled patch pixel is selected from the (1, 1) input patch pixel.

        self.patch_indices_to_hw = {idx: (int(idx / H), idx - int(idx / H) * H) for idx in range(H * W)}
        self.is_max_val_element = is_max_val_element.detach().cpu().numpy()
        self.indices_of_windows = indices_of_windows.detach().cpu().numpy()
        self.maxpool_indices = maxpool_indices.detach().cpu().numpy()
        self.violated_patches = violated_patches.detach().cpu().numpy()
        self.max_n = 1
        for sz in output.size():
            self.max_n *= sz

        self.in_features = input
        return output


class NewMaxPool2d_with_stride(nn.Module):
    """ If your
    """
    def __init__(self, kernel_size: BroadcastingList2[int],
                 stride: Optional[BroadcastingList2[int]] = None,
                 padding: BroadcastingList2[int] = 0,
                 dilation: BroadcastingList2[int] = 1,
                 return_indices: bool = False,
                 ceil_mode: bool = False,
                 skip_check: bool = False) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.pi_nc_indices = []
        self.pi_star_indices = []
        self.pi_others_indices = []
        self.in_hdim = 0
        self.in_wdim = 0

        self.max_n = 0
        self.n_violated = 0

        self.skip_check = skip_check

        # Use the unfold operation for handling stride > 1 and padding > 1.
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)

    def forward(self, input: Tensor):
        N, C, H, W = input.size()
        # Perform the standard maxpool operation
        # output: N x C x pooled_H x pooled_W
        # indices: N x C x pooled_H x pooled_W
        # indices: [ ..., [ 128,  130,  132,  ...,  154,  188,  158], ...]
        output, maxpool_indices = F.max_pool2d(input, self.kernel_size, self.stride,
                                               self.padding, self.dilation, ceil_mode=self.ceil_mode,
                                               return_indices=True)
        if self.skip_check:
            return output
        _, _, pooled_H, pooled_W = maxpool_indices.size()

        self.in_hdim = H
        self.in_wdim = W

        # input: N x C x H x W
        # input_windows: N x (C x kernel_H x kernel_W) x (pooled_H x pooled_W)
        input_windows = self.unfold(input)
        input_windows = input_windows.reshape(N, C, self.kernel_size * self.kernel_size, pooled_H * pooled_W)
        # N x C x (pooled_H x pooled_W) x (kernel_H x kernel_W)
        input_windows = input_windows.permute(0, 1, 3, 2)
        # N x C x pooled_H x pooled_W x (kernel_H x kernel_W)
        input_windows = input_windows.reshape(N, C, pooled_H, pooled_W, self.kernel_size * self.kernel_size)

        # ===========================================================================================
        # Assertion logic for checking if `max_idx` is equal to `maxpool_indices`
        # Extracting "MaxPool2D selected index (e.g., 992)" for each patch
        # indices_of_windows: pooled_H x pooled_W x (kernel_H x kernel_W)
        # e.g.,
        #    [ 0,  1, 32, 33],
        #    ....
        #    [ 988,  989, 1020, 1021],
        #    [ 990,  991, 1022, 1023]]])
        input_indices = torch.arange(H * W).reshape(1, 1, H, W)
        indices_of_windows = self.unfold(input_indices.float())[0]
        indices_of_windows = indices_of_windows.permute(1, 0)
        # pooled_H x pooled_W x (kernel_H x kernel_W)
        indices_of_windows = indices_of_windows.reshape(pooled_H, pooled_W, self.kernel_size * self.kernel_size)

        # ===========================================================================================
        # Find patches having multiple maximum (positive) values
        # is_max_val_element: N x C x pooled_H x pooled_W x (kernel_H x kernel_W)
        #   [[ False, False, True, False],
        #    [ True, False, False, False],
        #    ...,
        #    [False, False, False,  True],
        #    [False, False, True,  True],   # => it means that both 3rd and 4th elements are the maximum
        #    [False, False,  False, False]]]]])  # => it means that no positive maximum value
        is_max_val_element = torch.logical_and((input_windows - output.unsqueeze(-1) == 0), input_windows > 0)

        #   [[      0,       0, 8388576,       0],
        #    ...,
        #    [      0,       0, 8388604, 8388605],
        #    [      0,       0,       0,       0]]]]])
        # max_val_to_nchw_idx = (is_max_val_element * all_indices_of_windows)

        # Find violated_patches
        # violated_patches: number_of_collision_patches x 4 (denotes the coordinate of N, C, H, W of the collision patch)
        # ([[  0,   2,   2,  15],
        #   [  0,   6,   2,  15],
        #   [  0,   7,   2,  15],
        n_max_vals = is_max_val_element.sum(-1)
        n_pos_max_vals = torch.logical_and(n_max_vals > 1, output > 0)
        violated_patches = n_pos_max_vals.nonzero()
        self.n_violated = len(violated_patches)

        # star_indices: N x C x pooled_H x pooled_W x 2 (the final value denotes the original coordinate)
        # e.g., star_indices[0, 0, 0, 0, :] = [1, 1] means that the first pooled patch pixel is selected from the (1, 1) input patch pixel.

        self.patch_indices_to_hw = {idx: (int(idx / H), idx - int(idx / H) * H) for idx in range(H * W)}
        self.is_max_val_element = is_max_val_element.detach().cpu().numpy()
        self.indices_of_windows = indices_of_windows.detach().cpu().numpy()
        self.maxpool_indices = maxpool_indices.detach().cpu().numpy()
        self.violated_patches = violated_patches.detach().cpu().numpy()
        self.max_n = 1
        for sz in output.size():
            self.max_n *= sz

        return output
