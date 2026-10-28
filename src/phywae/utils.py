#!/usr/bin/env python3
import torch
import torch.nn as nn
# import torch.nn.functional as fun
# from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import List, Dict, Tuple, Optional, Union
import math
import numpy as np
import importlib.util
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os
import argparse
import dendropy as dp
import re
import pandas as pd


class _Transformer:
    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)


def _state_array(value):
    return value.detach().cpu().numpy().copy()


class StandardScaler(_Transformer):
    """Column-wise population standardization with NaNs omitted during fit.

    Statistics use float64 accumulation. Transforms preserve float32/float64
    input precision and return a copy; integer inputs become float64.
    """

    def __init__(self):
        self.mean_ = self.var_ = self.scale_ = None

    @staticmethod
    def _array(X):
        X = np.asarray(X)
        if X.dtype not in (np.dtype("float32"), np.dtype("float64")):
            X = X.astype(np.float64)
        return X

    def fit(self, X, y=None):
        X = self._array(X)
        count = np.sum(~np.isnan(X), axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.mean_ = np.nansum(X, axis=0, dtype=np.float64) / count
            residual = X - self.mean_
            correction = np.nansum(residual, axis=0, dtype=np.float64)
            # Corrected two-pass population variance avoids cancellation from
            # subtracting squared means when observations have large offsets.
            self.var_ = (
                np.nansum(residual ** 2, axis=0, dtype=np.float64)
                - correction ** 2 / count
            ) / count
        eps = np.finfo(np.float64).eps
        constant = self.var_ <= count * eps * self.var_ + (count * self.mean_ * eps) ** 2
        self.scale_ = np.sqrt(self.var_)
        self.scale_[constant] = 1.0
        return self

    def transform(self, X):
        X = self._array(X).copy(order="K")
        X -= self.mean_.astype(X.dtype, copy=False)
        X /= self.scale_.astype(X.dtype, copy=False)
        return X

    def inverse_transform(self, X):
        X = self._array(X).copy(order="K")
        X *= self.scale_.astype(X.dtype, copy=False)
        X += self.mean_.astype(X.dtype, copy=False)
        return X

    def state_dict(self):
        return {
            "version": 1,
            "type": type(self).__name__,
            **{name: torch.from_numpy(getattr(self, name + "_").copy())
               for name in ("mean", "var", "scale")},
        }

    @classmethod
    def from_state_dict(cls, state):
        scaler = cls()
        for name in ("mean", "var", "scale"):
            setattr(scaler, name + "_", _state_array(state[name]))
        return scaler


# Soft clipping functionality
class SoftClip(nn.Module):
    def __init__(self, low=1e-6):
        super().__init__()
        self.low = low

    def forward(self, x):
        return self.low + (1 - 2 * self.low) * torch.sigmoid(x)

# these classes work with datasets output from the Format step in Phyddle
# they are used to normalize the data before training
# they are used to create a DataSet object. See phywae.TreeDataSet class
class PositiveStandardScaler(_Transformer):
    # this does not use zero-padded values for standardization.
    def __init__(self):
        super().__init__()
        self.mean_ = None
        self.std_ = None
        self.scale_ = None

    def fit(self, X, y=None, mask=None):
        X = np.asarray(X)
        if mask is None:
            mask = X > 0
            mask[:, 0:2] = True  # Always include the first elements
        elif np.shape(mask) != np.shape(X):
            raise ValueError("mask must have the same shape as X")

        sum_X = np.sum(X * mask, axis=0)
        num_nonzero = np.sum(mask, axis=0)

        invalid = num_nonzero <= 1
        num_nonzero_safe = np.where(invalid, 2, num_nonzero)

        self.mean_ = sum_X / num_nonzero_safe

        residuals = (X - self.mean_) * mask
        self.std_ = np.sqrt(np.sum(residuals ** 2, axis=0) / (num_nonzero_safe - 1))

        self.mean_[invalid] = 0.
        self.std_[invalid] = 1.
        self.std_[self.std_ == 0] = 1.0
        self.scale_ = self.std_
        return self

    def transform(self, X):
        return np.array((X - self.mean_) / self.std_, dtype=np.float32)
    
    def inverse_transform(self, X):
        return np.array(X * self.std_ + self.mean_, dtype=np.float32)

    def state_dict(self):
        return {
            "version": 1,
            "type": type(self).__name__,
            "mean": torch.from_numpy(self.mean_.copy()),
            "std": torch.from_numpy(self.std_.copy()),
        }

    @classmethod
    def from_state_dict(cls, state):
        scaler = cls()
        scaler.mean_ = _state_array(state["mean"])
        scaler.std_ = _state_array(state["std"])
        scaler.scale_ = scaler.std_
        return scaler


class _StandardScalerPhy(_Transformer):
    """Shared structured-channel bookkeeping for phylogenetic scalers."""

    def __init__(self, num_chars, num_chans, max_tips):
        super().__init__()

        self.cblv_scaler = PositiveStandardScaler()

        self.mean_ = None
        self.scale_ = None
        self.std_ = None

        self.num_chars = num_chars
        self.num_chans = num_chans
        self.max_tips  = max_tips

        dim1_idx = np.arange(0, num_chans * max_tips, num_chans)
        char_idx = np.arange(num_chans - num_chars, num_chans)
        phy_idx  = np.arange(0, num_chans - num_chars)

        self.char_idxs = (dim1_idx[:, None] + char_idx).flatten()
        self.tree_idxs = (dim1_idx[:, None] + phy_idx).flatten()

    def tree_statistics(self):
        """Return fitted tree-channel means and standard deviations by channel."""
        tree_shape = (self.num_chans - self.num_chars, self.max_tips)
        return (
            self.cblv_scaler.mean_.reshape(tree_shape, order="F"),
            self.cblv_scaler.std_.reshape(tree_shape, order="F"),
        )

    def _set_tree_statistics(self):
        self.mean_ = self.cblv_scaler.mean_
        self.std_ = self.scale_ = self.cblv_scaler.std_

    def state_dict(self):
        char_scaler = getattr(self, "char_scaler", None)
        return {
            "version": 1,
            "type": type(self).__name__,
            "num_chars": int(self.num_chars),
            "num_chans": int(self.num_chans),
            "max_tips": int(self.max_tips),
            "tree_scaler": self.cblv_scaler.state_dict(),
            "char_scaler": char_scaler.state_dict() if self.num_chars and char_scaler is not None else None,
        }

    @classmethod
    def from_state_dict(cls, state):
        scaler = cls(state["num_chars"], state["num_chans"], state["max_tips"])
        scaler.cblv_scaler = PositiveStandardScaler.from_state_dict(state["tree_scaler"])
        scaler._set_tree_statistics()
        if hasattr(scaler, "char_scaler") and scaler.num_chars:
            scaler.char_scaler = StandardScaler.from_state_dict(state["char_scaler"])
        return scaler


class StandardScalerPhyCategorical(_StandardScalerPhy):
    # For one-hot encoded categorical data.
    # normalizes the tree data to have mean 0 and std 1.
    # ignores the categorical data
    # uses PositiveStandardScaler (see above)

    def fit(self, X, y=None):
        # Ensure X is a NumPy array
        X = np.asarray(X)
        self.cblv_scaler.fit(X[:, self.tree_idxs])
        self._set_tree_statistics()
        return self
    
    def transform(self, X):
        X = np.asarray(X)
        
        # Standardize the cblv-like data
        X_cblv = self.cblv_scaler.transform(X[:, self.tree_idxs])
        char_data = X[:, self.char_idxs]

        # reshape -> then concatenate X_cblv and chardata -> un-reshape
        X_cblv    = X_cblv.reshape(X.shape[0], self.num_chans - self.num_chars, self.max_tips, order = "F")
        char_data = char_data.reshape(X.shape[0], self.num_chars, self.max_tips, order= "F")
        X = np.concatenate((X_cblv, char_data), axis=1)
        X = X.reshape(X.shape[0], -1, order="F")

        return X

    def inverse_transform(self, X):
        # Convert PyTorch tensor back to NumPy array if necessary
        if isinstance(X, torch.Tensor):
            X = X.numpy()

        # separate cblv-like and character data
        X_cblv = X[:, self.tree_idxs]
        char_data = X[:, self.char_idxs]

        # Inverse transform the cblv-like data
        X_cblv = self.cblv_scaler.inverse_transform(X_cblv)

        # reshape before concatenating along axis=1 ( cblv-like data and character data)
        X_cblv = X_cblv.reshape(X.shape[0], self.num_chans - self.num_chars, self.max_tips, order="F")
        char_data = char_data.reshape(X.shape[0], self.num_chars, self.max_tips, order="F")

        # Concatenate the cblv-like and character data
        X = np.concatenate((X_cblv, char_data), axis=1)
        X = X.reshape(X.shape[0], -1, order="F")

        return X
    

class StandardScalerPhyContinuous(_StandardScalerPhy):
    """Scale each continuous character across all observed tips."""

    def __init__(self, num_chars, num_chans, max_tips):
        super().__init__(num_chars, num_chans, max_tips)
        self.char_scaler = StandardScaler()

    def fit(self, X, y=None, mask=None):
        X = np.asarray(X)
        if mask is None or np.shape(mask) != X.shape:
            raise ValueError("mask must have the same shape as X")

        self.cblv_scaler.fit(
            X[:, self.tree_idxs], mask=np.asarray(mask)[:, self.tree_idxs]
        )
        if self.num_chars:
            char_data = X[:, self.char_idxs].reshape(-1, self.num_chars)
            char_mask = np.asarray(mask)[:, self.char_idxs].reshape(
                -1, self.num_chars
            )
            char_data = np.where(char_mask, char_data, np.nan)
            char_data[:, ~char_mask.any(axis=0)] = 0.0
            self.char_scaler.fit(char_data)

        self._set_tree_statistics()
        return self

    def transform(self, X):
        X = np.asarray(X)
        transformed = np.empty_like(X, dtype=np.float32)
        transformed[:, self.tree_idxs] = self.cblv_scaler.transform(
            X[:, self.tree_idxs]
        )
        if self.num_chars:
            transformed[:, self.char_idxs] = self.char_scaler.transform(
                X[:, self.char_idxs].reshape(-1, self.num_chars)
            ).reshape(len(X), -1)
        return transformed

    def inverse_transform(self, X):
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        X = np.asarray(X)
        restored = np.empty_like(X, dtype=np.float32)
        restored[:, self.tree_idxs] = self.cblv_scaler.inverse_transform(
            X[:, self.tree_idxs]
        )
        if self.num_chars:
            restored[:, self.char_idxs] = self.char_scaler.inverse_transform(
                X[:, self.char_idxs].reshape(-1, self.num_chars)
            ).reshape(len(X), -1)
        return restored


def normalizer_from_state_dict(state):
    """Restore a supported normalizer from its explicit, versioned state."""
    if state["version"] != 1:
        raise ValueError("Unsupported normalizer state version")
    classes = (StandardScaler, PositiveStandardScaler,
               StandardScalerPhyCategorical, StandardScalerPhyContinuous)
    normalizers = {cls.__name__: cls for cls in classes}
    return normalizers[state["type"]].from_state_dict(state)


# other functions
def file_exists(fname):
    "Return error if no file"
    if not os.path.isfile(fname):
        raise argparse.ArgumentTypeError(f"File '{fname}' does not exist.")
    return fname

def read_config(config_file: str) -> Dict[str, Union[int, float, str]]:
    """
    Read a configuration file and return the settings as a dictionary.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        dict: Dictionary containing the settings.
    """
    # read a config file that contains one dictionary called settings
    spec = importlib.util.spec_from_file_location("settings", config_file)
    settings_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings_module)
    settings = settings_module.settings
    return settings

def split_params_by_wd(model: nn.Module, wd: float) -> list[dict]:
    """Create optimizer param groups with no decay on bias/norm-like parameters.

    Notes:
        Excluding 1D parameters is a common approximation that also captures most
        normalization scale vectors.
    """
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        name_l = name.lower()
        is_bias = name_l.endswith(".bias")
        is_norm = "norm" in name_l
        is_1d = param.ndim <= 1

        if is_bias or is_norm or is_1d:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": float(wd)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


# takes in 
def get_aux_data(full_aux_names : np.ndarray, 
                 aux_data : np.ndarray, 
                 which_aux : list) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Subset (and reorder) auxiliary data by column name.

    Args:
        full_aux_names: Array of auxiliary column names present in the data.
        aux_data: Array of auxiliary values with shape ``(N, A_full)``.
        which_aux: Either "all" or an ordered iterable of auxiliary column names to keep.

    Returns:
        Tuple of ``(aux_data_names, aux_data_tensor)`` where ``aux_data_names`` are the selected
        column names and ``aux_data_tensor`` has shape ``(N, A_selected)``.
    """

    full_aux_names = np.asarray([
        x.decode("utf-8") if isinstance(x, bytes) else str(x)
        for x in np.asarray(full_aux_names).flatten()
    ])

    if isinstance(which_aux, bytes):
        which_aux = which_aux.decode("utf-8")
    if which_aux is None or (isinstance(which_aux, str) and which_aux == "all"):
        return full_aux_names, torch.tensor(aux_data, dtype=torch.float32)

    which_aux = [
        x.decode("utf-8") if isinstance(x, bytes) else str(x)
        for x in np.asarray(which_aux).flatten()]

    idx_aux_data_names = []
    for aux_name in which_aux:
        if aux_name in full_aux_names:
            idx_aux_data_names.append(np.where(aux_name == full_aux_names)[0][0])
        else:
            raise ValueError(f"Aux column name not found in data set: {aux_name}")

    aux_data_names = full_aux_names[idx_aux_data_names]
    aux_data = torch.tensor(aux_data[:, idx_aux_data_names], dtype=torch.float32)
    return aux_data_names, aux_data


def get_outshape(sequential: nn.Sequential, input_channels: int, input_width: int) -> Tuple[int, int, int]:
    
    """
    Compute the output shape of a PyTorch Sequential model.
    Args:
        sequential (nn.Sequential): Sequential model.       
        input_shape (tuple): Input shape as (batch_size, channels, width).
    Returns:
        tuple: Output shape as (batch_size, channels, width).
    """
    # Initialize with input shape
    batch_size, channels, width = 1, input_channels, input_width
    # Iterate through the layers in the Sequential model
    for layer in sequential:
        if isinstance(layer, nn.Conv1d):
            width = conv1d_layer_outwidth(layer, width)
            channels = layer.out_channels
        elif isinstance(layer, torch.nn.AdaptiveAvgPool1d):
            width = layer.output_size if isinstance(layer.output_size, int) else layer.output_size[0]
        elif isinstance(layer, nn.ConvTranspose1d):
            width = tconv1d_layer_outwidth(layer, width)
            channels = layer.out_channels
        elif layer.__class__.__name__ == "ResidualBlockCNN":
            width = residual_block_outwidth(layer, width)
            channels = layer.cnn_layers[0].out_channels
        elif layer.__class__.__name__ == "ResidualBlockTransposeCNN":
            width = residual_block_transpose_outwidth(layer, width)
            channels = layer.tcnn_layers[0].out_channels
        elif isinstance(layer, nn.Sequential):
            # handle nested containers
            _, channels, width = get_outshape(layer, channels, width)
     
    return batch_size, channels, width

def residual_block_outwidth(block, input_width):
    first_conv = block.cnn_layers[0]
    return conv1d_layer_outwidth(first_conv, input_width)

def residual_block_transpose_outwidth(block, input_width):
    first_tconv = block.tcnn_layers[0]
    return tconv1d_layer_outwidth(first_tconv, input_width)

def _resolve_padding(padding):
    if isinstance(padding, str):
        return padding
    if isinstance(padding, tuple):
        if len(padding) > 0 and isinstance(padding[0], str):
            return "".join(padding)
        return padding[0]
    return padding

def conv1d_layer_outwidth(layer, input_width):
    # Extract Conv1d parameters
    kernel_size = layer.kernel_size[0]
    stride      = layer.stride[0]
    padding     = _resolve_padding(layer.padding)
    dilation    = layer.dilation[0]

    if isinstance(padding, str):
        padding_lower = padding.lower()
        if padding_lower == "same":
            return math.ceil(input_width / stride)
        if padding_lower == "valid":
            padding = 0
        else:
            raise ValueError(f"Unsupported padding '{padding}' for Conv1d layer")

    # Compute output width using the Conv1d formula
    width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    return width

def tconv1d_layer_outwidth(layer, input_width):
    # Extract ConvTranspose1d parameters
    kernel_size    = layer.kernel_size[0]
    stride         = layer.stride[0]
    padding        = _resolve_padding(layer.padding)
    output_padding = layer.output_padding[0] if isinstance(layer.output_padding, tuple) else layer.output_padding
    dilation       = layer.dilation[0]

    if isinstance(padding, str):
        padding_lower = padding.lower()
        if padding_lower == "same":
            return input_width * stride
        if padding_lower == "valid":
            padding = 0
        else:
            raise ValueError(f"Unsupported padding '{padding}' for ConvTranspose1d layer")

    # Compute output width using the ConvTranspose1d formula
    width = (input_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    return width

# TODO: May not be used anymore
def get_num_tips(phydata: np.ndarray, max_tips: int):
    """
      This computes the number of tips by finding the 
      first column in the cblv which only contains zeros

    Args:
        phydata (np.ndarray): Flattened cblv(+s)
        max_tips (int): width of all cblv

    Returns:
        torch.Tensor: number of tips in each tree
    """
        # change consitent zero in in second position to 1.
    phydata = phydata.reshape((phydata.shape[0], phydata.shape[1] // max_tips, max_tips), order = "F")
    phydata = phydata[:,0:2,:] 
    phydata[:,:,0] = 1.
    nt = torch.tensor([[np.where(np.max(phydata[x,...], axis = 0) == 0)[0][0]]
                       for x in range(phydata.shape[0])], dtype=torch.float32).view(-1,1)
    return nt + 1

def phylo_scatterplot(pred_phy_data, true_phy_data, 
                      pred_aux_data, true_aux_data, 
                      aux_names, output):
    """Create scatter plots true v pred for cblv.

    Args:
        pred_phy_data (_type_): _description_
        true_phy_data (_type_): _description_
        pred_aux_data (_type_): _description_
        true_aux_data (_type_): _description_
        aux_names (_type_): _description_
        output (_type_): _description_
    """
    # 
    with PdfPages(output + "_scatter_plots.pdf") as f:
        ndat = min(100, pred_phy_data.shape[1])
        # 9 plots per page
        # plot aux data
        for i in range(pred_aux_data.shape[1] // 9):
            fig, ax = plt.subplots(3, 3)
            fig.tight_layout(pad=2., h_pad=2., w_pad = 2.)
            aux_clabels = aux_names
            for j in range(9):
                d = 9 * i + j
                row = j // 3
                col = j % 3
                min_val = min([np.min(true_aux_data[0:ndat,d]), np.min(pred_aux_data[0:ndat,d])])
                max_val = max([np.max(true_aux_data[0:ndat,d]), np.max(pred_aux_data[0:ndat,d])])
                ax[row][col].scatter(true_aux_data[0:ndat,d], pred_aux_data[0:ndat,d], s = 5)
                ax[row][col].set_xlabel("True", fontsize = 8)
                ax[row][col].set_ylabel("Pred", fontsize = 8)
                ax[row][col].label_outer()
                ax[row][col].set_title(aux_clabels[d], fontsize = 8)
                ax[row][col].plot([min_val, max_val], [min_val, max_val],
                                   color = "r", linewidth=1)
            f.savefig(fig)
            plt.close()

        #plot cblv data
        for i in range(pred_phy_data.shape[1] // 9):
            fig, ax = plt.subplots(3, 3, sharex=True, sharey=True)
            fig.tight_layout(pad=2., h_pad=2., w_pad = 2.)
            for j in range(9):
                d = 9 * i + j
                row = j // 3
                col = j % 3
                min_val = min([np.min(true_phy_data[0:ndat,d]), np.min(pred_phy_data[0:ndat,d])])
                max_val = max([np.max(true_phy_data[0:ndat,d]), np.max(pred_phy_data[0:ndat,d])])
                ax[row][col].scatter(true_phy_data[0:ndat,d], pred_phy_data[0:ndat,d], s = 5)
                ax[row][col].set_xlabel("True", fontsize = 8)
                ax[row][col].set_ylabel("Pred", fontsize = 8)
                ax[row][col].label_outer()
                ax[row][col].set_title("dim " + str(d), fontsize = 8)
                ax[row][col].plot([min_val, max_val], [min_val, max_val],
                                   color = "r", linewidth=1)
            f.savefig(fig)
            plt.close()

def make_loss_plots(train_metrics, val_metrics = None,  *, out_prefix = "AElossplot",
                    log = True, starting_epoch = 10):
    train_history = train_metrics.epoch_history
    val_history = (
        val_metrics.epoch_history
        if val_metrics is not None
        else {name: [] for name in ("total", "phy", "char", "aux", "mmd")}
    )
    # Create subplots for the total and each loss component.
    # TODO: fix x-axis tick marks            
    num_subplots = 5
    fig, axs = plt.subplots((num_subplots + 1)//2, 2, figsize=(11, 8), sharex=True)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    fill_in_loss_comp_fig((val_history["total"], train_history["total"]),
                          "combined", axs[0,0], starting_epoch)
    axs[0,0].legend()
    fill_in_loss_comp_fig((val_history["phy"], train_history["phy"]),
                          "phy", axs[0,1], starting_epoch)
    fill_in_loss_comp_fig((val_history["char"], train_history["char"]),
                          "char", axs[1,1], starting_epoch)
    fill_in_loss_comp_fig((val_history["aux"], train_history["aux"]),
                          "aux", axs[1,0], starting_epoch)
    fill_in_loss_comp_fig((val_history["mmd"], train_history["mmd"]),
                          "mmd", axs[2,0], starting_epoch)
    axs[2,1].axis("off")
    plt.savefig(out_prefix + ".component_loss.pdf", bbox_inches='tight')
    plt.close(fig)

def fill_in_loss_comp_fig(losses, plot_label, ax, starting_epoch = 10):
    val_log_loss = _log10_positive(losses[0][starting_epoch:])
    train_log_loss = _log10_positive(losses[1][starting_epoch:])
    ax.plot(list(range(len(losses[0])))[starting_epoch:],
            val_log_loss, label="Validation", c="b")
    ax.plot(list(range(len(losses[1])))[starting_epoch:],
            train_log_loss, label="Training", c="r")
    if not np.any(np.isfinite(np.concatenate((val_log_loss, train_log_loss)))):
        ax.text(0.5, 0.5, "Zero or disabled", ha="center", transform=ax.transAxes)
    ax.set_title(f"{plot_label} Loss")
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Log10 Loss')
    # ax.set_ylabel('Loss')
    ax.grid(True)
    ax.set_xticks(ticks=np.arange(0, len(losses[0]),
                                 step=max(1, len(losses[0]) // 10)))


def _log10_positive(values):
    """Return log10 values while representing zero or negative values as NaN."""
    values = np.asarray(values, dtype=float)
    logged = np.full(values.shape, np.nan, dtype=float)
    np.log10(values, out=logged, where=values > 0)
    return logged

def set_pred_pad_to_zero(phy_pred: np.ndarray, pred_num_tips: np.ndarray) -> np.ndarray:
    """
    phy_pred: shape (batch_size, num_channels, max_tips)
    pred_num_tips: shape (batch_size,) or (batch_size, 1)
    """
    bs, nc, mt = phy_pred.shape

    # Ensure pred_num_tips is (bs, 1)
    pred_num_tips = pred_num_tips.reshape(-1, 1)

    # idx_grid: shape (bs, mt)
    idx_grid = np.arange(mt).reshape(1, -1)  # (1, mt)
    idx_grid = np.tile(idx_grid, (bs, 1))    # (bs, mt)

    # Mask: True for valid tips, False for padding
    # num_tips is a count, so valid indices are [0, num_tips).
    ntip_mask = (idx_grid < np.round(pred_num_tips, decimals=0))   # (bs, mt)

    # Expand mask to (bs, nc, mt)
    ntip_mask = np.expand_dims(ntip_mask, axis=1)

    # masked_phy_pred = phy_pred * ntip_mask
    masked_phy_pred = np.where(ntip_mask, phy_pred, 0.0)

    return masked_phy_pred

########################
# cblv -> newixk/nexus #
########################
def _get_node_heights(y, num_tips, chars = None):
    # y is a dataframe with shape (2, max_num_tips)
    # inorder node heights
    nodes = [ None ] * (2*num_tips - 1)

    for i in range(num_tips):
    
        # node heights
        int_height = y.iloc[1,i]
        tip_height = y.iloc[0,i] + int_height
        if(chars is not None):
            tip_char = {"char_" + str(k) : str(np.round(v, decimals=2).item()) 
                    for k,v in enumerate(chars[:,i])}

        # make tip node
        tip_nd = dp.Node(label=f't{i}')
        tip_nd.height = tip_height
        tip_nd.value = y.iloc[0,i]
        tip_nd.index = 2*i
        if chars is not None:
            for key, value in tip_char.items():
                tip_nd.annotations.add_new(key, value)


       # Debugging: Print annotations for the tip node
        # print(f"Tip Node {tip_nd.label} Annotations: {tip_nd.annotations}")


        nodes[tip_nd.index] = tip_nd

        # do not make int node for first tip
        if i == 0:
            continue
        
        # make int node
        int_nd = dp.Node(label=f'n{i}')
        int_nd.height = int_height
        int_nd.value = y.iloc[1,i]
        int_nd.index = 2*i - 1
        nodes[int_nd.index] = int_nd

    

    return nodes

# find oldest internal node from list
def _find_oldest_int_node(nodes):
    oldest = 1e6
    index = -1
    for i in range(1,len(nodes),2):
        if nodes[i].height <= oldest:
            index = i
            oldest = nodes[i].height
    return nodes[index]

# build (left,right) node relationships
def _recurse(nodes, nd):
    
    # tip node
    # if len(nodes) == 1:
    #     # do nothing
    #     pass
    if nd.label.startswith('t'):
        return nd


    # internal node
    else:
        # find left and right clades
        idx = [ v.index for v in nodes ].index(nd.index)
        nodes_left = nodes[:idx]
        nodes_right = nodes[(idx+1):]

        # find daughters
        nd_left  = _find_oldest_int_node(nodes_left)
        nd_right = _find_oldest_int_node(nodes_right)

        # recurse
        nd_left  = _recurse(nodes_left, nd_left)
        nd_right = _recurse(nodes_right, nd_right)

        # attach daughters
        nd.add_child(nd_left)
        nd.add_child(nd_right)

        # update edge lengths
        # for i,ch in enumerate(nd.child_nodes()):
        for ch in nd.child_nodes():
            ch.edge_length = ch.height - nd.height
    
    return nd

# print newick strings to stdout
def convert_to_newick(cblv, num_tips, char_data):
    '''
    Convert cblv format to newick format and print to stdout.
    cblv: cblv format, shape = (N, P, W), P is num phy channels
    num_tips: number of tips in each tree, shape = (N,)
    char_data: shape = (N, K, W), K is num char channels
    '''

    # loop through each tree encoded in cblv format and 
    # convert to newick string then print to stdout
    newick = []
    for i in range(cblv.shape[0]):
        num_tips_i = int(round(num_tips[i], ndigits = 0))
        if num_tips_i < 2:
            raise ValueError(f"num tips = {num_tips_i}, but should should be > 1.")
    
        # cblv is first two channels
        cblv_i = cblv[i, 0:2, 0:num_tips_i]

        char_data_i = char_data[i, :, 0:num_tips_i]
     
        nodes = _get_node_heights(pd.DataFrame(cblv_i), num_tips_i, char_data_i)
        heights = [ nd.height for nd in nodes ]        

        # find root node with smallest height and is an internal node (label starts with 'n')
        min_int_node = min([ nd.height for nd in nodes if nd.label.startswith('n') ])
        idx_root = heights.index(min_int_node)
        nd_root = nodes[idx_root]     
        nd_root = _recurse(nodes, nd_root)

        phy_decode = dp.Tree(seed_node=nd_root)

        tree_string = phy_decode.as_string("newick", suppress_leaf_node_labels=False, 
                                            suppress_annotations=False)

        newick.append(tree_string) 
        # newick.append(num_tips_i)    

    return newick


# DEV
class ShiftedStandardScaler(_Transformer):
    # standardizes the data then shifts it to be positive
    # this is used for the cblv-like data

    def __init__(self, buffer_factor = 1., copy = True):
        '''arguments:
        buffer_factor: float, default=1.0
            The factor by which to multiply the minimum value of the
            scaled data to ensure all values are positive.
            A value of 1.0 means no change in scale, while a value of 
            2.0 will shift the data to be at least twice the minimum value.'''
        
        print("Using ShiftedStandardScaler")
        super().__init__()
        self.copy = copy
        self.scaler = StandardScaler()
        self.msm = None
        self.bf = buffer_factor # default no change

    def fit(self, X, y=None):
        # Ensure X is a NumPy array
        X = np.asarray(X)
        self.scaler.fit(X)
        # find minimum value for whole dataset
        self.msm = self.bf * np.abs(np.min(self.scaler.transform(X)))

        return self
    
    def transform(self, X):
        # Ensure X is a NumPy array
        X = np.asarray(X)
        # Standardize the data
        X_scaled = self.scaler.transform(X)
        # Shift the data to make all values positive
        X_scaled_shifted = X_scaled + self.msm

        return X_scaled_shifted
    
    def inverse_transform(self, X):
        # Convert PyTorch tensor back to NumPy array if necessary
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        # Reverse the shift
        X_inv = X - self.msm
        # Reverse standardization
        return self.scaler.inverse_transform(X_inv)
   
class NoScalerNormalizer(_Transformer):
    # this just creates a normalizer object that 
    # returns the input data untransformed
    # for compatibility with downstream code
    def __init__(self, copy=True):
        self.copy = copy
        print("Using NoScalerNormalizer")
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        return X.copy() if self.copy else X

    def inverse_transform(self, X):
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        return X.copy() if self.copy else X

class LogitStandardScaler(_Transformer):
    """
    Combines a logit transformation with standard scaling.

    - Applies logit: log(x / (1 - x)), after clipping to avoid 0/1 boundaries.
    - Applies StandardScaler afterward.
    """

    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon
        self.scaler = StandardScaler()

    def _logit(self, x):
        x_clipped = np.clip(x, self.epsilon, 1 - self.epsilon)
        return np.log(x_clipped / (1 - x_clipped))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y=None):
        X_logit = self._logit(np.asarray(X))
        self.scaler.fit(X_logit)
        return self

    def transform(self, X):
        X_logit = self._logit(np.asarray(X))
        return self.scaler.transform(X_logit)

    def inverse_transform(self, X_scaled):
        X_logit = self.scaler.inverse_transform(X_scaled)
        return self._sigmoid(X_logit)
    
class StandardMinMaxScaler(_Transformer):
    def fit(self, X, y=None):
        X = np.asarray(X)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[np.where(self.std_ == 0)] = 1.
        X_std = (X - self.mean_) / self.std_
        self.min = X_std.min(axis=0)
        self.max = X_std.max(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X)
        X_std = (X - self.mean_) / self.std_
        # Avoid division by zero
        scale = np.where(self.max != self.min, self.max - self.min, 1)
        X_scaled = 2 * (X_std - self.min) / scale - 1
        return X_scaled

    def inverse_transform(self, X_scaled):
        X_std = ((X_scaled + 1) / 2) * (self.max - self.min) + self.min
        return X_std * self.std_ + self.mean_
    
class LogScaler(_Transformer):
    def __init__(self, min_pos = 1e-8):
        super().__init__()
        self.min_positive_values = min_pos

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        X = np.asarray(X)
        return(np.log(X + self.min_positive_values))

    def inverse_transform(self, X):
        return(np.exp(X) - self.min_positive_values)    

class LogStandardScaler(_Transformer):
    def __init__(self, base=np.float32(np.e)):
        super().__init__()
        self.base = base
        self.scaler = StandardScaler()
        self.min_positive_values = None

    def fit(self, X, y=None):
        # Ensure X is a NumPy array
        X = np.asarray(X)

        # Compute the smallest positive value for each feature
        self.min_positive_values = np.min(np.where(X > 0, X, np.inf), axis=0)

        # Replace infinities (features with no positive values) with 1e-8
        self.min_positive_values = np.where(np.isinf(self.min_positive_values), 
        1e-8, self.min_positive_values)

        # Shift the data to make all values strictly positive
        X_shifted = X + self.min_positive_values

        # Apply log transformation
        X_log = np.log(X_shifted) / np.log(self.base)

        # Fit the standard scaler on the log-transformed data
        self.scaler.fit(X_log)
        return self

    def transform(self, X):
        # Ensure X is a NumPy array
        X = np.asarray(X)

        # Shift the data using the precomputed smallest positive values
        X_shifted = X + self.min_positive_values

        # Apply log transformation
        X_log = np.log(X_shifted) / np.log(self.base)

        # Standardize the log-transformed data
        X_scaled = self.scaler.transform(X_log)

        return X_scaled

    def inverse_transform(self, X):
        # Convert PyTorch tensor back to NumPy array if necessary
        if isinstance(X, torch.Tensor):
            X = X.numpy()

        # Reverse standardization
        X_inv = self.scaler.inverse_transform(X)

        # Reverse log transformation
        X_exp = np.exp(X_inv * np.log(self.base))

        # Reverse the shift
        return X_exp - self.min_positive_values
