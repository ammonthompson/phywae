#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
from torch.optim.lr_scheduler import LRScheduler
from phywae.PhyLoss import PhyLoss
from phywae.DataProcessors import AEData
from phywae.PhyloAEModel import AECNN
import phywae.utils as utils
import time
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union


_NO_CHECKPOINT_OVERRIDE = object()
_LOSS_METRIC_NAMES = ("total", "phy", "char", "aux", "mmd")


class AETrainer(object):
    """Train and evaluate a phylogenetic autoencoder.

    This class is a lightweight training loop around an ``phywae.PhyloAEModel.AECNN``
    (or compatible) autoencoder. It handles:

    - Device placement and optional global seeding for reproducibility.
    - Epoch-based training and validation over PyTorch DataLoader instances.
    - Tracking and reporting detached training and validation loss metrics.

    Notes:
        Data loaders must yield ``(phy, aux, num_tips)``:

        - ``phy`` is a ``torch.Tensor`` shaped ``(batch, channels, width)``.
        - ``aux`` is a ``torch.Tensor`` shaped ``(batch, aux_dim)``.
        - ``num_tips`` is an unnormalized ``torch.int64`` tensor shaped
          ``(batch,)``.

        If ``model.num_chars > 0``, the last ``num_chars`` channels of ``phy``
        are treated as character channels and separated from the tree channels
        before loss computation. The loss constructs padding masks from
        ``num_tips`` on the compute device.

    Attributes:
        device (str): ``"cuda"`` or ``"cpu"``.
        model (AECNN): Autoencoder model being optimized.
        optimizer: PyTorch optimizer used during training.
        lr_sched: Optional learning-rate scheduler stepped once per batch.
        train_loader (Optional[torch.utils.data.DataLoader]): Training data loader.
        val_loader (Optional[torch.utils.data.DataLoader]): Validation data loader.
        total_epochs (int): Total epochs trained so far (cumulative across ``train()`` calls).
        loss (PhyLoss): Objective used for both training and validation batches.
        train_metrics (_LossMetricTracker): Training loss metric history.
        val_metrics (_LossMetricTracker): Validation loss metric history.
    """

    _CHECKPOINT_VERSION = 7

    def __init__(self,
                 model: AECNN,
                 optimizer : torch.optim.Optimizer, 
                 *, 
                 loss : PhyLoss,
                 lr_scheduler : Optional[LRScheduler] = None, 
                 seed : Optional[int] = None, 
                 device : Optional[str] = "auto",
                 track_grad : Optional[bool] = False,
                 checkpoints : Optional[list[int]] = None,
                 checkpt_file_prefix : Optional[str] = "train_out"):
        """Initialize the training loop.

        Args:
            model (AECNN): Phylogenetic autoencoder model to train. It accepts a
                ``(phy, aux)`` tuple and returns ``(tree, char, aux, latent)``.
            optimizer: Instantiated PyTorch optimizer (e.g. ``torch.optim.AdamW``) configured
                with ``model.parameters()``.
            lr_scheduler: Optional learning-rate scheduler with a ``.step()`` method.
                If provided, it is stepped once per training batch (not per epoch).
            loss: Objective used during training and validation. It is called as
                ``loss(pred, true, num_tips)`` and returns the scalar objective
                plus a component-metric mapping.
            seed (int, optional): If provided, seeds Python, NumPy, and PyTorch RNGs and
                enables deterministic cuDNN behavior for reproducibility. Defaults to None.
            device (str, optional): ``"auto"``, ``"cuda"``, or ``"cpu"``. If ``"auto"``, selects
                CUDA when available. Defaults to ``"auto"``.
            track_grad (bool, optional): If True, record per-parameter gradient norms during
                training so they can be plotted after the run. Defaults to False.
            checkpoints (list[int], optional): Epoch numbers to save checkpoints. defaults to None.
        """
        
        if not isinstance(model, AECNN):
            raise TypeError(
                f"model must be an AECNN instance, got {type(model).__name__}."
            )
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.seed = None
        if seed is not None:
            self.set_seed(seed)

        self.epoch = 0
        self.train_loader = None
        self.val_loader   = None
        self._pending_data_loader_rng_state = None
        self.model        = model
        self.model.to(self.device)
        self.optimizer = optimizer
        self.lr_sched  = lr_scheduler
        self.checkpoints = checkpoints
        self.checkpt_file_prefix = checkpt_file_prefix

        self.set_loss(loss)
        self.train_metrics = _LossMetricTracker()
        self.val_metrics = _LossMetricTracker()

        self.track_grad = False
        self.batch_layer_grad_norm = None
        self.mean_layer_grad_norm = None
        self.set_track_grad(track_grad)


    def train(self, num_epochs, seed = None):
        """Train the model for a number of epochs.

        This method requires that ``train_loader`` has been set.

        If ``val_loader`` is set, a validation pass is run after each epoch.

        Args:
            num_epochs (int): Number of epochs to train for.
            seed (int, optional): Optional seed to (re-)seed RNGs before training.
                If ``None``, the current RNG state is used. Defaults to None.

        Raises:
            ValueError: If training data has not been loaded.
        """

        if self.train_loader is None:
            raise ValueError("Must load training data fist.")

        # A loaded checkpoint has already restored the RNG state. Only an explicit seed
        # should replace that state here.
        self.set_seed(seed)
        for epoch in range(self.epoch + 1, num_epochs):
            self.epoch = epoch #bookeeping
            epoch_time = time.time()

            # perform all mini batch steps for the epoch for training data
            self._mini_batch(validation=False)
            self.train_metrics.finalize_epoch()

            # print training epoch mean losses to screen
            _print_latest_loss_metrics(
                self.train_metrics, "Train loss: ", time.time() - epoch_time,
                print_epoch=True,
            )


            # perform mini batch on validation data
            if self.val_loader is not None:
                with torch.no_grad():
                    self._mini_batch(validation=True)             
                    self.val_metrics.finalize_epoch()
                    # print epoch mean component losses to screen       
                    _print_latest_loss_metrics(
                        self.val_metrics, "Val loss:   ", time.time() - epoch_time
                    )

            # save checkpoints
            if self.checkpoints != None:
                if self.epoch in self.checkpoints:
                    self.save_checkpoint(self.checkpt_file_prefix + "_epoch_" + str(self.epoch) + ".ckpt.pt")

    def _mini_batch(self, validation = False):
        """Run one full pass over a data loader (train or validation).

        Iterates through ``train_loader`` or ``val_loader`` and
        calls the appropriate step function on each batch
        (``_train_step`` for training, ``evaluate`` for validation).

        Args:
            validation (bool, optional): If True, uses ``val_loader`` and ``evaluate()``.
                If False, uses ``train_loader`` and ``_train_step()``. Defaults
                to False.

        Returns:
            None: This method updates trainer-owned metric state as a side
                effect.
        """
        # 

        # set data loader and step function
        if validation:
            data_loader   = self.val_loader
            step_function = self.evaluate
        else:
            data_loader   = self.train_loader
            step_function = self._train_step

        if data_loader == None:
            return None

        # perform step with each batch
        for batch in data_loader:
            phy_batch, aux_batch, numtips_batch = batch

            phy_batch     = phy_batch.to(self.device)
            aux_batch     = aux_batch.to(self.device)
            numtips_batch = numtips_batch.to(self.device)
               
            # perform SGD step for batch
            step_function(phy_batch, aux_batch, numtips_batch)

        # compute mean of batch grad norms per layer
        if self.track_grad and not validation:
            for k,v in self.batch_layer_grad_norm.items():
                self.mean_layer_grad_norm[k].append(np.mean(v))
                self.batch_layer_grad_norm[k].clear()

    
    def _train_step(self, phy: torch.Tensor, aux: torch.Tensor, 
                   num_tips: torch.Tensor):
        """Run a single gradient update on one batch.

        Args:
            phy (torch.Tensor): Structured input with shape
                ``(batch, channels, width)``.
            aux (torch.Tensor): Auxiliary input with shape
                ``(batch, aux_dim)``.
            num_tips (torch.Tensor): Raw, unnormalized integer tip counts with
                shape ``(batch,)``.
        """
        # batch train loss
        # set model to train mode
        self.model.train()

        # divide phy and mask into tree and character data
        tree, char = self._split_tree_char(phy)

        true = (tree, char, aux)
        pred = self.model((phy, aux))

        objective, metrics = self.loss(pred, true, num_tips)
        self.train_metrics.record_batch(metrics)

        # compute gradient
        objective.backward()

        # record gradient for assessments
        if self.track_grad:
            self._record_grad_norm()

        # update model paremeters with gradient
        self.optimizer.step()

        # Clear the gradient for the next iteration so it doesnt accumulate 
        self.optimizer.zero_grad()

        # update learning rate according to schedule
        if self.lr_sched != None and self.lr_sched._step_count <= self.lr_sched.state_dict()['total_steps']:
            self.lr_sched.step()

        
    def evaluate(self, phy: torch.Tensor, aux: torch.Tensor,
                  num_tips: torch.Tensor):
        """Evaluate the model on one batch and record validation metrics.

        This method does not disable gradients by itself; call it under
        ``torch.no_grad()`` during evaluation.

        Args:
            phy (torch.Tensor): Structured input tensor shaped ``(batch, channels, width)``.
            aux (torch.Tensor): Unstructured/auxiliary input tensor shaped ``(batch, aux_dim)``.
            num_tips (torch.Tensor): Raw, unnormalized integer tip counts with
                shape ``(batch,)``.
        """
        
        # batch val loss        
        self.model.eval()

        # divide phy into tree and character data
        tree, char = self._split_tree_char(phy)

        pred = self.model((phy, aux))
        true = (tree, char, aux)

        _, metrics = self.loss(pred, true, num_tips)
        self.val_metrics.record_batch(metrics)
        
    def to_device(self, device):
        """Move the model and this trainer to a new device.

        Args:
            device (str): ``"cpu"`` or ``"cuda"``.
        """
        try:
            self.device = device
            self.model.to(self.device)
            self.loss.to(self.device)
        except RuntimeError:
            print(f"Didn't work, sending to {self.device} instead.")

    def set_data_loaders(self, train_loader : torch.utils.data.DataLoader, 
                               val_loader   : Optional[torch.utils.data.DataLoader] = None):
        """Set the training and validation data loaders.

        Each loader must yield ``(phy, aux, num_tips)`` as described in the
        class-level docstring.

        Args:
            train_loader (torch.utils.data.DataLoader): Training data loader.
            val_loader (torch.utils.data.DataLoader, optional): Validation data loader.
                Defaults to None.

        Notes:
            When loading a checkpoint, saved generator states are applied here so the
            next epoch uses the same sample order as uninterrupted training.

        Raises:
            TypeError: If either loader is not a ``torch.utils.data.DataLoader``.
        """
        
        self.train_loader = train_loader
        self.val_loader   = val_loader

        if not isinstance(self.train_loader, torch.utils.data.DataLoader):
            raise TypeError(f"train_loader must be a DataLoader, got {type(self.train_loader).__name__}.")

        if self.val_loader is not None and not isinstance(self.val_loader, torch.utils.data.DataLoader):
            raise TypeError(f"val_loader must be a DataLoader, got {type(self.val_loader).__name__}.")

        self._pending_data_loader_rng_state = self._restore_data_loader_rng_state(
            self._pending_data_loader_rng_state
        )

    def _get_data_loader_rng_state(self):
        """Return generator states without serializing the DataLoaders themselves."""
        rng_state = dict(self._pending_data_loader_rng_state or {})
        for name, data_loader in (
                ("train", self.train_loader), ("validation", self.val_loader)):
            if data_loader is None:
                continue
            sampler_generator = getattr(data_loader.sampler, "generator", None)
            generator = (sampler_generator if sampler_generator is not None
                         else data_loader.generator)
            if generator is not None:
                rng_state[name] = generator.get_state().clone()
        return rng_state or None

    def _restore_data_loader_rng_state(self, rng_state):
        """Apply saved states and return any awaiting a DataLoader."""
        if rng_state is None:
            return None

        pending_state = {}
        restored_generators = {}
        for name, data_loader in (
                ("train", self.train_loader), ("validation", self.val_loader)):
            if name not in rng_state:
                continue
            if data_loader is None:
                pending_state[name] = rng_state[name]
                continue
            generators = [
                data_loader.generator,
                getattr(data_loader.sampler, "generator", None),
            ]
            generators = [generator for generator in generators
                          if generator is not None]
            if not generators:
                raise ValueError(
                    f"Cannot restore {name} DataLoader RNG state: "
                    "the attached DataLoader has no generator."
                )

            saved_state = rng_state[name].cpu()
            for generator in generators:
                generator_id = id(generator)
                if generator_id in restored_generators:
                    if not torch.equal(restored_generators[generator_id], saved_state):
                        raise ValueError(
                            "Cannot restore distinct train and validation RNG states "
                            "to a shared DataLoader generator."
                        )
                    continue

                generator.set_state(saved_state)
                restored_generators[generator_id] = saved_state
        return pending_state or None


    def set_loss(self, loss: PhyLoss):
        """Set the objective used for training and validation.

        Args:
            loss: ``PhyLoss`` object called as ``loss(pred, true, num_tips)``.
        """
        if not isinstance(loss, PhyLoss):
            raise TypeError(
                f"loss must be a PhyLoss instance, got {type(loss).__name__}."
            )
        self.loss = loss
        self.loss.to(self.device)

    def set_track_grad(self, track_grad: bool = False):
        """Enable or disable gradient-norm tracking for future training steps."""
        self.track_grad = bool(track_grad)
        if self.track_grad:
            self.batch_layer_grad_norm = {
                layer_name: [] for layer_name, _ in self.model.named_parameters()
            }
            self.mean_layer_grad_norm = {
                layer_name: [] for layer_name, _ in self.model.named_parameters()
            }
        else:
            self.batch_layer_grad_norm = None
            self.mean_layer_grad_norm = None

    def _record_grad_norm(self):
        """Record parameter gradient norms for the current batch.

        This is only used when ``self.track_grad`` is enabled.
        """
        with torch.no_grad():
            for layer_name, g in self.model.named_parameters():
                # gn = g.grad.data.norm().item()
                gn = g.grad.norm().item()
                self.batch_layer_grad_norm[layer_name].append(gn)

    def plot_gradient_norms(layer_grad_norms, out_file, plots_per_page = 4):

        laynorm = [z for z in layer_grad_norms.items()]
        n_plots = len(laynorm)
        n_pages = n_plots // 4 + ((n_plots % 4) > 0)
        with PdfPages(out_file) as pdf:
            for page in range(n_pages):
                fig, axes = plt.subplots(plots_per_page // 2, 2)
                axes = axes.flatten()
                for plot_i in range(plots_per_page):
                    idx = page * plots_per_page + plot_i
                    if idx >= n_plots:
                        axes.axis('off')
                        continue
                    axes[plot_i].plot(laynorm[idx][1])
                    axes[plot_i].set_title(laynorm[idx][0], size = 6.)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)


    def make_graph(self):
        """Add the model graph to a TensorBoard writer (if configured).

        Notes:
            This method expects ``self.writer`` to be set externally to a TensorBoard
            ``SummaryWriter``-like object.
        """
        if self.train_loader and self.writer:
            x_sample, y_sample = next(iter(self.train_loader))
            self.writer.add_graph(self.model, x_sample.to(self.device))

    def set_seed(self, seed = None):
        """Seed Python, NumPy, and PyTorch RNGs for reproducibility.

        Notes:
            This mutates global RNG state (``random``, ``numpy.random``, and ``torch``) and sets
            cuDNN to deterministic mode.

        Args:
            seed (int, optional): Seed value. If None, this is a no-op. Defaults to None.
        """
        if seed is None:
            return  # use module-level RNGs as-is

        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def save_checkpoint(self, filename):
        """Save a training checkpoint to disk.

        DataLoader objects remain external, but their generator states are included
        and applied when replacement loaders are attached after loading.

        Args:
            filename (str): Output path for ``torch.save(...)``.
        """
        if Path(filename).exists():
            raise FileExistsError(
                f"Refusing to overwrite existing checkpoint: {filename}"
            )

        checkpoint = {
            'checkpoint_version': self._CHECKPOINT_VERSION,
            'model': self.model,
            'seed': self.seed,
            'epoch': self.epoch,
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_sched,
            'loss': self.loss,
            'loss_metric_state': {
                'train': self.train_metrics.state_dict(),
                'validation': self.val_metrics.state_dict(),
            },
            'trainer_state': {
                'device': self.device,
                'checkpoints': self.checkpoints,
                'checkpt_file_prefix': self.checkpt_file_prefix,
                'track_grad': self.track_grad,
                'batch_layer_grad_norm': self.batch_layer_grad_norm,
                'mean_layer_grad_norm': self.mean_layer_grad_norm,
            },
            'rng_state': self._get_rng_state(),
            'data_loader_rng_state': self._get_data_loader_rng_state(),
        }
        torch.save(checkpoint, filename)

    @classmethod
    def load_checkpoint(cls, filename, map_location : Optional[str] = "cpu",
                        track_grad=_NO_CHECKPOINT_OVERRIDE,
                        checkpoints=_NO_CHECKPOINT_OVERRIDE,
                        checkpt_file_prefix=_NO_CHECKPOINT_OVERRIDE) -> "AETrainer":
        """Restore a trainer from a checkpoint.

        Trainer configuration is restored from the checkpoint unless an explicit
        override is supplied. DataLoaders remain external and must be reattached
        with ``set_data_loaders()``, which restores their saved generator states.

        Args:
            filename (str): Checkpoint file produced by ``save_checkpoint()``.
            map_location (str, optional): Device mapping passed to ``torch.load``.
            track_grad (bool, optional): Override the saved gradient-tracking setting.
            checkpoints (list[int], optional): Override the saved checkpoint schedule.
            checkpt_file_prefix (str, optional): Override the saved checkpoint prefix.
        """
        checkpoint = torch.load(filename, map_location=map_location, weights_only=False)
        if checkpoint.get('checkpoint_version') != cls._CHECKPOINT_VERSION:
            raise ValueError(
                "Unsupported checkpoint format. "
                "Recreate the checkpoint with the current training pipeline."
            )

        model = checkpoint['model']
        opt = checkpoint['optimizer']
        lr_sched = checkpoint['lr_scheduler']
        loss = checkpoint['loss']
        loss_metric_state = checkpoint['loss_metric_state']
        trainer_state = checkpoint['trainer_state']

        if map_location is None:
            load_device = trainer_state.get('device', "auto")
        else:
            load_device = str(map_location)

        saved_track_grad = trainer_state.get('track_grad', False)
        restored_track_grad = (
            saved_track_grad
            if track_grad is _NO_CHECKPOINT_OVERRIDE
            else bool(track_grad)
        )
        restored_checkpoints = (
            trainer_state.get('checkpoints')
            if checkpoints is _NO_CHECKPOINT_OVERRIDE
            else checkpoints
        )
        restored_prefix = (
            trainer_state.get('checkpt_file_prefix', "train_out")
            if checkpt_file_prefix is _NO_CHECKPOINT_OVERRIDE
            else checkpt_file_prefix
        )

        self = cls(
            model=model,
            optimizer=opt,
            lr_scheduler=lr_sched,
            loss=loss,
            seed=checkpoint['seed'],
            device=load_device,
            track_grad=restored_track_grad,
            checkpoints=restored_checkpoints,
            checkpt_file_prefix=restored_prefix,
        )
        self.epoch = checkpoint['epoch']
        self.train_metrics.load_state_dict(loss_metric_state['train'])
        self.val_metrics.load_state_dict(loss_metric_state['validation'])
        if restored_track_grad and saved_track_grad:
            self.batch_layer_grad_norm = trainer_state.get(
                'batch_layer_grad_norm', self.batch_layer_grad_norm
            )
            self.mean_layer_grad_norm = trainer_state.get(
                'mean_layer_grad_norm', self.mean_layer_grad_norm
            )
        self._pending_data_loader_rng_state = checkpoint.get(
            'data_loader_rng_state'
        )
        self.model.train()
        self._set_rng_state(checkpoint.get('rng_state'))
        return self

    def _get_rng_state(self):
        """Return the process and trainer RNG states needed for continuation."""
        cuda_rng_state = None
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            cuda_rng_state = torch.cuda.get_rng_state_all()

        return {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'torch_cuda': cuda_rng_state,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
        }

    def _set_rng_state(self, rng_state):
        """Restore process and trainer RNG states from a checkpoint."""
        if rng_state is None:
            return

        random.setstate(rng_state['python'])
        np.random.set_state(rng_state['numpy'])
        torch.set_rng_state(rng_state['torch'].cpu())

        cuda_rng_state = rng_state.get('torch_cuda')
        if cuda_rng_state is not None and torch.cuda.is_available():
            for device_idx, state in enumerate(
                    cuda_rng_state[:torch.cuda.device_count()]):
                torch.cuda.set_rng_state(state.cpu(), device=device_idx)

        torch.backends.cudnn.deterministic = rng_state.get(
            'cudnn_deterministic', torch.backends.cudnn.deterministic
        )
        torch.backends.cudnn.benchmark = rng_state.get(
            'cudnn_benchmark', torch.backends.cudnn.benchmark
        )

    def plot_losses(self, out_prefix = "AElossplot", log = True, starting_epoch = 10):
        """Plot training and validation loss curves.

        Args:
            out_prefix (str, optional): Output filename prefix. Defaults to ``"AElossplot"``.
            log (bool, optional): If True, use a log y-scale. Defaults to True.
            starting_epoch (int, optional): First epoch to include in plot. Defaults to 10.
        """

        utils.make_loss_plots(self.train_metrics, self.val_metrics,
                              out_prefix=out_prefix, log=log,
                              starting_epoch=starting_epoch)

    def _split_tree_char(self, phy : torch.Tensor) \
                            -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Split structured input into tree and character channels.

        Args:
            phy (torch.Tensor): Structured input tensor shaped ``(batch, channels, width)``.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: ``(tree, char)``. If
                ``model.num_chars == 0``, ``char`` is None.
        """
        # divide phy into tree and character data
        if self.model.num_chars > 0:
            char_start_idx = self.model.char_start_idx
            tree = phy[:, :char_start_idx, :]
            char = phy[:, char_start_idx:, :]

        else:
            tree = phy
            char = None

        return tree, char


# Backward compatibility for imports and checkpoints created before the rename.
PhyloAutoencoder = AETrainer


class _LossMetricTracker:
    """Accumulate detached batch loss metrics and retain epoch histories."""

    def __init__(self):
        self.epoch_history = {name: [] for name in _LOSS_METRIC_NAMES}
        self._batch_history = {name: [] for name in _LOSS_METRIC_NAMES}

    def record_batch(self, metrics):
        """Record one batch of metric tensors without retaining autograd graphs."""
        missing = set(_LOSS_METRIC_NAMES).difference(metrics)
        if missing:
            raise KeyError(f"Missing loss metrics: {sorted(missing)}")

        for name in _LOSS_METRIC_NAMES:
            value = metrics[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Loss metric '{name}' must be a tensor, got "
                    f"{type(value).__name__}."
                )
            self._batch_history[name].append(value.detach())

    def finalize_epoch(self):
        """Append mean batch metrics to epoch history and clear batch history."""
        if not self._batch_history["total"]:
            raise ValueError("Cannot finalize loss metrics without any batches.")

        for name in _LOSS_METRIC_NAMES:
            values = self._batch_history[name]
            self.epoch_history[name].append(torch.stack(values).mean().item())
            values.clear()

    def state_dict(self):
        """Return all state needed to resume metric tracking."""
        return {
            "epoch_history": {
                name: list(values) for name, values in self.epoch_history.items()
            },
            "batch_history": {
                name: [value.clone() for value in values]
                for name, values in self._batch_history.items()
            },
        }

    def load_state_dict(self, state):
        """Restore metric tracking state."""
        epoch_history = state.get("epoch_history", {})
        batch_history = state.get("batch_history", {})
        self.epoch_history = {
            name: list(epoch_history.get(name, [])) for name in _LOSS_METRIC_NAMES
        }
        self._batch_history = {
            name: [value.detach() for value in batch_history.get(name, [])]
            for name in _LOSS_METRIC_NAMES
        }


# module functions
def _print_latest_loss_metrics(metrics, label, elapsed_time, print_epoch=False):
    """Print the latest epoch summary from a loss metric tracker."""
    history = metrics.epoch_history
    if print_epoch:
        print(f"Epoch {len(history['total'])}")
    print(
        f"\t {label}{history['total'][-1]:.4f},  "
        f"phy L: {history['phy'][-1]:.4f},  "
        f"char L: {history['char'][-1]:.4f},  "
        f"aux L: {history['aux'][-1]:.4f},  "
        f"MMD L: {history['mmd'][-1]:.4f},  "
        f"Run time: {elapsed_time:.3f} sec"
    )
