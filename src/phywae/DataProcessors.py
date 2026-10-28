import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

import phywae.utils as utils


_OPTIMIZED_SUFFIX = ".phywae.hdf5"
_TRAINING_DATASETS = ("phy_data", "aux_data")
_COPY_BLOCK_BYTES = 64 * 1024 * 1024


def optimized_hdf5_path(hdf5_file):
    """Return the conventional optimized-file path for a Phyddle HDF5 file."""
    path = Path(hdf5_file)
    if path.name.endswith(_OPTIMIZED_SUFFIX):
        return path
    if path.suffix.lower() in {".hdf5", ".h5", ".h5py"}:
        return path.with_suffix(_OPTIMIZED_SUFFIX)
    return path.parent / f"{path.name}{_OPTIMIZED_SUFFIX}"


def is_hdf5_optimized(hdf5_file):
    """Return whether training arrays are stored contiguously and uncompressed."""
    path = Path(hdf5_file)
    if not path.is_file():
        raise FileNotFoundError(f"Training data file does not exist: {path}")

    with h5py.File(path, "r") as h5:
        _validate_training_hdf5(h5, path)
        return all(h5[name].chunks is None for name in _TRAINING_DATASETS)


def optimize_hdf5(hdf5_file):
    """Create or reuse a contiguous copy without changing the source file."""
    source_path = Path(hdf5_file)
    if is_hdf5_optimized(source_path):
        return source_path

    output_path = optimized_hdf5_path(source_path)
    if output_path == source_path:
        raise ValueError(
            f"{source_path} has an optimized filename but not an optimized layout. "
            "Rename it before creating a new optimized copy."
        )
    if output_path.exists():
        if is_hdf5_optimized(output_path):
            return output_path
        raise FileExistsError(
            f"Refusing to overwrite existing file that does not use the current "
            f"optimized layout: {output_path}. Move or remove that generated copy, "
            "then run with --optimize_data again."
        )

    created_output = False
    try:
        with h5py.File(source_path, "r") as source, \
                h5py.File(output_path, "x") as output:
            created_output = True
            _validate_training_hdf5(source, source_path)
            for key, value in source.attrs.items():
                output.attrs[key] = value
            for name in source:
                if name in _TRAINING_DATASETS:
                    _copy_contiguous(source[name], output, name)
                else:
                    source.copy(name, output)
            output.attrs["phyloencode_format"] = "phywae"
            output.attrs["phyloencode_format_version"] = 2
    except BaseException:
        if created_output and output_path.exists():
            output_path.unlink()
        raise

    return output_path


def prepare_training_hdf5(hdf5_file, optimize=False):
    """Select the input file, optionally creating an optimized copy."""
    source_path = Path(hdf5_file)
    if is_hdf5_optimized(source_path):
        return source_path

    output_path = optimized_hdf5_path(source_path)
    if not optimize:
        print(
            f"Warning: slow HDF5 layout: {source_path}\n"
            "Run phywae train with --optimize_data on the original Phyddle file "
            "to create a faster training copy."
        )
        return source_path

    print(
        f"Warning: slow HDF5 layout: {source_path}\n"
        f"Using uncompressed training copy: {output_path} "
        "(original unchanged)."
    )
    return optimize_hdf5(source_path)


class AEData:
    """Prepare lazy HDF5-backed datasets and fitted normalizers for training.

    ``AEData`` owns the data split and normalization state for training. It reads
    just enough from the HDF5 file at construction time to choose rows and fit
    the structured and auxiliary normalizers on the training split.

    The HDF5 file is expected to contain:
        - ``phy_data`` with shape ``(N, C_file * max_tips)``. Each row is a
          flattened structured tree matrix in Fortran/column-major order. Here
          ``C_file`` is the number of stored structured channels in the file;
          only the first ``num_channels`` channels are used by ``AEData``.
        - ``aux_data`` with shape ``(N, A)`` or ``(N,)``.
        - ``aux_data_names`` with shape ``(A,)`` or ``(1, A)``. One selected
          column must be named ``"num_taxa"``. Its raw value is used when
          fitting the continuous-data normalizer and is returned with each
          dataset sample for downstream padding-mask construction.

    Public attributes used by training include ``train_dataset``,
    ``val_dataset``, ``phy_width`` (``max_tips``), ``aux_width``,
    ``ntax_cidx``, ``aux_colnames``, fitted ``phy_normalizer`` and
    ``aux_normalizer``, input-schema metadata (``num_channels``, ``num_chars``,
    ``char_data_type``, and ``max_tips``), plus train/validation shape metadata.
    """

    def __init__(
        self,
        hdf5_file: str,
        prop_train: float = 0.85,
        num_channels: int = 2,
        char_data_type: str = "categorical",
        num_chars: int = 0,
        seed: Optional[int] = None,
        max_tips: int = 1000,
        num_subset: Optional[Union[int, str]] = None,
        which_aux: Union[str, List[str]] = "all",
        optimize: bool = False,
        phy_normalizer=None,
        aux_normalizer=None,
    ):
        """Create data splits, fit normalizers, and build lazy datasets.

        Args:
            hdf5_file: Path to a Phyddle-style HDF5 file containing
                ``phy_data``, ``aux_data``, and ``aux_data_names``.
            prop_train: Fraction of the selected rows assigned to training.
                The validation split receives the remainder.
            num_channels: Number of structured channels to keep from
                ``phy_data``. If the file has more channels, channels after this
                count are ignored.
            char_data_type: Structured-data normalization mode. Use
                ``"continuous"`` for ``utils.StandardScalerPhyContinuous`` or
                ``"categorical"`` for ``utils.StandardScalerPhyCategorical``.
            num_chars: Number of trailing structured channels treated as
                character channels.
            seed: Random seed passed to the train/validation split and to the
                PyTorch ``DataLoader`` generator.
            max_tips: Structured matrix width. ``phy_data.shape[1]`` must be an
                integer multiple of this value.
            num_subset: Number of initial rows from the HDF5 file to consider,
                or ``None``/``"all"`` to use all rows.
            which_aux: ``"all"`` or an ordered list of auxiliary column names to
                keep. The selected columns must include ``"num_taxa"``.
            optimize: Create or reuse a contiguous ``.phywae.hdf5`` copy when
                the input layout is inefficient for shuffled batch reads. The
                original file is never changed.
            phy_normalizer: Fitted structured normalizer to reuse, for example
                when resuming a saved model. Supply together with
                ``aux_normalizer`` to skip fitting on this file.
            aux_normalizer: Fitted auxiliary normalizer matching the selected
                columns, in the order specified by ``which_aux``.

        Returns:
            None. The constructed object exposes datasets, normalizers, and
            shape metadata as attributes.
        """
        self.source_hdf5_file = str(hdf5_file)
        self.hdf5_file = str(prepare_training_hdf5(hdf5_file, optimize=optimize))
        self.prop_train = float(prop_train)
        self.num_channels = int(num_channels)
        self.char_data_type = char_data_type
        self.num_chars = int(num_chars)
        self.seed = seed
        self.max_tips = int(max_tips)
        self.phy_width = self.max_tips
        self.torch_g = torch.Generator().manual_seed(seed) if seed is not None else None

        with h5py.File(self.hdf5_file, "r") as h5:
            rows = np.arange(_nrows(num_subset, h5["phy_data"].shape[0]), dtype=np.int64)
            aux_names = _decode_names(h5["aux_data_names"][...])
            self.aux_colnames, aux_indices = _select_aux(aux_names, which_aux)
            self.aux_width = len(aux_indices)
            self.ntax_cidx = _num_taxa_index(self.aux_colnames)
            num_tips_aux_index = int(aux_indices[self.ntax_cidx])

            train_idx, val_idx = _split_train_val(rows, self.prop_train, seed)

            if phy_normalizer is None:
                train_phy = _read_phy(h5, train_idx, self.num_channels, self.max_tips)
                train_aux = _read_aux(h5, train_idx, aux_indices)
                self._fit_normalizers(train_phy, train_aux)
            else:
                self.phy_normalizer = phy_normalizer
                self.aux_normalizer = aux_normalizer

        self.train_phy_shape = (len(train_idx), self.num_channels * self.max_tips)
        self.val_phy_shape = (len(val_idx), self.num_channels * self.max_tips)
        self.train_aux_shape = (len(train_idx), self.aux_width)
        self.val_aux_shape = (len(val_idx), self.aux_width)

        dataset_args = dict(
            hdf5_file=self.hdf5_file,
            phy_normalizer=self.phy_normalizer,
            aux_normalizer=self.aux_normalizer,
            max_tips=self.max_tips,
            num_channels=self.num_channels,
            aux_indices=aux_indices,
            num_tips_aux_index=num_tips_aux_index,
        )
        self.train_dataset = TreeDataSet(indices=train_idx, **dataset_args)
        self.val_dataset   = TreeDataSet(indices=val_idx, **dataset_args)

    def _fit_normalizers(self, phy, aux):
        if self.char_data_type == "continuous":
            num_tips = aux[:, self.ntax_cidx].astype(np.int64)
            mask = np.arange(self.max_tips)[None, None, :] < num_tips[:, None, None]
            mask = np.broadcast_to(mask, (len(phy), self.num_channels, self.max_tips))
            self.phy_normalizer = utils.StandardScalerPhyContinuous(
                self.num_chars, self.num_channels, self.max_tips
            ).fit(phy, mask=mask.reshape(phy.shape, order="F"))
        elif self.char_data_type == "categorical":
            self.phy_normalizer = utils.StandardScalerPhyCategorical(
                self.num_chars, self.num_channels, self.max_tips
            ).fit(phy)
        else:
            raise ValueError("char_data_type must be 'continuous' or 'categorical'")
        self.aux_normalizer = utils.StandardScaler().fit(aux)

    def get_datasets(self) -> Tuple[Dataset, Dataset]:
        """Return the training and validation datasets.

        Returns:
            Tuple ``(train_dataset, val_dataset)`` where each item is a
            ``TreeDataSet``. Dataset samples are ``(phy, aux, num_tips)``
            tensors. ``phy`` has shape ``(num_channels, max_tips)``, ``aux``
            has shape ``(aux_width,)``, and ``num_tips`` is an unnormalized
            scalar tip count.
        """
        return self.train_dataset, self.val_dataset

    def get_normalizers(self):
        """Return the fitted structured and auxiliary normalizers.

        Returns:
            Tuple ``(phy_normalizer, aux_normalizer)``. ``phy_normalizer`` is a
            structured transformer fitted on flattened training ``phy_data``
            with shape ``(N_train, num_channels * max_tips)``.
            ``aux_normalizer`` is a ``utils.StandardScaler``
            fitted on training auxiliary data with shape
            ``(N_train, aux_width)``.
        """
        return self.phy_normalizer, self.aux_normalizer

    def get_dataloaders(self, batch_size=32, shuffle=True, num_workers=0) -> Tuple[DataLoader, DataLoader]:
        """Build PyTorch dataloaders for the train and validation datasets.

        Args:
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle the training dataset. Validation is not
                shuffled.
            num_workers: Number of PyTorch worker processes. Each worker opens
                its own HDF5 handle lazily through ``TreeDataSet`` and remains
                alive between epochs when multiprocessing is enabled.

        Returns:
            Tuple ``(train_dataloader, val_dataloader)``. Batches yield
            ``(phy, aux, num_tips)`` where ``phy`` has shape
            ``(B, num_channels, max_tips)``, ``aux`` has shape
            ``(B, aux_width)``, and ``num_tips`` is an unnormalized
            ``torch.int64`` tensor with shape ``(B,)``.
        """
        drop_last = (len(self.train_dataset) % batch_size) < 32
        train_sampler = RandomSampler(
            self.train_dataset, generator=self.torch_g
        ) if shuffle else None
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            drop_last=drop_last,
            generator=_clone_generator(self.torch_g),
        )
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            generator=_clone_generator(self.torch_g),
        )
        return train_dataloader, val_dataloader

class TreeDataSet(Dataset):
    """Lazy HDF5-backed PyTorch dataset for normalized tree samples.

    ``TreeDataSet`` stores row indices and fitted normalizers.
    Each worker process opens its own read-only HDF5 handle on first access.
    Each sample is read from disk, channel-selected, normalized, reshaped,
    and returned with the unnormalized tip count from the raw ``num_taxa``
    auxiliary column. Downstream consumers can use this compact value to
    construct padding masks on the compute device.
    """

    def __init__(
        self,
        hdf5_file: str,
        indices,
        phy_normalizer,
        aux_normalizer,
        max_tips: int,
        num_channels: int,
        aux_indices,
        num_tips_aux_index: int,
    ):
        """Create a lazy dataset view over selected HDF5 rows.

        Args:
            hdf5_file: Path to an HDF5 file containing ``phy_data`` and
                ``aux_data``.
            indices: One-dimensional sequence of integer row indices into the
                HDF5 datasets. Dataset index ``i`` maps to HDF5 row
                ``indices[i]``.
            phy_normalizer: Fitted transformer for flattened
                structured rows with shape ``(1, num_channels * max_tips)``.
            aux_normalizer: Fitted transformer for selected
                auxiliary rows with shape ``(1, aux_width)``.
            max_tips: Structured matrix width before padding with zeros.
            num_channels: Number of structured channels to read from
                ``phy_data``.
            aux_indices: One-dimensional sequence of auxiliary column indices
                to return, in output order.
            num_tips_aux_index: Column index in raw ``aux_data`` containing
                ``num_taxa``. This unnormalized value is returned with each
                sample.

        Returns:
            None.
        """
        super().__init__()
        
        self.hdf5_file = hdf5_file
        self.indices = np.asarray(indices, dtype=np.int64)
        self.length = len(self.indices)
        self.phy_normalizer = phy_normalizer
        self.aux_normalizer = aux_normalizer
        self.max_tips = int(max_tips)
        self.num_channels = int(num_channels)
        self.aux_col_indices = np.asarray(aux_indices, dtype=np.int64)
        self.num_tips_aux_index = int(num_tips_aux_index)
        self._h5 = None
        self._h5_pid = None

    def __len__(self):
        """Return the number of samples in this dataset view.

        Returns:
            Integer number of row indices stored in the dataset.
        """
        return self.length

    def __getitem__(self, index):
        """Read, normalize, and return one sample.

        Args:
            index: Integer dataset-relative index in ``[0, len(self))``.

        Returns:
            Tuple ``(phy, aux, num_tips)``:
                - ``phy``: ``torch.float32`` tensor with shape
                  ``(num_channels, max_tips)``.
                - ``aux``: ``torch.float32`` tensor with shape
                  ``(aux_width,)``.
                - ``num_tips``: Scalar ``torch.int64`` tensor containing the
                  unnormalized number of tips for this sample.
        """
        return self.__getitems__([index])[0]

    def __getitems__(self, indices):
        """Read and normalize samples in one batch-oriented HDF5 operation.

        Args:
            indices: Dataset-relative indices to read.

        Returns:
            List of ``(phy, aux, num_tips)`` sample tuples in the requested
            order. ``DataLoader`` collates this list into three batched
            tensors.
        """
        indices = np.asarray(indices, dtype=np.int64)
        rows = self.indices[indices]
        h5 = self._file()

        phy     = _read_phy(h5, rows, self.num_channels, self.max_tips)
        aux_raw = _as_2d(_read_rows(h5["aux_data"], rows))
        aux     = aux_raw[:, self.aux_col_indices].astype(np.float32, copy=False)

        phy = self.phy_normalizer.transform(phy)
        aux = self.aux_normalizer.transform(aux)
        phy = phy.reshape(
            (len(indices), self.num_channels, self.max_tips), order="F"
        )

        num_tips = aux_raw[:, self.num_tips_aux_index].astype(np.int64)
        # mask = np.arange(self.max_tips)[None, None, :] < num_tips[:, None, None]
        # mask = np.broadcast_to(
        #     mask, (len(indices), self.num_channels, self.max_tips)
        # ).copy()

        phy = torch.as_tensor(phy, dtype=torch.float32)
        aux = torch.as_tensor(aux, dtype=torch.float32)
        # mask = torch.as_tensor(mask, dtype=torch.bool)
        num_tips = torch.as_tensor(num_tips, dtype=torch.int64)
        # return list(zip(phy, aux, mask))
        return list(zip(phy, aux, num_tips))

    def _file(self):
        pid = os.getpid()
        if self._h5 is None or self._h5_pid != pid:
            if self._h5 is not None:
                self._h5.close()
            self._h5 = h5py.File(self.hdf5_file, "r")
            self._h5_pid = pid
        return self._h5

    def close(self):
        """Close this process's cached HDF5 file handle, if one is open.

        Returns:
            None.
        """
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
            self._h5_pid = None

    def __getstate__(self):
        """Return pickle state without an open HDF5 handle.

        PyTorch may pickle datasets when using multiprocessing start methods
        such as ``spawn``. HDF5 handles are process-local and not pickle-safe,
        so the handle fields are cleared from the serialized state.

        Returns:
            Dictionary suitable for pickling.
        """
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_h5_pid"] = None
        return state


def _clone_generator(generator):
    if generator is None:
        return None
    cloned = torch.Generator(device=generator.device)
    cloned.set_state(generator.get_state())
    return cloned


def _validate_training_hdf5(h5, path):
    required = {"phy_data", "aux_data", "aux_data_names"}
    missing = required.difference(h5.keys())
    if missing:
        raise ValueError(
            f"Training data file {path} is missing: {', '.join(sorted(missing))}"
        )

    phy = h5["phy_data"]
    aux = h5["aux_data"]
    if phy.ndim != 2:
        raise ValueError(f"phy_data must be two-dimensional, got shape {phy.shape}")
    if aux.ndim not in {1, 2}:
        raise ValueError(f"aux_data must be one- or two-dimensional, got shape {aux.shape}")
    if phy.shape[0] == 0 or phy.shape[0] != aux.shape[0]:
        raise ValueError("phy_data and aux_data must contain the same non-zero number of rows")


def _copy_contiguous(source, output, name):
    target = output.create_dataset(name, shape=source.shape, dtype=source.dtype)
    for key, value in source.attrs.items():
        target.attrs[key] = value

    values_per_row = max(1, int(np.prod(source.shape[1:])))
    bytes_per_row = values_per_row * source.dtype.itemsize
    rows_per_block = max(1, _COPY_BLOCK_BYTES // bytes_per_row)
    for start in range(0, source.shape[0], rows_per_block):
        stop = min(start + rows_per_block, source.shape[0])
        target[start:stop] = source[start:stop]


def _as_1d(x):
    return np.asarray(x).reshape(-1)


def _as_2d(x):
    x = np.asarray(x)
    return x.reshape((x.shape[0], 1)) if x.ndim == 1 else x


def _names_as_str(names):
    return np.asarray([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in np.asarray(names)])


def _decode_names(names):
    names = np.asarray(names)
    return _names_as_str(names[0] if names.ndim > 1 else names)


def _select_aux(all_names, which_aux):
    if which_aux == "all":
        idx = np.arange(len(all_names), dtype=np.int64)
        return all_names, idx

    idx = []
    for name in which_aux:
        hits = np.where(name == all_names)[0]
        if len(hits) == 0:
            raise ValueError(f"Aux column name not found in data set: {name}")
        idx.append(hits[0])
    idx = np.asarray(idx, dtype=np.int64)
    return all_names[idx], idx


def _num_taxa_index(names):
    hits = np.where(_names_as_str(names) == "num_taxa")[0]
    if len(hits) == 0:
        raise ValueError('"num_taxa" must be in aux_data.')
    return int(hits[0])


def _nrows(num_subset, total_rows):
    if num_subset is None or num_subset == "all":
        return int(total_rows)
    return int(num_subset)


def _split_train_val(rows, prop_train, seed):
    num_train = int(float(prop_train) * len(rows))
    if not 0 < num_train < len(rows):
        raise ValueError("Training and validation splits must both contain rows")
    rng = np.random if seed is None else np.random.RandomState(seed)
    shuffled = rng.permutation(len(rows))
    num_val = len(rows) - num_train
    return rows[shuffled[num_val:]], rows[shuffled[:num_val]]


def _read_rows(dataset, indices):
    indices = np.asarray(indices, dtype=np.int64)
    unique_indices, inverse = np.unique(indices, return_inverse=True)
    rows = np.asarray(dataset[unique_indices, ...], dtype=np.float32)
    return rows[inverse]


def _read_phy(h5, indices, num_channels, max_tips):
    return _select_channels(_read_rows(h5["phy_data"], indices), num_channels, max_tips)


def _read_aux(h5, indices, aux_indices):
    aux = _as_2d(_read_rows(h5["aux_data"], indices))
    return aux[:, aux_indices].astype(np.float32, copy=False)


def _select_channels(phy, num_channels, max_tips):
    phy = np.asarray(phy, dtype=np.float32)
    squeeze = phy.ndim == 1
    if squeeze:
        phy = phy.reshape(1, -1)
    phy = phy.reshape((phy.shape[0], phy.shape[1] // max_tips, max_tips), order="F")
    phy = phy[:, :num_channels, :].reshape((phy.shape[0], -1), order="F")
    return phy[0] if squeeze else phy
