import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from phywae import utils
from phywae.ResNet import ResCnnEncoder, ResCnnDecoder
import random

class AECNN(nn.Module):
    """Autoencoder for structured phylogenetic tensors plus auxiliary features.

    The model encodes:

    - Structured input (e.g. CBLV-like tree tensors) with a 1D CNN (``CnnEncoder``).
    - Unstructured/auxiliary input with a small MLP (``DenseEncoder``).

    The two embeddings are concatenated and passed through a shared latent layer, then
    decoded back into:

    - Structured output with a 1D transposed CNN (``CnnDecoder``).
    - Auxiliary output with a small MLP (``DenseDecoder``).

    Character channels:
        If ``num_chars > 0``, the last ``num_chars`` structured channels are treated as
        "character" channels. In ``forward()``, the structured output is split into
        ``(phy_decoded, char_decoded)``.

    Normalization:
        Training typically expects inputs already normalized with the provided scalers.
        Use ``norm_and_encode()``, ``decode_and_denorm()``, or ``norm_predict_denorm()`` when
        you want the model to handle normalization/denormalization.
    """

    _MODEL_FORMAT_VERSION = 1

    def __init__(self, 
                 num_structured_input_channel, 
                 structured_input_width,  # Input width for structured data
                 unstructured_input_width,
                 *,
                 unstructured_latent_width = None, # must be integer multiple of num_structured_latent_channels
                 aux_inner_dim: int = 10,
                 aux_numtips_idx = None,
                 aux_data_names = None,
                 num_chars = 0,
                 char_type = "categorical", # categorical, continuous, integer (TODO: integer not implemented yet)
                 stride = [2,2],
                 kernel = [3,3],
                 out_channels = [16, 32],
                 latent_output_dim = None, # if None, then controled by structured latent channels
                 device = "auto",
                 seed = None,
                 phy_normalizer = None,
                 aux_normalizer = None):
                 
        """Constructor sets up all layers needed for network.  

        Args:
            num_structured_input_channel (int): Number of structured channels.
                For CBLV-like tree tensors this is typically ``>= 2`` (more when including
                extra features and/or character channels).
            structured_input_width (int): Structured input width (e.g., maximum number of tips).
            unstructured_input_width (int): Number of auxiliary features per sample.
            unstructured_latent_width (Optional[int]): Width of the auxiliary embedding produced
                by the unstructured encoder. If None, defaults to 10.
            aux_inner_dim (int): Hidden width used in the auxiliary encoder/decoder MLPs
                (``DenseEncoder`` and ``DenseDecoder``). Defaults to 10.
            aux_numtips_idx (Optional[int]): Column index in the auxiliary vector that contains
                the number of taxa/tips (e.g. ``num_taxa``). Required (cannot be None): used to
                constrain the decoded value to ``[2, structured_input_width]`` in normalized space.
            num_chars (int): Number of character channels in the structured input. If > 0, the
                last ``num_chars`` channels are treated as character channels. Defaults to 0.
            char_type (str): Character data type. Common values are ``"categorical"`` and
                ``"continuous"``. If ``"categorical"`` and ``inference=True``, character channels
                are softmaxed in ``decode()``. Defaults to ``"categorical"``.
            stride (List[int]): Convolution stride for each structured encoder layer. Defaults to
                ``[2, 2]``.
            kernel (List[int]): Convolution kernel size for each structured encoder layer. Defaults
                to ``[3, 3]``.
            out_channels (List[int]): Output channels for each structured encoder layer. Defaults to
                ``[16, 32]``.
            latent_output_dim (Optional[int]): Size of the shared Gaussian latent vector. If
                None, defaults to the flattened structured embedding width.
            device (str): ``"auto"``, ``"cpu"``, or ``"cuda"``. If ``"auto"``, selects CUDA when
                available. Defaults to ``"auto"``.
            seed (Optional[int]): Random seed for Python, NumPy, and PyTorch. Defaults to None.
            phy_normalizer: Fitted structured-data normalizer or its versioned state.
            aux_normalizer: Fitted auxiliary-data normalizer or its versioned state.

        Raises:
            ValueError: If ``stride``, ``kernel``, and ``out_channels`` lengths differ.
        """

        super().__init__()

        self.seed = seed
        self.device_setting = device

        self.set_seed(seed=seed)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        # Persist the resolved device choice for reproducible model config dumps.
        self.device_setting = self.device


        self.char_type = char_type
        self.num_chars = num_chars
        self.aux_numtips_idx = aux_numtips_idx
        self.aux_data_names = None if aux_data_names is None else np.asarray(aux_data_names)
        self.num_structured_input_channel = num_structured_input_channel
        self.structured_input_width       = structured_input_width
        self.unstructured_input_width     = unstructured_input_width
        self.aux_inner_dim                = aux_inner_dim

        self.char_start_idx = num_structured_input_channel - self.num_chars
        
        # check that num_chars and num_channels are compatible
        # TODO: fix this
        if num_chars > 0 and num_structured_input_channel <= 2:
            self.num_chars = 0
            Warning("""num_chars > 0 but data_channels <= 2.
                                num_structured_input_channel must be greater
                                than 2 to use character data.
                          Setting num_chars to 0""")

        if not (len(stride) == len(kernel) == len(out_channels)):
            raise ValueError("stride, kernel  and out_channels arrays should have same length," + 
                             f" but got lengths {len(stride)}, {len(kernel)}, {len(out_channels)} respectively.")

        self.set_normalizers(phy_normalizer, aux_normalizer)


        ######################################
        # Preliminaries for layer dimensions #
        ######################################

        # convolution layers parameters
        self.layer_params = {"out_channels": out_channels,
                            "kernel"      : kernel,
                            "stride"      : stride,
                            "latent_dim"  : latent_output_dim}
                
        self.latent_layer_dim = latent_output_dim
        

        # some latent layer dimensions
        self.unstructured_latent_width = unstructured_latent_width
        self.num_structured_latent_channels = out_channels[-1]
        
        # if not set, use a compact auxiliary embedding
        if unstructured_latent_width is None:
            self.unstructured_latent_width = 10
        
       
        ######################################
        # CREATE ENCODER AND DECODER LAYERS ##
        ######################################

        # Unstructured Encoder
        self.unstructured_encoder = DenseEncoder(self.unstructured_input_width, 
                                                 self.unstructured_latent_width,
                                                 hidden_width=self.aux_inner_dim)

        # Structured Encoder
        self.structured_encoder = CnnEncoder(self.num_structured_input_channel, 
                                             self.structured_input_width,
                                             self.layer_params)
        
        # Calculate final structured output width after encoder
        # get shape for latent shared concatenated layer
        # TODO: utils.get_outshape here is just used to get the encoder out_width
        #       alternatively, this can be controled by using adaptive avg pooling's out_size
        self.struct_outshape = utils.get_outshape(self.structured_encoder.cnn_layers, 
                                                  self.num_structured_input_channel, 
                                                  self.structured_input_width)
        self.struct_outwidth = self.struct_outshape[2]

        self.flat_struct_outwidth = self.struct_outshape[1] * self.struct_outshape[2]
        
        self.combined_latent_inwidth = self.flat_struct_outwidth + self.unstructured_latent_width

        self.reshaped_shared_latent_width = self.struct_outshape[2]
        self.latent_outwidth = self.flat_struct_outwidth if self.latent_layer_dim is None else self.latent_layer_dim
        self.latent_layer = LatentDense(self.combined_latent_inwidth, self.latent_outwidth)
        self.latent_layer_decoder = LatentDenseDecoder(self.latent_outwidth, self.flat_struct_outwidth)

        self.unstructured_decoder = DenseDecoder(self.flat_struct_outwidth,
                                                 self.unstructured_input_width,
                                                 hidden_width=self.aux_inner_dim)

        # Structured Decoder
        self.structured_decoder = CnnDecoder(encoder_layer_widths = self.structured_encoder.conv_out_width,
                                             latent_width         = self.reshaped_shared_latent_width,
                                             data_channels        = self.num_structured_input_channel,
                                             data_width           = self.structured_input_width,
                                             layer_params         = self.layer_params,
                                             num_chars            = self.num_chars,
                                             char_type            = self.char_type)
        
        self.to(self.device)
        
    def _runtime_device(self) -> torch.device:
        """Return the current module device based on registered parameters."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device(self.device)

    def to(self, *args, **kwargs):
        """Move module and keep ``self.device`` metadata in sync with actual placement."""
        module = super().to(*args, **kwargs)
        runtime_device = str(module._runtime_device())
        module.device = runtime_device
        module.device_setting = runtime_device
        return module

 
    def forward(self, data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a full autoencoder pass: encode then decode.

        Args:
            data (Tuple[torch.Tensor, torch.Tensor]): ``(phy, aux)`` input tensors. ``phy`` has
                shape ``(N, C, W)`` and ``aux`` has shape ``(N, A)``. Inputs are moved to
                ``self.device`` inside ``encode()``.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
            ``(phy_decoded, char_decoded, aux_decoded, latent)`` where:

            - ``phy_decoded``: Phylogeny reconstruction excluding character channels.
            - ``char_decoded``: Character reconstruction (or None if ``num_chars == 0``).
            - ``aux_decoded``: Auxiliary reconstruction.
            - ``latent``: The Gaussian latent representation.
        """

        # data is a tuple (structured, unstructured)

        # Encode
        latent = self.encode(data[0], data[1])
        
        # Decode
        structured_decoded_x, unstructured_decoded_x = self.decode(latent)

        # separate the two type of structured decoded: tree and character data
        # maybe this should be done somewhere more downstream
        if self.num_chars > 0:
            phy_decoded_x = structured_decoded_x[:,:(structured_decoded_x.shape[1]-self.num_chars),:]
            char_decoded_x = structured_decoded_x[:,(structured_decoded_x.shape[1]-self.num_chars):,:]
        else:
            phy_decoded_x = structured_decoded_x
            char_decoded_x = None
        return phy_decoded_x, char_decoded_x, unstructured_decoded_x, latent


    def encode(self, phy: torch.Tensor, aux: torch.Tensor, *,
               inference = False, detach = False ) -> torch.Tensor:
        """Encode structured and auxiliary inputs into a shared latent representation.

        Args:
            phy (torch.Tensor): PHylogeny + charecter (cblv+s) input tensor with shape ``(N, C, W)``.
            aux (torch.Tensor): Auxiliary input tensor with shape ``(N, A)``.
            inference (bool, optional): If True, runs in eval mode and disables gradients
                for the duration of the call. Defaults to False.
            detach (bool, optional): If True (and ``inference`` is True), detaches the returned
                latent tensor from the computation graph. Defaults to False.

        Returns:
            torch.Tensor: Latent tensor with shape ``(N, latent_dim)`` (flattened).
        """
        
        run_device = self._runtime_device()
        phy = phy.to(run_device)
        aux = aux.to(run_device)

        is_training = self.training
        if inference:
            self.eval()
            grad_context = torch.no_grad()
        else:
            grad_context = torch.enable_grad()

        try:        
            with grad_context:
                # get latent unstrcutured and structured embeddings
                structured_encoded_x   = self.structured_encoder(phy)    # (N, nchannels, out_width)
                unstructured_encoded_x = self.unstructured_encoder(aux)  # (N, out_width)

                # reshape structured embeddings
                flat_structured_encoded_x = structured_encoded_x.flatten(start_dim=1)                
                combined_latent = torch.cat((flat_structured_encoded_x, unstructured_encoded_x), dim=1)
        
                # get combined Gaussian latent output
                shared_latent_out = self.latent_layer(combined_latent)

            if inference and detach:
                shared_latent_out = shared_latent_out.detach()
                self.train(is_training)

            return(shared_latent_out.flatten(start_dim=1))
        
        finally:
            # restore original mode
            self.train(is_training)
    
    
    def decode(self, z: torch.Tensor, *,
               inference = False, detach = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode a latent representation back to structured and auxiliary outputs.

        Decoding applies a few constraints in normalized space:

        - Structured phylogenetic channels are clamped so their inverse-transformed values
          remain within the original ``[0, 1]`` bounds.
        - The auxiliary "num tips" field (``aux_numtips_idx``) is passed through a bounded
          transform to keep it within ``[2, structured_input_width]`` after inverse transform.
        - If ``inference=True`` and ``char_type == "categorical"``, character channels are softmaxed.

        Args:
            z (torch.Tensor): Latent tensor with shape ``(N, latent_dim)``.
            inference (bool, optional): If True, runs in eval mode and disables gradients
                for the duration of the call. Defaults to False.
            detach (bool, optional): If True (and ``inference`` is True), detaches the returned
                tensors from the computation graph. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: ``(decoded_structured, decoded_aux)`` where
            ``decoded_structured`` has shape ``(N, C, W)`` and includes both phylogenetic and
            (optionally) character channels, and ``decoded_aux`` has shape ``(N, A)``.
        """
        
        is_training = self.training
        z = z.to(self._runtime_device())

        if inference:
            self.eval()
            grad_context = torch.no_grad()
        else:
            grad_context = torch.enable_grad()

        try:
            with grad_context:
                decoded_latent_layer = self.latent_layer_decoder(z)
                # reshape
                reshaped_decoded_latent_layer = decoded_latent_layer.view(-1, self.num_structured_latent_channels, 
                                                                                self.reshaped_shared_latent_width) 
                decoded_tree = self.structured_decoder(reshaped_decoded_latent_layer)
                decoded_aux  = self.unstructured_decoder(decoded_latent_layer.flatten(start_dim=1))

                char_idx = self.char_start_idx
                tip_idx  = self.aux_numtips_idx

                # clamp phy cblv to [-mu/s, (1-mu)/s] which is [0,1] in untransformed space
                # see phyddle documentation for output format of phyddle -s F
                phy_clamp_left = self.phy_two_sided_ReLU(decoded_tree[:, :char_idx, :])
                not_phy_right  = decoded_tree[:, char_idx:, :]
                decoded_tree   = torch.cat([phy_clamp_left, not_phy_right], dim=1)

                # ensure ntips in [2, max tips] 
                left  = decoded_aux[:, :tip_idx]

                ntips = self.ntip_base + self.ntip_scale * torch.sigmoid(decoded_aux[:, tip_idx:tip_idx+1])
                # ntips = self.ntip_base + self.ntip_scale * ( 0.5 + torch.atan(decoded_aux[:, tip_idx:tip_idx+1]) / torch.pi )

                right = decoded_aux[:, tip_idx+1:]
                decoded_aux = torch.cat([left, ntips, right], dim=1)

                if inference and self.char_type == "categorical" and self.num_chars > 0:
                    # softmax the char data
                    not_char_left      = decoded_tree[:, :char_idx, :]
                    char_softmax_right = torch.softmax(decoded_tree[:, char_idx:, :], dim = 1)
                    decoded_tree       = torch.cat([not_char_left, char_softmax_right], dim = 1)


                if inference and detach:
                    decoded_tree = decoded_tree.detach()
                    decoded_aux  = decoded_aux.detach()
                    
                return decoded_tree, decoded_aux
            
        finally:
            self.train(is_training)

    # TODO: should prob output torch.Tensors like everything else?
    def predict(self, phy: torch.Tensor, aux: torch.Tensor, *, 
                inference = False, detach = False) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct inputs and return NumPy arrays on CPU.

        This is a convenience wrapper around calling the module directly and converting
        outputs to NumPy. Outputs are in the same (typically normalized) space as the model
        was trained in; use ``norm_predict_denorm()`` if you want predictions in the original
        input scale.

        Args:
            phy (torch.Tensor): Structured input tensor with shape ``(N, C, W)``.
            aux (torch.Tensor): Auxiliary input tensor with shape ``(N, A)``.
            inference (bool, optional): If True, runs in eval mode and disables gradients.
                Defaults to False.
            detach (bool, optional): If True (and ``inference`` is True), detaches the outputs
                before converting to NumPy. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: ``(phy_pred, aux_pred)`` arrays on CPU. ``phy_pred`` has
            shape ``(N, C, W)`` and ``aux_pred`` has shape ``(N, A)``.
        """
        
        # pushes data through full autoencoder
        is_training = self.training
        if inference:
            self.eval()
            grad_context = torch.no_grad()
        else:
            grad_context = torch.enable_grad()
        
        try:
            with grad_context:
                run_device = self._runtime_device()
                phy = phy.to(run_device)
                aux = aux.to(run_device)
                # model PREDICTS here
                tree_pred, char_pred, aux_pred, _ = self((phy, aux)) 

                if char_pred is not None:
                    phy_pred =  torch.cat((tree_pred, char_pred), dim = 1)
                else:
                    phy_pred = tree_pred

                if inference and detach:
                    phy_pred = phy_pred.detach()
                    aux_pred = aux_pred.detach()

                return phy_pred.cpu().numpy(), aux_pred.cpu().numpy()
        finally:
            self.train(is_training)

    def set_normalizers(self, phy_normalizer, aux_normalizer) -> None:
        """Replace fitted normalizers and refresh normalization-dependent bounds."""
        if isinstance(phy_normalizer, dict):
            phy_normalizer = utils.normalizer_from_state_dict(phy_normalizer)
        if isinstance(aux_normalizer, dict):
            aux_normalizer = utils.normalizer_from_state_dict(aux_normalizer)
        self.phy_normalizer = phy_normalizer
        self.aux_normalizer = aux_normalizer

        ntip_sd = aux_normalizer.scale_[self.aux_numtips_idx]
        if ntip_sd == 0:
            ntip_sd = 1.0
        self.ntip_base = (2.0 - aux_normalizer.mean_[self.aux_numtips_idx]) / ntip_sd
        self.ntip_scale = (self.structured_input_width - 2.0) / ntip_sd

        phy_mean, phy_sd = phy_normalizer.tree_statistics()
        device = self._runtime_device()
        phy_lower_bound = torch.as_tensor(-phy_mean / phy_sd, dtype=torch.float32, device=device)
        phy_upper_bound = torch.as_tensor((1 - phy_mean) / phy_sd, dtype=torch.float32, device=device)
        self.phy_two_sided_ReLU = TwoSidedReLU(phy_lower_bound, phy_upper_bound)
        self.phy_two_sided_ReLU.train(self.training)

    def normalizer_state_dict(self):
        """Return the fitted normalization state for model and trainer saves."""
        return {name: getattr(self, name).state_dict()
                for name in ("phy_normalizer", "aux_normalizer")}

    def __getstate__(self):
        state = super().__getstate__()
        state.update(self.normalizer_state_dict())
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.set_normalizers(self.phy_normalizer, self.aux_normalizer)

    def get_config_dict(self) -> Dict[str, object]:
        """Return constructor kwargs to recreate this model's configuration."""
        return {
            "num_structured_input_channel": self.num_structured_input_channel,
            "structured_input_width": self.structured_input_width,
            "unstructured_input_width": self.unstructured_input_width,
            "unstructured_latent_width": self.unstructured_latent_width,
            "aux_inner_dim": self.aux_inner_dim,
            "aux_numtips_idx": self.aux_numtips_idx,
            "aux_data_names": None if self.aux_data_names is None else [
                name.decode() if isinstance(name, (bytes, np.bytes_)) else str(name)
                for name in np.asarray(self.aux_data_names).reshape(-1)
            ],
            "num_chars": self.num_chars,
            "char_type": self.char_type,
            "stride": list(self.layer_params["stride"]),
            "kernel": list(self.layer_params["kernel"]),
            "out_channels": list(self.layer_params["out_channels"]),
            "latent_output_dim": self.layer_params.get("latent_dim"),
            "device": getattr(self, "device_setting", self.device),
            "seed": getattr(self, "seed", None),
            **self.normalizer_state_dict(),
        }

    def validate_data_compatibility(self, data) -> None:
        """Validate that an ``AEData`` object matches this model's input schema.

        Raises:
            ValueError: If any structured dimensions, character settings, or
                auxiliary-column metadata differ from the model configuration.
        """
        def normalize_names(names):
            if names is None:
                return None
            return tuple(
                name.decode() if isinstance(name, (bytes, np.bytes_)) else str(name)
                for name in np.asarray(names).reshape(-1)
            )

        model_schema = {
            "num_structured_input_channel": self.num_structured_input_channel,
            "structured_input_width": self.structured_input_width,
            "unstructured_input_width": self.unstructured_input_width,
            "num_chars": self.num_chars,
            "char_type": self.char_type,
            "aux_numtips_idx": self.aux_numtips_idx,
            "aux_data_names": normalize_names(self.aux_data_names),
        }
        data_schema = {
            "num_structured_input_channel": data.num_channels,
            "structured_input_width": data.phy_width,
            "unstructured_input_width": data.aux_width,
            "num_chars": data.num_chars,
            "char_type": data.char_data_type,
            "aux_numtips_idx": data.ntax_cidx,
            "aux_data_names": normalize_names(data.aux_colnames),
        }
        mismatches = [
            f"{name} (model={model_value!r}, data={data_schema[name]!r})"
            for name, model_value in model_schema.items()
            if model_value != data_schema[name]
        ]
        if mismatches:
            raise ValueError(
                "Training data schema is incompatible with the model: "
                + "; ".join(mismatches)
            )

    # inference machinery. Handles normalization too.
    def validate_num_tips(self, aux: np.ndarray) -> None:
        """Reject auxiliary data containing trees wider than this model supports."""
        max_num_tips = np.max(np.asarray(aux)[:, self.aux_numtips_idx])
        if max_num_tips > self.structured_input_width:
            raise ValueError(
                f"Input contains trees with up to {max_num_tips:g} tips, "
                f"but the model supports at most {self.structured_input_width}."
            )

    def norm_and_encode(self, phy: np.array, aux: np.array) -> np.array:
        """Normalize raw inputs and return their latent encoding.

        Args:
            phy (np.ndarray or torch.Tensor): Structured input with shape ``(N, C, W)``.
            aux (np.ndarray or torch.Tensor): Auxiliary input with shape ``(N, A)``.

        Returns:
            np.ndarray: Latent encodings with shape ``(N, latent_dim)``.
        """
        # phy shape = (N, nc, mt)
        # normalize
        # encode
        if isinstance(phy, torch.Tensor):
            phy = phy.numpy()
        if isinstance(aux, torch.Tensor):
            aux = aux.numpy()

        flat_phy = phy.reshape((phy.shape[0], -1), order = "F")
        flat_norm_phy = self.phy_normalizer.transform(flat_phy)
        norm_phy = torch.tensor(flat_norm_phy.reshape((flat_norm_phy.shape[0], 
                                          self.num_structured_input_channel, 
                                          self.structured_input_width), order = "F"), 
                                          dtype=torch.float32)
        norm_aux = torch.tensor(self.aux_normalizer.transform(aux), dtype = torch.float32)
        latent = self.encode(norm_phy, norm_aux, inference = True, detach = True)
        return latent.cpu().numpy()

    def decode_and_denorm(self, latent : np.array) -> Tuple[np.array, np.array]:
        """Decode latent vectors and inverse-transform to original data space.

        Args:
            latent (np.ndarray or torch.Tensor): Latent vectors with shape ``(N, latent_dim)``.

        Returns:
            Tuple[np.ndarray, np.ndarray]: ``(phy, aux)`` in the original (inverse-transformed)
            scale. ``phy`` has shape ``(N, C, W)`` and ``aux`` has shape ``(N, A)``.
        """
        # decode
        # inverse_transform
        # phy output shape is (N, nc, mt)
        latent = torch.tensor(latent, dtype=torch.float32)
        norm_phy, norm_aux = self.decode(latent, inference = True, detach = True)
        flat_norm_phy = norm_phy.cpu().numpy().reshape((norm_phy.shape[0], -1), order = "F")
        phy = self.phy_normalizer.inverse_transform(flat_norm_phy)
        aux = self.aux_normalizer.inverse_transform(norm_aux.cpu().numpy())
        phy = phy.reshape(norm_phy.shape, order = "F")
        return phy, aux

    def norm_predict_denorm(self, phy : np.array, aux : np.array) -> Tuple[np.array, np.array]:
        """Normalize inputs, run the autoencoder, and inverse-transform outputs.

        Args:
            phy (np.ndarray or torch.Tensor): Structured input with shape ``(N, C, W)``.
            aux (np.ndarray or torch.Tensor): Auxiliary input with shape ``(N, A)``.

        Returns:
            Tuple[np.ndarray, np.ndarray]: ``(phy_pred, aux_pred)`` in the original input scale.
        """
        # phy shape = (N, nc, mt)
        # phy output shape = (N, nc, mt)
        return self.decode_and_denorm( self.norm_and_encode(phy, aux) )
        

    def write_network_to_file(self, out_fn) -> None:
        """Write network architecture to simple text file.

        Args:
            out_fn (str): Output path for the text file.
        """

        with open(out_fn, "w") as f:  
            f.write("PHYLOGENETIC ENCODER AND DECODER:\n")    
            f.write(str(self.structured_encoder) + "\n")
            f.write(str(self.latent_layer) + "\n")
            f.write(str(self.latent_layer_decoder) + "\n")
            f.write(str(self.structured_decoder) + "\n")
            f.write("\n\nAUXILLIARY ENCODER AND DECODER:\n")
            f.write(str(self.unstructured_encoder) + "\n")
            f.write("\nSee latent layers above.\n")
            f.write(str(self.unstructured_decoder))

    def set_seed(self, seed = None):
        """Seed Python, NumPy, and PyTorch RNGs for reproducibility.

        Notes:
            This mutates global RNG state (``random``, ``numpy.random``, and ``torch``) and sets
            cuDNN to deterministic mode.

        Args:
            seed (Optional[int]): Seed value. If None, this is a no-op. Defaults to None.
        """
        if seed is None:
                return  # use module-level RNGs as-is

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def save_model(self, filename: str) -> None:
        """Saves model config and state dict to filename.

        Args:
            filename (str): Output filename.
        """
        model_settings = self.get_config_dict()
        model_state_dict = self.state_dict()
        torch.save({
            "model_format_version": self._MODEL_FORMAT_VERSION,
            "model_config": model_settings,
            "model_state_dict": model_state_dict,
        }, filename)

    @classmethod
    def load_pretrained_from_file(
        cls,
        trained_model_fn: str,
        map_location: Optional[Union[str, torch.device]] = "cpu") -> "AECNN":
        """
        Load a model artifact or trainer checkpoint with native normalization.

        Args:
            trained_model_fn (str): Input file.
            map_location (Optional[Union[str, torch.device]]): Device placement override.
                Common values are ``"cpu"``, ``"cuda"``, ``"cuda:0"``, or ``None``.
                Use ``None`` to keep original device placement from the artifact.
        """
        model_obj = torch.load(trained_model_fn, map_location=map_location, weights_only=False)
        if isinstance(model_obj, dict) and isinstance(model_obj.get("model"), cls):
            trained_model = model_obj["model"]
        elif (isinstance(model_obj, dict)
              and model_obj.get("model_format_version") == cls._MODEL_FORMAT_VERSION):
            model_config = dict(model_obj["model_config"])
            if map_location is not None:
                model_config["device"] = str(map_location)
            trained_model = cls(**model_config)
            trained_model.load_state_dict(model_obj["model_state_dict"])
        else:
            raise ValueError(
                f"Unsupported model format in {trained_model_fn}. "
                "Recreate the model with the current training pipeline."
            )
        if map_location is not None:
            trained_model.to(map_location)
        trained_model.eval()
        return trained_model

# encoder classes
class DenseEncoder(nn.Module):
    """Small MLP encoder for auxiliary (unstructured) features."""
    def __init__(self, in_width, out_width, hidden_width: int = 10):
        super().__init__()
        self.unstructured_encoder = nn.Sequential(
            nn.Linear(in_width, hidden_width),
            nn.BatchNorm1d(hidden_width),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_width, out_width),
            nn.BatchNorm1d(out_width),
        )
    def forward(self, x):
        """Encode auxiliary inputs.

        Args:
            x (torch.Tensor): Auxiliary tensor of shape ``(N, in_width)``.

        Returns:
            torch.Tensor: Encoded auxiliary tensor of shape ``(N, out_width)``.
        """
        return self.unstructured_encoder(x)

class CnnEncoder(nn.Module):
    """1D CNN encoder for structured inputs (e.g. phylogenetic tensors)."""
    def __init__(self, data_channels, data_width, layer_params):
        super().__init__()
        out_channels = layer_params['out_channels']
        kernel       = layer_params['kernel']
        stride       = layer_params['stride']
        nl = len(out_channels)
        self.cnn_layers = nn.Sequential()
        self.cnn_layers.add_module("conv1d_0", 
                                   nn.Conv1d(in_channels  = data_channels, 
                                             out_channels = out_channels[0], 
                                             kernel_size  = kernel[0], 
                                             stride       = stride[0], 
                                             padding      = kernel[0]// 2,
                                             bias         = False))  # padding = kernel_size // 2 for same padding
                                        
        conv_out_shape = utils.get_outshape(self.cnn_layers, data_channels, data_width)
        self.conv_out_width = [conv_out_shape[2]]

        # experimenting
        self.cnn_layers.add_module("norm_0", nn.BatchNorm1d(out_channels[0]))

        # add more layers if depth greater than 1.
        if len(out_channels) > 1:
            self.cnn_layers.add_module("conv_ReLU_0", nn.ReLU())
            
            for i in range(1, nl):
                self.cnn_layers.add_module("conv1d_" + str(i), 
                                            nn.Conv1d(in_channels  = out_channels[i-1], 
                                                      out_channels = out_channels[i], 
                                                      kernel_size  = kernel[i], 
                                                      stride       = stride[i], 
                                                      padding      = kernel[i]// 2,
                                                      bias         = i >= (nl-1)))                                                     
                
                conv_out_shape = utils.get_outshape(self.cnn_layers, data_channels, data_width)
                self.conv_out_width.append(conv_out_shape[2])  # bookkeeping           

                if i < (nl-1):
                    self.cnn_layers.add_module("norm_" + str(i), nn.BatchNorm1d(out_channels[i]))
                    self.cnn_layers.add_module("conv_ReLU_" + str(i), nn.ReLU())


        # TODO: adaptive average pooling TESTING
        # out_size = layer_params['latent_dim'] // out_channels[-1]
        # self.cnn_layers.add_module("adapt_avg_pool", torch.nn.AdaptiveAvgPool1d(output_size = out_size))
        # conv_out_shape = utils.get_outshape(self.cnn_layers, data_channels, data_width)
        # self.conv_out_width.append(conv_out_shape[2])
        # print(conv_out_shape)



    def forward(self, x):
        """Encode structured inputs.

        Args:
            x (torch.Tensor): Structured tensor of shape ``(N, C, W)``.

        Returns:
            torch.Tensor: Encoded structured tensor of shape ``(N, out_channels[-1], W_out)``.
        """
        return self.cnn_layers(x)
        

# decoder classes
class DenseDecoder(nn.Module):
    """Small MLP decoder for auxiliary (unstructured) features."""
    def __init__(self, in_width, out_width, hidden_width: int = 10):
        super().__init__()
        self.unstructured_decoder = nn.Sequential(
            nn.Linear(in_width, hidden_width),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_width, out_width)
        )
    def forward(self, x):
        """Decode auxiliary outputs.

        Args:
            x (torch.Tensor): Decoded latent tensor of shape ``(N, in_width)``.

        Returns:
            torch.Tensor: Reconstructed auxiliary tensor of shape ``(N, out_width)``.
        """
        return self.unstructured_decoder(x)
    
class CnnDecoder(nn.Module):
    """1D transposed-CNN decoder for structured outputs.

    This decoder mirrors the structured encoder and uses ``torch.nn.ConvTranspose1d`` layers with
    computed padding/output-padding to hit requested intermediate widths and the final
    ``data_width``.
    """
    def __init__(self, 
                 encoder_layer_widths, 
                 latent_width, 
                 data_channels, 
                 data_width, 
                 layer_params,
                 num_chars,
                 char_type):
        
        super().__init__()

        # Set up decoder dimensions
        out_channels = layer_params['out_channels']
        kernel       = layer_params['kernel']
        stride       = layer_params['stride']
        num_cnn_latent_channels = out_channels[-1]
        self.char_type          = char_type
        self.num_chars          = num_chars
        self.data_channels      = data_channels
        nl = len(out_channels)

        ########################
        # build decoder layers #
        ########################

        # get correct padding and output_padding for first decoder layer
        w_in = latent_width
        new_target_width = encoder_layer_widths[-2]
        npad, noutpad = self._get_paddings(new_target_width, w_in, stride[-1], kernel[-1])
        
        self.tcnn_layers = nn.Sequential() 
        self.tcnn_layers.add_module("trans_conv1d_0", 
                                    nn.ConvTranspose1d(
                                        in_channels     = out_channels[nl-1], 
                                        out_channels    = out_channels[nl-2], 
                                        kernel_size     = kernel[nl-1], 
                                        stride          = stride[nl-1], 
                                        padding         = npad,
                                        output_padding  = noutpad,
                                        bias            = (nl - 2) <= 0 
                                        ))

        outshape = utils.get_outshape(self.tcnn_layers,  num_cnn_latent_channels, latent_width)

        if (nl - 2) > 0:
            self.tcnn_layers.add_module("tconv_norm_0", nn.BatchNorm1d(out_channels[nl-2]))


        self.tcnn_layers.add_module("tconv_ReLU_0", nn.ReLU())        

        for i in range(nl-2, 0, -1):
            outshape = utils.get_outshape(self.tcnn_layers, num_cnn_latent_channels, latent_width)
            w_in = outshape[2]
            new_target_width = encoder_layer_widths[i-1]
            npad, noutpad = self._get_paddings(new_target_width, w_in, stride[i], kernel[i])
            self.tcnn_layers.add_module("tconv1d_" + str(nl - i - 1), 
                                        nn.ConvTranspose1d(
                                            in_channels     = out_channels[i], 
                                            out_channels    = out_channels[i-1], 
                                            kernel_size     = kernel[i], 
                                            stride          = stride[i], 
                                            padding         = npad,
                                            output_padding  = noutpad,
                                            bias            = i <= 1
                                            ))  
                      
            if i > 1:
                self.tcnn_layers.add_module("tconv_norm_" + str(nl-i-1), 
                                            nn.BatchNorm1d(out_channels[i-1]))


            self.tcnn_layers.add_module("tconv_ReLU_" + str(nl - i - 1), nn.ReLU())

        # final decoder layer
        # get correct padding and output_padding for final decoder layer
        outshape = utils.get_outshape(self.tcnn_layers, num_cnn_latent_channels, latent_width)
        width = outshape[2]
        new_target_width = data_width
        pad, outpad = self._get_paddings(new_target_width, width, stride[0], kernel[0])
        self.tcnn_layers.add_module("struct_decoder_out", 
                                    nn.ConvTranspose1d(
                                        in_channels    = out_channels[0], 
                                        out_channels   = data_channels, 
                                        kernel_size    = kernel[0], 
                                        stride         = stride[0], 
                                        padding        = pad, 
                                        output_padding = outpad
                                        ))
        
        # for character logit rescaling
        # self.phy_head_out = Head(data_channels - num_chars, data_width)
        self.char_head_out = Head(num_chars, data_width)

        
    def forward(self, x):
        """Decode structured outputs from a structured latent tensor.

        Args:
            x (torch.Tensor): Structured latent tensor of shape
                ``(N, out_channels[-1], latent_width)``.

        Returns:
            torch.Tensor: Reconstructed structured tensor of shape ``(N, data_channels, data_width)``.
        """
        decoded_x = self.tcnn_layers(x)
        char_start_idx = decoded_x.shape[1] - self.num_chars

        decoded_phylo = decoded_x[:, :char_start_idx, :]
        # decoded_phylo = self.phy_head_out(decoded_x[:, :char_start_idx, :])


        # If has categorical character data, add layers for scaling for logit outputs then
        # concatenate categorical output to phhylo output        
        if self.char_type == "categorical" and self.num_chars > 0 and self.data_channels > 2:
            # decoded_char = decoded_x[:, char_start_idx:, :]
            decoded_char = self.char_head_out(decoded_x[:, char_start_idx:, :])
            # print(self.head_out.Lin[2].weight[50:51,50:60])
            final_decoded_x = torch.cat((decoded_phylo, decoded_char), dim = 1)
        else:
            final_decoded_x = torch.cat((decoded_phylo, decoded_x[:, char_start_idx:, :]), dim = 1)

        return final_decoded_x

    def _get_paddings(self, w_out_target, w_in, s, k, d=1):
        """Compute ``padding`` and ``output_padding`` for ``torch.nn.ConvTranspose1d``.

        Args:
            w_out_target (int): Desired output width.
            w_in (int): Input width to the transpose convolution.
            s (int): Stride.
            k (int): Kernel size.
            d (int, optional): Dilation. Defaults to 1.

        Returns:
            Tuple[int, int]: ``(padding, output_padding)`` values.
        """
        # convtranspose1d formulat: 
        # w_out_target = (w_in - 1)s + d(k-1) + 1 + outpad - 2pad
        # returns the paddings necessary for target output width 
        E = s*(w_in - 1) + d*(k-1) + 1

        if (w_out_target - E) < 0:
            pad = (E - w_out_target)/2
            pad = int(pad + pad % 1)
            outpad = w_out_target - E + 2 * pad    #if (w_out_target - E + 2 * pad) <= (s-1) else 0
        elif (w_out_target - E) >=0 and (w_out_target - E) <= (s - 1):
            outpad = w_out_target - E
            pad = 0
        else:
            pad = (E - w_out_target + s - 1)/2
            pad = int(pad + pad % 1)
            outpad = s-1

        return pad, outpad

class Head(nn.Module):
    """Learned per-element scaling for structured output channels.

    Used to rescale character logits before concatenation with phylogenetic channels.
    """
    def __init__(self, c, w):
        super().__init__()
        self.c = c
        self.w = w
        self.scale = nn.Parameter(torch.ones(c * w))
        # self.head = nn.Linear(d, d)
        # self.head = nn.Sequential(nn.Linear(d, d),
        #                          nn.ReLU(),
        #                          nn.Linear(d,d))


    def forward(self, x):
        """Scale an input tensor elementwise.

        Args:
            x (torch.Tensor): Tensor of shape ``(N, c, w)``.

        Returns:
            torch.Tensor: Tensor of shape ``(N, c, w)`` with learned scaling applied.
        """
        # print(torch.max(self.scale))
        flat_x = x.view((x.shape[0], self.c * self.w))
        scaled_flat_x = self.scale * flat_x
        scaled_x = scaled_flat_x.view(x.shape[0], self.c, self.w)
        return scaled_x
        # print(torch.max(self.Lin.weight))
        # return(self.head(x))
    


# Latent layer classes
class LatentDense(nn.Module):
    """Dense latent layer operating on flattened embeddings."""
    def __init__(self, in_width: int, out_width: int):
        super().__init__()
        # Shared Latent Layer
        self.shared_layer = nn.Sequential(
            nn.Linear(in_width, out_width),
            nn.ReLU(),
            nn.Linear(out_width, out_width),
        )
    
    def forward(self, x):
        """Apply the latent MLP.

        Args:
            x (torch.Tensor): Tensor of shape ``(N, in_width)``.

        Returns:
            torch.Tensor: Tensor of shape ``(N, out_width)``.
        """
        return self.shared_layer(x)
    
    def __iter__(self):
        return iter(self.shared_layer)

    
class LatentDenseDecoder(nn.Module):
    """Dense decoder that expands the latent vector back to a flattened structured embedding."""
    def __init__(self, in_width: int, out_width: int):
        super().__init__()
        self.shared_layer = nn.Sequential(
            # nn.Identity(),
            # nn.Linear(in_width, out_width),
            # nn.BatchNorm1d(in_width),
            # nn.ELU(),
            # SoftPower(0.1, 3),
            nn.Linear(in_width, out_width),
        )
    
    def forward(self, x):
        """Decode latent vectors.

        Args:
            x (torch.Tensor): Tensor of shape ``(N, in_width)``.

        Returns:
            torch.Tensor: Tensor of shape ``(N, out_width)``.
        """
        return self.shared_layer(x)
    
    def __iter__(self):
        return iter(self.shared_layer)
# Dev
class TwoSidedReLU(nn.Module):
    """Clamp an input tensor elementwise between per-element min/max bounds."""
    def __init__(self, min_val, max_val):
        # shape of min_val and max_val is the same as the cblv tensor (N, C, W)
        super().__init__()
        min_tensor = min_val if isinstance(min_val, torch.Tensor) else torch.as_tensor(min_val)
        max_tensor = max_val if isinstance(max_val, torch.Tensor) else torch.as_tensor(max_val)
        self.register_buffer("min_val", min_tensor)
        self.register_buffer("max_val", max_tensor)

    def forward(self, x):
        return torch.clamp(x, self.min_val, self.max_val)

class SoftPower(nn.Module):
    def __init__(self, alpha = 1., power = 2):
        super().__init__()
        self.alpha = alpha
        self.power = power
    
    def forward(self, x):
        return x + self.alpha * x**self.power

class SamePadConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.k = kernel_size
        self.s = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        input_len = x.shape[-1]
        out_len = (input_len + self.s - 1) // self.s  # ceil division
        total_pad = max((out_len - 1) * self.s + self.k - input_len, 0)
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
        x = nn.functional.pad(x, (pad_left, pad_right))
        return self.conv(x)
 
