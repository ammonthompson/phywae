#!/usr/bin/env python3
"""Train a phylogenetic Wasserstein autoencoder (PhyWAE).

Training uses four main objects; AEData, AECNN, PhyLoss and AETrainer.
The ``phywae train`` command creates or restores these objects and starts
training.

`API:`
``AEData`` processes the input data and creates the data loaders.
``AECNN`` holds the PhyWAE neural network.
``PhyLoss`` holds and computes the WAE loss.
``AETrainer`` is the trainer object that uses AEData and PhyLoss to
train AECNN. The trainer manages the optimizer, scheduler, losses, and
checkpoints.

"""

import random
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import numpy  as np
import pandas as pd

from phywae.PhyloAutoencoder   import AETrainer
from phywae.PhyloAEModel       import AECNN
from phywae.DataProcessors     import AEData
from phywae.PhyLoss            import PhyLoss
from phywae.cli._output        import print_output_files
import phywae.utils as utils


def main(args):
    settings, checkpoint_overrides = _process_settings(args)
    map_location = None if settings["device"] == "auto" else settings["device"]
    tree_ae = None
    if settings["resume_from_checkpoint"] is not None:
        tree_ae = AETrainer.load_checkpoint(
            settings["resume_from_checkpoint"], map_location=map_location,
            **checkpoint_overrides,
        )
        settings["seed"] = tree_ae.seed
    else:
        _set_seed(settings)

    ###################################
    # Set up network training objects #
    ###################################
    # AEData processes and holds the datasets, normalizers, dimensions, and column metadata.
    # It splits HDF5 rows, fits normalizers on training data, and builds lazy datasets.
    ae_data     = AEData(
                        hdf5_file        = args.trn_data,
                        prop_train       = settings['proportion_train'],
                        num_channels     = settings["num_channels"],
                        char_data_type   = settings["char_type"],
                        num_chars        = settings["num_chars"],
                        seed             = settings["seed"],
                        max_tips         = settings["max_tips"],
                        num_subset       = settings["num_subset"],
                        which_aux        = settings["which_aux"],
                        optimize         = settings["optimize_data"],
                        phy_normalizer   = tree_ae.model.phy_normalizer if tree_ae else None,
                        aux_normalizer   = tree_ae.model.aux_normalizer if tree_ae else None,
                        )
    output_files = []
    if settings["optimize_data"] and ae_data.hdf5_file != args.trn_data:
        output_files.append(ae_data.hdf5_file)
    aux_data_names = ae_data.aux_colnames
    settings["train_phy_shape"] = ae_data.train_phy_shape
    settings["val_phy_shape"]   = ae_data.val_phy_shape
    settings["train_aux_shape"] = ae_data.train_aux_shape
    settings["val_aux_shape"]   = ae_data.val_aux_shape
    
    trn_loader, val_loader = \
        ae_data.get_dataloaders(settings["batch_size"], shuffle = True,
                                 num_workers = settings["num_workers"])
    
    phy_normalizer, aux_normalizer = ae_data.get_normalizers()

    if tree_ae is not None:
        tree_ae.model.validate_data_compatibility(ae_data)
    else:
        # create model
        if settings["pretrained_model"] is not None:
            ae_model = AECNN.load_pretrained_from_file(
                        settings["pretrained_model"],
                        map_location = map_location
                        )
        else:
            ae_model = AECNN(
                            num_structured_input_channel  = ae_data.num_channels, 
                            structured_input_width        = ae_data.phy_width,
                            unstructured_input_width      = ae_data.aux_width,
                            aux_inner_dim                 = settings["aux_inner_dim"],
                            aux_numtips_idx               = ae_data.ntax_cidx,
                            aux_data_names                = aux_data_names,
                            stride                        = settings["stride"],
                            kernel                        = settings["kernel"],
                            out_channels                  = settings["out_channels"],
                            latent_output_dim             = settings["latent_output_dim"],
                            num_chars                     = settings["num_chars"],
                            char_type                     = settings["char_type"],
                            device                        = settings["device"],
                            phy_normalizer                = phy_normalizer,
                            aux_normalizer                = aux_normalizer
                            )

        ae_model.validate_data_compatibility(ae_data)
        if settings["pretrained_model"] is not None:
            ae_model.set_normalizers(phy_normalizer, aux_normalizer)

        # Setup optimizer and lr scheduler
        lr = settings['learning_rate']
        wd = settings['weight_decay']
        opt = AdamW(utils.split_params_by_wd(ae_model, wd), lr=lr)
        lr_schedlr = OneCycleLR(
                            opt,
                            max_lr=lr,
                            epochs=settings["num_epochs"], 
                            steps_per_epoch=len(trn_loader),
                            pct_start=0.1,               # 10% warmup
                            anneal_strategy='cos',
                            cycle_momentum=False
                            )
        
        # Create loss objetc. PhyLoss computes the objective and component metrics.
        loss_weights = _get_loss_weights(settings)
        loss = PhyLoss(
                        loss_weights,
                        ae_model.aux_numtips_idx,
                        ae_model.char_type,
                        mmd_num_kernels = settings["mmd_num_kernels"]
                        )


        # AETrainer is the model trainer
        # the model, the data, and the loss come together here
        tree_ae = AETrainer(
                            model           = ae_model, 
                            optimizer       = opt, 
                            lr_scheduler    = lr_schedlr,
                            loss            = loss,
                            device          = settings["device"],
                            track_grad      = settings["track_grad"],
                            checkpoints     = settings["checkpoints"],
                            checkpt_file_prefix = settings["out_prefix"]
                            )

    # Create some data for output files describing model attributes etc.
    if settings["resume_from_checkpoint"] is not None:
        settings["track_grad"] = tree_ae.track_grad
        settings["checkpoints"] = tree_ae.checkpoints
        settings["out_prefix"] = tree_ae.checkpt_file_prefix
        settings["mmd_num_kernels"] = tree_ae.loss.mmd.num_kernels
    settings_out_file = settings['out_prefix'] + "_settings.csv"
    network_out_file = settings["out_prefix"] + ".network.txt"
    _save_settings(settings, settings_out_file)
    tree_ae.model.write_network_to_file(network_out_file)
    output_files.extend([settings_out_file, network_out_file])


    ###############################################################
    # Use tree_ae to train ae_model with data from ae_data.       #
    ###############################################################
    tree_ae.set_data_loaders(train_loader=trn_loader, val_loader=val_loader) 

    train_seed = None if settings["resume_from_checkpoint"] is not None else settings["seed"]
    starting_epoch = tree_ae.epoch

    tree_ae.train(num_epochs = settings["num_epochs"], seed = train_seed)

    if tree_ae.checkpoints is not None:
        output_files.extend(
            settings["out_prefix"] + "_epoch_" + str(epoch) + ".ckpt.pt"
            for epoch in range(starting_epoch + 1, settings["num_epochs"])
            if epoch in tree_ae.checkpoints
        )

    # insepect layer gradient norms if interested
    if tree_ae.track_grad:
        gradient_out_file = settings["out_prefix"] + ".layer_grad_norms.pdf"
        tree_ae.plot_gradient_norms(tree_ae.mean_layer_grad_norm, gradient_out_file)
        output_files.append(gradient_out_file)

    # save model with normalizers
    model_out_file = settings["out_prefix"] + ".ae_trained.pt"
    tree_ae.model.save_model(model_out_file)
    output_files.append(model_out_file)

    # plot loss curves
    tree_ae.plot_losses(settings["out_prefix"])
    output_files.append(settings["out_prefix"] + ".component_loss.pdf")

    print_output_files(output_files)


# Helpers
def _process_settings(args):
    # Training settings: Architecture, num epochs, batch size, etc.
    # Override settings provided in config file if provided
    settings = _get_default_settings()
    checkpoint_override_names = {
        "track_grad": "track_grad",
        "checkpoints": "checkpoints",
        "out_prefix": "checkpt_file_prefix",
    }
    overridden_settings = set()
    if args.config:
        config = utils.read_config(args.config)
        _update_settings_from_config(settings = settings, config = config)
        overridden_settings.update(config.keys() & checkpoint_override_names.keys())
    # override settings provided as command line arguments
    _update_settings_from_command_line(settings = settings, args = args)
    settings["which_aux"] = _normalize_which_aux(settings["which_aux"])
    for setting_name in checkpoint_override_names:
        if getattr(args, setting_name) is not None:
            overridden_settings.add(setting_name)

    if settings["resume_from_checkpoint"] is not None and settings["pretrained_model"] is not None:
        raise ValueError("Cannot specify both --resume_from_checkpoint and --pretrained_model.")
    checkpoint_overrides = {
        checkpoint_override_names[name]: settings[name]
        for name in overridden_settings
    }
    return settings, checkpoint_overrides

def add_arguments(parser):
    parser.add_argument("-d", "--trn_data",         required = True,  help = "Training data in hdf5 format.")
    parser.add_argument("-o", "--out_prefix",       required = False,  help = "Output prefix.")
    parser.add_argument("-cfg", "--config",         required = False, help = "Configuration file. Settings dictionary. Default None.")
    parser.add_argument("-s", "--seed",             required = False, type = int, help = "Random seed. Default random.")
    parser.add_argument("-nw", "--num_workers",     required = False, type = int, help = "Number of workers. Default 0")
    parser.add_argument("--optimize_data", action="store_true", default=None, help = "Create or reuse an uncompressed .phywae.hdf5 copy for faster shuffled batch loading.")
    parser.add_argument("-ne", "--num_epochs",      required = False, type = int, help = "Number of training epochs. Default 100")
    parser.add_argument("-b", "--batch_size",       required = False, type = int, help = "Batch size. Default 128")
    parser.add_argument("-aid", "--aux_inner_dim",  required = False, type = int, help = "Hidden width of auxiliary encoder/decoder MLPs. Default 10")
    parser.add_argument("-mmd","--mmd_loss_weight", required = False, type = float, help = "MMD lambda (>= 0). Default 1.0")
    parser.add_argument("--mmd-num-kernels", required=False, type=int, help="Positive odd number of MMD RBF kernels. Default 3")
    parser.add_argument("-pw", "--phy_loss_weight", required = False, type = float, help = "Phylogenetic loss weight. Default 0.9")
    parser.add_argument("-aw", "--aux_loss_weight", required = False, type = float, help = "Auxiliary loss weight. Default 0.1")
    parser.add_argument("-cw", "--char_loss_weight",required = False, type = float, help = "how much weight to give to char loss. Default 0.0")
    parser.add_argument("-mt", "--max_tips",        required = False, type = float, help = "maximum number of tips.   Default 1000")
    parser.add_argument("-ns", "--num_subset",      required = False, type = int , help = "subset of data used for training/testing. Default None")
    parser.add_argument("-nchans", "--num_channels",  required = False, type = int, help = "number of data channels. Default 9")
    parser.add_argument("-num_chars", "--num_chars",  required = False, type = int, help = "number of characters. Default 5")
    parser.add_argument("-ld", "--latent_output_dim", required = False, type = int, help = "latent output dimension. Default None (determined by structured encoder output shape)")
    parser.add_argument("-k", "--kernel",           required = False, type = int, help = "kernel size. Default 3,5,5")
    parser.add_argument("-r", "--stride",           required = False, type = int, help = "stride size. Default 2,4,4")
    parser.add_argument("-oc", "--out_channels",    required = False, type = int, help = "output channels. Default 32,32,128")
    parser.add_argument("-ct", "--char_type",       required = False, help = "character type (categorical or continuous). Default categorical")
    parser.add_argument("-pt", "--proportion-train",required = False, type = float, help = "Proportion of num-subset used for training vs validation. Default 0.85")
    parser.add_argument("-lr", "--learning-rate",   required = False, type = float, help = "Optimizer learning rate. Default 1e-3")
    parser.add_argument("-wd", "--weight-decay",    required = False, type = float, help = "Optimizer weight decay. Default 1e-3")
    parser.add_argument("-t", "--testing",          required = False, type = bool, help = "Testing mode sets torch trianing optimization behavior to deterministic. Default True")
    parser.add_argument("-tg", "--track_grad",      required = False, help = "Track parameter gradient norms during training. Default False")
    parser.add_argument("-dv", "--device",          required = False, type = bool, help = "Device. Default auto")
    parser.add_argument("-waux", "--which_aux",     required = False, help = "Comma separated list of auxilliary data column names to inclued. Default: All")
    parser.add_argument("-ckpt", "--checkpoints", required = False, help = "Comma separated list of epochs to save training checkpoints. Default: None")
    parser.add_argument("--resume_from_checkpoint", required = False, help = "Resume training from a provided checkpoint file produced by save_checkpoint.")
    parser.add_argument("--pretrained_model", required = False, help = "Initialize model weights from a model artifact or trainer checkpoint and start a fresh training run.")

def _get_loss_weights(settings):
    return {k : v for k, v in settings.items() if "_loss_weight" in k}

def _set_seed(settings):
    if settings['seed'] is None:
        settings['seed'] = np.random.randint(0, 2**32 - 1)
    seed = settings['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("seed: ", seed)
    if settings['testing']:
        # torch does some random stuff for more efficient training.
        # causes slight differences despite same seed. Use below for exact.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def _get_default_settings():
    return {
        "testing": True,
        "out_prefix": "out",
        "num_subset": "all",
        "num_workers": 0,
        "optimize_data": False,
        "seed": np.random.randint(0, 2**32 - 1),
        "num_epochs": 100,
        "batch_size": 128,
        "aux_inner_dim": 10,
        "num_chars": 0,
        "num_channels": 2,
        "max_tips": 1000,
        "mmd_loss_weight": 1.0,
        "mmd_num_kernels": 3,
        "phy_loss_weight": 0.9,
        "aux_loss_weight": 0.1,
        "char_loss_weight": 1.0,
        "latent_output_dim": None,
        "stride": [2, 4, 8],
        "kernel": [3, 5, 9],
        "out_channels": [16, 64, 128],
        "char_type": "categorical",
        "proportion_train" : 0.85,
        "learning_rate" : 1e-3,
        "weight_decay"  : 1e-3,
        "track_grad" : False,
        "device" : "auto",
        "which_aux" : "all",
        "checkpoints" : None,
        "resume_from_checkpoint": None,
        "pretrained_model": None,
    }

def _update_settings_from_command_line(settings, args):

    arg_map = {
        "testing"       : args.testing,
        "out_prefix"    : args.out_prefix,
        "num_subset"    : args.num_subset,
        "num_workers"   : args.num_workers,
        "optimize_data" : args.optimize_data,
        "seed"          : args.seed,
        "num_epochs"    : args.num_epochs,
        "batch_size"    : args.batch_size,
        "aux_inner_dim" : args.aux_inner_dim,
        "num_chars"     : args.num_chars,
        "num_channels"  : args.num_channels,
        "max_tips"      : args.max_tips,
        "mmd_loss_weight" : args.mmd_loss_weight,
        "mmd_num_kernels" : args.mmd_num_kernels,
        "phy_loss_weight": args.phy_loss_weight,
        "char_loss_weight" : args.char_loss_weight,
        "aux_loss_weight": args.aux_loss_weight,
        "latent_output_dim": args.latent_output_dim,
        "kernel"        : args.kernel,
        "stride"        : args.stride,
        "out_channels"  : args.out_channels,
        "char_type"     : args.char_type,
        "proportion_train": args.proportion_train,
        "learning_rate" : args.learning_rate,
        "weight_decay"  : args.weight_decay,
        "track_grad"    : args.track_grad,
        "device"        : args.device,
        "which_aux"     : args.which_aux,
        "checkpoints"   : args.checkpoints,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "pretrained_model": args.pretrained_model,

    }

    # override defaults and config with command line args
    for k, v in arg_map.items():
        v = _normalize_cli_value(v)
        if v is not None:
            if k in {"kernel", "stride", "out_channels", "checkpoints"} and isinstance(v, str):
                settings[k] = [int(x) for x in v.split(",")]
            elif k in {"latent_output_dim", "num_channels", "num_chars", "num_subset",
                       "num_epochs", "batch_size", "max_tips", "num_workers", "seed", "aux_inner_dim",
                       "mmd_num_kernels"}:
                settings[k] = int(v)
            elif k in {"mmd_loss_weight", "aux_loss_weight", "phy_loss_weight", "char_loss_weight"}:
                settings[k] = float(v)
            elif k in {"testing", "track_grad", "optimize_data"}:
                settings[k] = bool(v)
            elif k in {"which_aux"}:
                settings[k] = _normalize_which_aux(v)
            else:
                settings[k] = v

def _normalize_cli_value(value):
    if value == "None":
        return None
    if isinstance(value, str):
        value_lower = value.lower()
        if value_lower == "true":
            return True
        if value_lower == "false":
            return False
    return value

def _normalize_which_aux(value):
    if isinstance(value, str):
        return "all" if value.lower() == "all" else [x.strip() for x in value.split(",")]
    if isinstance(value, (list, tuple)) and len(value) == 1 and str(value[0]).lower() == "all":
        return "all"
    return value

def _update_settings_from_config(settings : dict, config : dict):
    for key in settings:
        if key in config:
            settings[key] = config[key]

def _save_settings(settings, out_file):
    df_index= [x for x in settings.keys()]
    df_val  = [str(x) for x in settings.values()]
    df = pd.DataFrame(df_val, index=df_index, columns=None)
    df.to_csv(out_file, sep="\t", header = False)
    print("Settings saved to", out_file)
