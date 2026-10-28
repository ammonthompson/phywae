#!/usr/bin/env python3
# test a pretrained model
from phywae.PhyloAEModel import AECNN
from phywae import utils
import torch
import h5py
import pandas as pd
import os
import numpy as np
import os
from phywae.utils import get_num_tips
from phywae.cli._output import print_output_files


def add_arguments(cmd):
    cmd.add_argument("-m", "--model", required=True, help="Path to a trained model artifact or trainer checkpoint")
    cmd.add_argument("-t", "--tree-data", required=True, help="Path to the phyddle formated tree CBLV(S). If using a phyddle -s F output hdf5, " \
    "use the key 'phy_data' for the tree data and 'aux_data' for the auxiliary data. If using csv, for the cblv tree data, then" \
        " use the -s [--aux-data] flag to specify the auxiliary data file.")
    cmd.add_argument("-s", "--aux-data", required=False, help="Path to the auxiliary data if using csv format. " )
    cmd.add_argument("-nc", "--num-channels", required=False, type=int, help="Number of channels in the data. Default is 2.")
    cmd.add_argument("-mt", "--max-tips", required=False, type=int,help="Max number of tips in the data. Default is 500.")
    cmd.add_argument("-o", "--out-prefix", required=False, help="Path to out file prefix")


def main(args):
    ae_model_fn       = utils.file_exists(args.model)
    tree_data_fn      = utils.file_exists(args.tree_data)
    aux_data_fn       = utils.file_exists(args.aux_data) if args.aux_data is not None else None
    num_channels      = args.num_channels if args.num_channels is not None else None
    max_tips          = args.max_tips if args.max_tips is not None else None

    if (max_tips is not None and num_channels is None) or \
        (num_channels is not None and max_tips is None):
        raise ValueError("If using num_tips, then num_channels must also be specified. " \
        "If using num_channels, then num_tips must also be specified.")

    
    if args.out_prefix is None:
        out_file_prefix = tree_data_fn.split('/')[-1].split('.')[0]
    else:
        out_file_prefix = args.out_prefix
    output_files = []

    # load the trained model and normalizers
    # ae_model = torch.load(ae_model_fn, weights_only=False)
    ae_model = AECNN.load_pretrained_from_file(ae_model_fn, map_location="cpu")
    model_aux_names = ae_model.aux_data_names
    if model_aux_names is None:
        raise ValueError("Loaded model is missing 'aux_data_names'.")
    full_aux_colnames = None

    # import test data
    # test if file type is hdf5
    if tree_data_fn.endswith('.hdf5') or tree_data_fn.endswith('.h5py'):
        with h5py.File(tree_data_fn, "r") as f:
            test_phy_data = f['phy_data'][...]
            test_aux_data = f['aux_data'][...]
            if "aux_data_names" in f.keys():
                raw_aux_names = f["aux_data_names"][...]
                full_aux_colnames = np.array(
                    [
                        x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
                        for x in raw_aux_names[0]
                    ],
                    dtype=str,
                )

            # TODO: Probably not necessary
            if len(test_aux_data.shape) == 1:
                test_aux_data = test_aux_data.reshape((test_aux_data.shape[0], 1))
            
            if 'labels' in f.keys() and 'label_names' in f.keys():
                # if labels are present, save them to a csv file
                # check if labels are in the correct format
                labels = f['labels'][...]
                label_names = [lbl.decode('utf-8') if isinstance(lbl, bytes) 
                               else lbl for lbl in f['label_names'][...][0]]
                df = pd.DataFrame(labels, columns=label_names)
                labels_out_file = out_file_prefix + ".labels.csv"
                df.to_csv(labels_out_file, index = False)
                output_files.append(labels_out_file)
            else:
                print("No labels found in the hdf5 file. Continuing without making labels file.")

    elif tree_data_fn.endswith('.csv') or tree_data_fn.endswith(".cblv"):
        if aux_data_fn is None:
            raise ValueError("When providing tree data as csv/cblv, you must also provide --aux-data.")
        test_phy_data = pd.read_csv(tree_data_fn, header = None, 
                                    index_col = None).to_numpy(dtype=np.float32)
        test_aux_data = pd.read_csv(aux_data_fn, header = None, 
                                    index_col = None).to_numpy(dtype=np.float32)
    else:
        raise ValueError("Input file must be in hdf5 or csv format.")
    
    if num_channels is not None:
        nc = test_phy_data.shape[1] // max_tips
        if test_phy_data.shape[1] % max_tips != 0:
            raise ValueError("Number of tips does not match the number of tips in the trained model.")  
        # reshape, and then reduce the tensor to the number of channels desired (might be fewer than the original)
        test_phy_data = test_phy_data.reshape(-1, max_tips, nc)[:, :, :num_channels].reshape((test_phy_data.shape[0], -1))
        test_phy_data = test_phy_data.reshape((test_phy_data.shape[0], ae_model.num_structured_input_channel, 
                            int(test_phy_data.shape[1]/ae_model.num_structured_input_channel)), order = "F")

    # Align aux columns by name when available. Data may contain extra aux columns beyond the model's.
    # print(test_aux_data)
    if full_aux_colnames is not None:
        _, test_aux_data = utils.get_aux_data(full_aux_colnames, test_aux_data, model_aux_names)
    else:
        expected_aux_cols = int(model_aux_names.shape[0])
        if test_aux_data.ndim != 2 or test_aux_data.shape[1] != expected_aux_cols:
            raise ValueError(
                f"Aux data has shape {test_aux_data.shape} but model expects {expected_aux_cols} columns "
                f"({list(model_aux_names)}). Provide phyddle -s F data with aux_data_names for alignment, "
                "or ensure aux columns already match the model exactly."
            )

    ae_model.validate_num_tips(test_aux_data)

    # print(ae_model.aux_data_names)
    # print(test_aux_data)


    # make encoded tree file
    latent_dat = ae_model.norm_and_encode(test_phy_data, test_aux_data)
    latent_dat_df = pd.DataFrame(latent_dat, columns = None, index = None)
    encoded_out_file = out_file_prefix + ".ae_encoded.csv"
    latent_dat_df.to_csv(encoded_out_file, header = None, index = None)
    output_files.append(encoded_out_file)

    print_output_files(output_files)
