#!/usr/bin/env python3
# test a pretrained model
import phywae.utils as utils
from phywae.PhyloAEModel import AECNN
import torch
# import h5py
import numpy as np
import pandas as pd
# import os
from phywae.cli._output import print_output_files

# takes in an encoded tree (encoded by the same trained model as the decoder)
# outputs a cblv file

# TODO: Output a nwk string too?

def add_arguments(cmd):
    cmd.add_argument("-m", "--model", required=True, help="Path to a trained model artifact or trainer checkpoint")
    cmd.add_argument("-e", "--encoded-data", required=True, help="Path to the model encoded data file")
    cmd.add_argument("-o", "--out-prefix", required=False, help="Path to out file prefix")


def main(args):
    ae_model_fn       = utils.file_exists(args.model)
    encoded_data_fn   = utils.file_exists(args.encoded_data)
    
    if args.out_prefix is None:
        out_file_prefix = encoded_data_fn.split('/')[-1].split('.')[0]
    else:
        out_file_prefix = args.out_prefix

    # load the trained model and normalizers
    # ae_model = torch.load(ae_model_fn, weights_only=False)
    ae_model = AECNN.load_pretrained_from_file(ae_model_fn, map_location="cpu")

    # import test data
    encoded_data = pd.read_csv(encoded_data_fn, header = None, index_col = None).to_numpy(dtype=np.float32)

    test_phy_data, test_aux_data = ae_model.decode_and_denorm(encoded_data)

    test_phy_data = utils.set_pred_pad_to_zero(test_phy_data,  test_aux_data[:, ae_model.aux_numtips_idx])    
    test_phy_data = test_phy_data.reshape((test_phy_data.shape[0], -1), order = "F")

 
    cblv_df = pd.DataFrame(test_phy_data)
    aux_df  = pd.DataFrame(test_aux_data)
    cblv_out_file = out_file_prefix + ".ae_decoded.cblv.csv"
    aux_out_file = out_file_prefix + ".ae_decoded.aux.csv"
    cblv_df.to_csv(cblv_out_file, header = None, index = None)
    aux_df.to_csv(aux_out_file, header = None, index = None)

    print_output_files([cblv_out_file, aux_out_file])
