import os
import sys
import numpy as np

import torch
from torch.nn.parameter import Parameter
import nibabel as nib

from models.text2brain_model import Text2BrainModel

PHRASES = {
            "false-belief_tale_vs_mecha_tale": "False-belief tale vs mecha tale",
            "false-belief_tale_vs_mechanistic_tale": "False-belief tale vs mechanistic tale",
          }


if __name__ == "__main__":

    query = sys.argv[1]
    output_file = sys.argv[2]

    checkpoint_file = "checkpoints/fc64_d128_relu_lr0.03_decay1e-06_drop0.55_seed28_checkpoint.pth"
    pretrained_bert_dir = "scibert_scivocab_uncased"

    """Init Model"""
    model = Text2BrainModel(
        out_channels=1,
        fc_channels=64,
        decoder_filters=32,
        pretrained_bert_dir=pretrained_bert_dir,
        drop_p=0.55)

    state_dict = torch.load(checkpoint_file, map_location=torch.device('cpu'))['state_dict']
    model.load_state_dict(state_dict)
    
    model.eval()

    """ Output brain image """
    vol_data = np.zeros((46, 55, 46))
    affine = np.array([[   4.,    0.,    0.,  -90.],
       [   0.,    4.,    0., -126.],
       [   0.,    0.,    4.,  -72.],
       [   0.,    0.,    0.,    1.]])

    with torch.no_grad():
        text = (query.replace("/", ""), )

        pred = model(text).numpy().squeeze(0).squeeze(0)

        vol_data[3:-3, 3:-4, :-6] = pred
        
        pred_img = nib.Nifti1Image(vol_data, affine)
        nib.save(pred_img, output_file)
