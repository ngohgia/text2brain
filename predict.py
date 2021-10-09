import sys
import numpy as np

import torch
import nibabel as nib

from models.text2brain_model import Text2BrainModel


if __name__ == "__main__":

    query = sys.argv[1]
    output_file = sys.argv[2]
    fc_channels = 64
    decoder_filters = 32

    checkpoint_file = f"best_loss.pth"
    pretrained_bert_dir = "scibert_scivocab_uncased"

    # Init Model
    model = Text2BrainModel(
        out_channels=1,
        fc_channels=fc_channels,
        decoder_filters=decoder_filters,
        pretrained_bert_dir=pretrained_bert_dir,
        drop_p=0.55)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(checkpoint_file, map_location=device)['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # Output brain image
    vol_data = np.zeros((46, 55, 46))
    affine = np.array([[   4.,    0.,    0.,  -90.],
       [   0.,    4.,    0., -126.],
       [   0.,    0.,    4.,  -72.],
       [   0.,    0.,    0.,    1.]])

    text = (query.replace("/", ""), )
    with torch.no_grad():
        pred = model(text).cpu().numpy().squeeze(axis=(0, 1))

    vol_data[3:-3, 3:-4, :-6] = pred

    pred_img = nib.Nifti1Image(vol_data, affine)
    nib.save(pred_img, output_file)
