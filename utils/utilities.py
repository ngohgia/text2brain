import os
import numpy as np
import torch

def compute_corr_coeff(A,B):
    # Rowwise mean of input arrays & subtract from input arrays
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    # corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))

def save_checkpoint(model, optimizer, scheduler, epoch, fname, output_dir):
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'scheduler': scheduler.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, os.path.join(output_dir, fname))

