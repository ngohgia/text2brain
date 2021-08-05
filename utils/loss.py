import torch.nn as nn

def mse_with_mask_loss(output, target, mask):
    # mask = torch.from_numpy(mask.astype(np.uint8)).type(torch.bool).cuda()
    # print(output.shape)
    # print(target.shape)
    # print(mask.shape)
    return ((output[:, :, mask] - target[:, : , mask])**2).mean()
    return ((output[:, :, mask] - target[:, : , mask])**2).mean()
