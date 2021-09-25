def mse_with_mask_loss(output, target, mask):
    return ((output[:, :, mask] - target[:, : , mask])**2).mean()
