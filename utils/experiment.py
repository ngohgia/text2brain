import sys
import torch
import numpy as np

from .utilities import compute_corr_coeff


def train(model, train_loader, optimizer, loss_fn, mask):
    model.train()
    avg_loss = 0
    avg_corr = 0

    for batch_idx, (text, target) in enumerate(train_loader):
        target = target.cuda()
        optimizer.zero_grad()
        output = model(text)

        loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()
        avg_loss += (loss.item() / len(train_loader))

        output_np = output.cpu().detach().numpy()[:, :, mask]
        target_np = target.cpu().detach().numpy()[:, :, mask]
        all_corr = compute_corr_coeff(
            output_np.reshape(output_np.shape[0], -1),
            target_np.reshape(output_np.shape[0], -1))
        corr = np.mean(np.diag(all_corr))
        if np.isnan(corr):
            print(text,
                  np.isnan(output_np).any(),
                  np.isnan(target_np).any(), all_corr, corr)
            print("Output", torch.max(output), output)
            print("Target", torch.max(target), target)
            sys.exit(1)

        avg_corr = avg_corr + corr / len(train_loader)

        if batch_idx % 100 == 99:
            print('[{}/{} ({:.0f}%)] Loss: {:.6f} Corr: {:.6f}'.format(
                batch_idx * len(text), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), corr))

    print('  Train: avg loss: {:.6f} - avg corr: {:.6f}'.format(avg_loss, avg_corr))

    return avg_loss, avg_corr


def eval(model, val_loader, loss_fn, mask):
    model.eval()
    avg_loss = 0
    avg_corr = 0

    with torch.no_grad():
        for batch_idx, (text, target) in enumerate(val_loader):
            target = target.cuda()
            output = model(text)

            loss = loss_fn(output, target)

            avg_loss += (loss.item() / len(val_loader))

            output_np = output.cpu().detach().numpy()[:, :, mask]
            target_np = target.cpu().detach().numpy()[:, :, mask]
            corr = np.mean(
                np.diag(
                    compute_corr_coeff(
                        output_np.reshape(output_np.shape[0], -1),
                        target_np.reshape(output_np.shape[0], -1))))
            avg_corr = avg_corr + corr / len(val_loader)

    print('  Val: avg loss: {:.6f} - avg corr: {:.6f}'.format(
        avg_loss, avg_corr))

    return avg_loss, avg_corr
