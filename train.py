import os
import sys
import subprocess
import pandas as pd

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.text2brain_model import *

from utils.loss import mse_with_mask_loss
from utils.parser import init_args
from utils.dataset import Text2BrainDataset
from utils.utilities import save_checkpoint
from utils import experiment

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

if __name__ == "__main__":

    args = init_args()

    """Init"""
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpus

    output_name = "%s_%s_%s_fc%d_dec%d_lr%s_decay%s_drop%s_seed%d" % (args.ver, args.source, args.model, args.n_fc_channels, args.n_decoder_channels, str(args.lr), str(args.weight_decay), str(args.drop_p), args.seed)
    if args.debug:
        output_name = "debug_" + output_name
    output_dir = os.path.join(args.save_dir, output_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        if not args.debug:
            raise Exception("Output dir exists: %s" % output_dir)
    mask = np.load(args.mask_file)
    writer = SummaryWriter(os.path.join(output_dir, "logs"))

    """Load Data"""
    np.random.seed(args.seed)
    val_df = pd.read_csv(args.train_csv, nrows=2, dtype={ "article-pmid": str }) if args.debug else pd.read_csv(args.val_csv, dtype={ "article-pmid": str })
    train_df = pd.read_csv(args.train_csv, nrows=2, dtype={ "article-pmid": str }) if args.debug else pd.read_csv(args.train_csv, dtype={ "article-pmid": str })
    val_df.dropna(subset=['abstract'], inplace=True)
    # val_df.dropna(subset=['article-title'], inplace=True)
    train_df.dropna(subset=['abstract'], inplace=True)
    # train_df.dropna(subset=['article-title'], inplace=True)

    if args.filter_keywords:
        val_df.dropna(subset=['keywords'], inplace=True)
        train_df.dropna(subset=['keywords'], inplace=True)

    train_dataset = Text2BrainDataset(train_df, args.images_dir, args.source)
    val_dataset = Text2BrainDataset(val_df, args.images_dir, args.source)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)
    print("Number of training articles:", len(train_dataset))
    print("Number of validation articles:", len(val_dataset))

    """Init model"""
    if args.model == "Text2BrainModel":
        model = Text2BrainModel(
            out_channels=1,
            fc_channels=args.n_fc_channels,
            decoder_filters=args.n_decoder_channels,
            pretrained_bert_dir=args.pretrained_bert_dir,
            drop_p=args.drop_p)
    elif args.model == "Text2BrainModelAltEnc":
        model = Text2BrainModelAltEnc(
            out_channels=1,
            fc_channels=args.n_fc_channels,
            decoder_filters=args.n_decoder_channels,
            pretrained_bert_dir=args.pretrained_bert_dir,
            drop_p=args.drop_p)
    elif args.model == "Text2BrainModelAugmentedTokenizer":
        model = Text2BrainModelAugmentedTokenizer(
            out_channels=1,
            fc_channels=args.n_fc_channels,
            decoder_filters=args.n_decoder_channels,
            pretrained_bert_dir=args.pretrained_bert_dir,
            pretrained_tokenizer_dir=args.pretrained_tokenizer_dir,
            drop_p=args.drop_p)
    else:
        raise Exception("Model not implemented")
    model.cuda()

    """Loading checkpoint"""
    if args.checkpoint_file is not None:
        state_dict = torch.load(args.checkpoint_file)['state_dict']
        model.load_state_dict(state_dict)

    """Optimizer"""
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    n_training_steps = len(train_loader) * args.epochs
    n_warmup_steps = n_training_steps // 3
    optimizer = AdamW([
            {'params': model.fc.parameters()},
            {'params': model.decoder.parameters()},
            {'params': model.encoder.parameters(), 'lr': 1e-5},
        ], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = n_warmup_steps, num_training_steps = n_training_steps)

    loss_fn = nn.MSELoss(reduction="sum")

    """Cooking"""
    val_losses = []
    val_corrs = []
    best_loss = sys.float_info.max
    best_corr = 0.0

    epochs = 1000 if args.debug else args.epochs
    for epoch in range(epochs):
        print("Epoch", epoch+1)
        train_loss, train_corr = experiment.train(model, train_loader, optimizer, loss_fn, mask)
        val_loss, val_corr     = experiment.eval(model, val_loader, loss_fn, mask)

        if epoch == 0:
            subprocess.run("nvidia-smi", shell=True)

        writer.add_scalar('training loss', train_loss, epoch)
        writer.add_scalar('training corr', train_corr, epoch)

        writer.add_scalar('validation loss', val_loss, epoch)
        writer.add_scalar('validation corr', val_corr, epoch)

        val_losses.append(val_loss)
        val_corrs.append(val_corr)

        mean_loss = np.mean(val_losses[-args.checkpoint_interval:])
        mean_corr = np.mean(val_corrs[-args.checkpoint_interval:])

        if (epoch > epochs * 0.1) and (epoch % args.checkpoint_interval == 0):
            if mean_loss < best_loss:
                save_checkpoint(model, optimizer, scheduler, epoch, "best_loss.pth", output_dir)
                best_loss = mean_loss
            if mean_corr > best_corr:
                save_checkpoint(model, optimizer, scheduler, epoch, "best_corr.pth", output_dir)
                best_corr = mean_corr
            save_checkpoint(model, optimizer, scheduler, epoch, "checkpoint_%d.pth" % epoch, output_dir)
        scheduler.step()
    save_checkpoint(model, optimizer, scheduler, args.epochs, "checkpoint_%d.pth" % args.epochs, output_dir)
    writer.close()
