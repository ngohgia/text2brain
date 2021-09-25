import os
import sys
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import transformers as tfm

from models.text2brain_model import Text2BrainModel
from utils.parser import init_args
from utils.dataset import Text2BrainDataset
from utils.utilities import save_checkpoint
from utils import experiment


if __name__ == "__main__":
    args = init_args()

    # Init
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    output_name = f"{args.ver}_{args.source}_fc{args.n_fc_channels}_dec{args.n_decoder_channels}_lr{args.lr}_decay{args.weight_decay}_drop{args.drop_p}_seed{args.seed}"
    output_dir = os.path.join(args.save_dir, output_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        raise Exception(f"Output dir exists: {output_dir}")
    mask = np.load(args.mask_file)
    writer = SummaryWriter(os.path.join(output_dir, "logs"))

    # Load Data
    np.random.seed(args.seed)
    train_df = pd.read_csv(args.train_csv, dtype={'article-pmid': str}).dropna(subset=['abstract'])
    val_df = pd.read_csv(args.val_csv, dtype={'article-pmid': str}).dropna(subset=['abstract'])

    train_dataset = Text2BrainDataset(train_df, args.images_dir, args.source)
    val_dataset = Text2BrainDataset(val_df, args.images_dir, args.source)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)
    print("Number of training articles:", len(train_dataset))
    print("Number of validation articles:", len(val_dataset))

    # Init model
    model = Text2BrainModel(
        out_channels=1,
        fc_channels=args.n_fc_channels,
        decoder_filters=args.n_decoder_channels,
        pretrained_bert_dir=args.pretrained_bert_dir,
        drop_p=args.drop_p)
    model.cuda()

    # Loading checkpoint
    if args.checkpoint_file is not None:
        state_dict = torch.load(args.checkpoint_file)['state_dict']
        model.load_state_dict(state_dict)

    # Optimizer
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = num_training_steps // 3
    opt = tfm.AdamW([
            {'params': model.fc.parameters()},
            {'params': model.decoder.parameters()},
            {'params': model.encoder.parameters(), 'lr': 1e-5},
        ], lr=args.lr, weight_decay=args.weight_decay)
    sched = tfm.get_linear_schedule_with_warmup(opt, num_warmup_steps, num_training_steps)

    loss_fn = nn.MSELoss(reduction="sum")

    val_losses = []
    val_corrs = []
    best_loss = sys.float_info.max
    best_corr = 0.0

    for epoch in range(args.epochs):
        print("Epoch", epoch+1)
        train_loss, train_corr = experiment.train(model, train_loader, opt, loss_fn, mask)
        val_loss, val_corr     = experiment.eval(model, val_loader, loss_fn, mask)

        writer.add_scalar('training loss', train_loss, epoch)
        writer.add_scalar('training corr', train_corr, epoch)

        writer.add_scalar('validation loss', val_loss, epoch)
        writer.add_scalar('validation corr', val_corr, epoch)

        val_losses.append(val_loss)
        val_corrs.append(val_corr)

        mean_loss = np.mean(val_losses[-args.checkpoint_interval:])
        mean_corr = np.mean(val_corrs[-args.checkpoint_interval:])

        if (epoch > args.epochs * 0.1) and (epoch % args.checkpoint_interval == 0):
            if mean_loss < best_loss:
                save_checkpoint(model, opt, sched, epoch, "best_loss.pth", output_dir)
                best_loss = mean_loss
            if mean_corr > best_corr:
                save_checkpoint(model, opt, sched, epoch, "best_corr.pth", output_dir)
                best_corr = mean_corr
            save_checkpoint(model, opt, sched, epoch, f'checkpoint_{epoch}.pth', output_dir)
        sched.step()
    save_checkpoint(model, opt, sched, args.epochs, f'checkpoint_{args.epochs}.pth', output_dir)
    writer.close()
