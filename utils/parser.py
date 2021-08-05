from argparse import ArgumentParser

def init_args():
    
    parser = ArgumentParser()

    parser.add_argument("--gpus", type=str,
                        default="",
                        help="Which gpus to use?")
    
    parser.add_argument("--ver",
                        type=str,
                        help="Additional string for the name of the file")
    
    parser.add_argument("--train_csv",
                        type=str,
                        help="Path to the csv containing the training articles data")
    
    parser.add_argument("--val_csv",
                        type=str,
                        help="Path to the csv containing the validation articles data")

    parser.add_argument("--images_dir",
                        type=str,
                        help="Directory containing activation maps, should be of the size (40, 48, 40)")

    parser.add_argument("--pretrained_bert_dir",
                        type=str,
                        help="Directory containing pretrained BERT model")

    parser.add_argument("--pretrained_tokenizer_dir",
                        type=str,
                        default=None,
                        help="Directory containing pretrained tokenizer")

    parser.add_argument("--mask_file",
                        type=str,
                        help="Brain mask file")

    parser.add_argument("--save_dir",
                        type=str,
                        help="Path to the output directory")

    parser.add_argument("--n_fc_channels",
                        type=int,
                        default=1024,
                        help="Base number of channels in the FC layer, default=1024")

    parser.add_argument("--n_decoder_channels",
                        type=int,
                        default=256,
                        help="Base number of channels in the image decoder, default=256")

    parser.add_argument("--n_output_channels",
                        type=int,
                        default=1,
                        help="Number of output channels, default=1")

    parser.add_argument("--lr",
                        type=float,
                        default=1e-2,
                        help="Learning rate, default: 1e-2")

    parser.add_argument("--weight_decay",
                        type=float,
                        default=1e-6,
                        help="Weight decay of the optimizer")

    parser.add_argument("--drop_p",
                        type=float,
                        default=0.6,
                        help="Dropout proportion for FC layer")

    parser.add_argument("--epochs",
                        type=int,
                        default=1000,
                        help="Training epochs, default: 1000")
    
    parser.add_argument("--seed",
                        type=int,
                        default=28,
                        help="Random seed for numpy to create train/val split, default = 28")
    

    parser.add_argument("--checkpoint_file",
                        type=str,
                        help="Path to the checkpoint file to be loaded into the model")

    parser.add_argument("--checkpoint_interval",
                        type=int,
                        default=10,
                        help="Number of epochs between saved checkpoints, default = 50")

    parser.add_argument("--batch_size",
                        type=int,
                        default=16,
                        help="Batch size")

    parser.add_argument("--debug",
                        type=bool,
                        default=False,
                        help="Is debugging")

    parser.add_argument("--filter_keywords",
                        type=bool,
                        default=False,
                        help="If True, ignore articles without keywords")

    parser.add_argument("--phrase",
                        type=str,
                        default=None,
                        help="Input phrase for prediction")

    parser.add_argument("--source",
                        type=str,
                        default="title",
                        help="Source type")

    parser.add_argument("--model",
                        type=str,
                        default="Text2BrainModel",
                        help="Model type")

    return parser.parse_args()
