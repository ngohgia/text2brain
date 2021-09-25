import os
import numpy as np
import torch
from torch.utils.data import Dataset


class Text2BrainDataset(Dataset):
    def __init__(self, articles_df, images_dir, source):
        self.articles_df = articles_df
        self.images_dir = images_dir
        self.source = source

    def __getitem__(self, index):
        row = self.articles_df.iloc[index]
        pmid = row["article-pmid"]

        if self.source == "abstract":
            text = row.abstract
        elif self.source == "title":
            text = row["article-title"]
        else:
            raise Exception("Data source not implemented")

        image_file = os.path.join(self.images_dir, "pmid_%s.npy" % pmid)
        image = np.expand_dims(np.load(image_file), 0)
        image = image / np.max(image)
        image = np.nan_to_num(image, copy=False)

        text = text.lower()

        return text, torch.cuda.FloatTensor(image)

    def __len__(self):
        return len(self.articles_df.index)


if __name__ == "__main__":
    import pandas as pd
    data_dir = "/nfs03/users/ghn8/text2brain/data/processed"
    df = pd.read_csv(os.path.join(data_dir, "valid.csv"))
    dataset = Text2BrainDataset(df, os.path.join(data_dir, "images"), "title")

    for i, (text, img) in enumerate(dataset):
        print(i, text, img.shape)
        if i == 3:
            break
