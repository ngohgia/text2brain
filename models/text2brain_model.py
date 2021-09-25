import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from .decoder import ImageDecoder
import transformers


class Text2BrainModel(nn.Module):
    def __init__(self, out_channels, fc_channels, decoder_filters, pretrained_bert_dir, decoder_act_fn=nn.Sigmoid, drop_p=0.5, decoder_input_shape=[4, 5, 4]):
        super().__init__()
        self.out_channels = out_channels
        self.fc_channels = fc_channels
        self.decoder_filters = decoder_filters
        self.decoder_input_shape = decoder_input_shape
        self.drop_p = drop_p

        self.tokenizer = transformers.BertTokenizer.from_pretrained(pretrained_bert_dir)
        self.encoder = transformers.BertModel.from_pretrained(pretrained_bert_dir)
        if torch.cuda.is_available():
          self.encoder = self.encoder.cuda()

        self.fc = nn.Linear(
          in_features=768,
          out_features=self.decoder_input_shape[0]*self.decoder_input_shape[1]*self.decoder_input_shape[2]*self.fc_channels)
        self.dropout = nn.Dropout(self.drop_p)
        self.relu = nn.ReLU()

        self.decoder = ImageDecoder(in_channels=self.fc_channels, out_channels=1, num_filter=self.decoder_filters, act_fn=decoder_act_fn)


    def forward(self, texts):
        batch = [self._tokenize(x) for x in texts]

        in_mask = self._pad_mask(batch, batch_first=True)
        in_ = pad_sequence(batch, batch_first=True)
        if torch.cuda.is_available():
          in_ = in_.cuda()
          in_mask = in_mask.cuda()

        _, embedding = self.encoder(in_, attention_mask=in_mask)

        x = self.dropout(embedding)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.relu(x)

        decoder_tensor_shape = [-1, self.fc_channels] + self.decoder_input_shape
        x = x.view(decoder_tensor_shape)

        out = self.decoder(x)

        return out

    def _tokenize(self, text):
        return self.tokenizer.encode(text, add_special_tokens=True, return_tensors='pt', truncation=True, max_length=512).squeeze(0)

    def _pad_mask(self, sequences, batch_first=False):
        ret = [torch.ones(len(s)) for s in sequences]
        return pad_sequence(ret, batch_first=batch_first)


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import os

    data_dir = "/nfs03/users/ghn8/text2brain/data/processed"
    raw_df = pd.read_csv(os.path.join(data_dir, "tables", "210108_cleaned_abs_discussion", "train_articles_210109.csv"))
    df = raw_df.loc[raw_df["article-pmid"].isin(["24937006"])]

    # x = df["text"].values.tolist()
    x = ('our findings are in agreement with other studies with stroke patients , and extend our previous research with resting state providing structural ground to the difficulty of sp patients to compensate their cognitive deficits after stroke',)
    model = Text2BrainModel(1, 64, 32, "../scibert_scivocab_uncased")
    if torch.cuda.is_available():
        model = model.cuda()
    out = model(x)
    out = out.cpu().detach().numpy()
    print(np.isnan(out).any())
    print(out.shape)
