import math
import re

import nltk
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from sentence_transformers import InputExample, SentenceTransformer, losses, models
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from sentence_transformers.losses import DenoisingAutoEncoderLoss
from torch.utils.data import DataLoader

nltk.download("punkt")

# load data
df = pd.read_json("./code_instructions_120k/code_instructions_120k.json")

# doing manually to prevent data leakage and make sure test data is only reviewty data
# test_set_size = round((len(df) + len(df_shopee)) * 0.1)
test_set_size = round(len(df) * 0.1)
test_set_frac = test_set_size / len(df)
test_df = df.sample(frac=test_set_frac, random_state=42)
train_df = df.drop(test_df.index)


# train_df = pd.concat([train_df, df_shopee])
train_df, test_df = train_df.dropna(), test_df.dropna()
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
# clean df
for index, row in df.iterrows():
    if "[" in row["query"] and "]" in row["query"]:
        df.at[index, "query"] = re.sub("[\(\[].*?[\)\]]", "", row["query"])

dataset = DatasetDict(
    {
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(test_df),
    }
)


print(dataset)


# model = models.Transformer("distilbert-base-multilingual-cased")
model = models.Transformer("bert-base-multilingual-cased")

train_data = DenoisingAutoEncoderDataset(train_df["result"].tolist())
loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)

pooling = models.Pooling(model.get_word_embedding_dimension(), "cls")
model = SentenceTransformer(modules=[model, pooling])

loss = DenoisingAutoEncoderLoss(model, tie_encoder_decoder=True)
num_epoch = math.ceil(
    len(train_df) / 64 * 2
)  # round up to ensure all examples are seen

model.fit(
    train_objectives=[(loader, loss)],
    epochs=num_epoch,
    weight_decay=0,
    scheduler="constantlr",
    optimizer_params={"lr": 3e-5},
    show_progress_bar=True,
)
# save model
model.save("./unsupervised_fine_tune")
