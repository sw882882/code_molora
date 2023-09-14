import math

import nltk
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from sentence_transformers import InputExample, SentenceTransformer, losses, models
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from sentence_transformers.losses import DenoisingAutoEncoderLoss
from torch.utils.data import DataLoader

nltk.download("punkt")

# load data
df = pd.read_json("./evol-codealpaca-v1/train.jsonl", lines=True)

dataset = DatasetDict(
    {
        "train": Dataset.from_pandas(df),
    }
)

print(dataset)

model = models.Transformer("distilroberta-base")

train_data = DenoisingAutoEncoderDataset(df["instruction"].tolist())
loader = DataLoader(train_data, batch_size=128, shuffle=True, drop_last=True)

pooling = models.Pooling(model.get_word_embedding_dimension(), "cls")
model = SentenceTransformer(modules=[model, pooling])

loss = DenoisingAutoEncoderLoss(model, tie_encoder_decoder=True)
# round up to ensure all examples are seen
num_epoch = math.ceil(len(df) / 64 * 2)

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

model.push_to_hub("bert_code_instruct")
