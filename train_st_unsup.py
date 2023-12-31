import math

import pandas as pd
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses
from torch.utils.data import DataLoader


nltk.download("punkt")

# load data
df = pd.read_json("./evol-codealpaca-v1/train.jsonl", lines=True)
# Define your sentence transformer model using CLS pooling
model_name = "distilroberta-base"
word_embedding_model = models.Transformer(model_name)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(), "cls"
)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Define a list with sentences (1k - 100k sentences)
train_sentences = df["instruction"].tolist()

# Create the special denoising dataset that adds noise on-the-fly
train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)

# DataLoader to batch your data
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Use the denoising auto-encoder loss
train_loss = losses.DenoisingAutoEncoderLoss(
    model, decoder_name_or_path=model_name, tie_encoder_decoder=True
)

# Call the fit method
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    weight_decay=0,
    scheduler="constantlr",
    optimizer_params={"lr": 3e-5},
    show_progress_bar=True,
)

model.save("output/tsdae-model")
