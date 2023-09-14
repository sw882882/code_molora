import os

import faiss
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
# change this to your embedding model
# refer to huggingface embedding leaderboard
model = SentenceTransformer("bigcode/starencoder")

# change this to your dataset
df = pd.read_json("./code_instructions_120k/code_instructions_120k.json")

df_temp = pd.read_json("./evol-codealpaca-v1/train.jsonl", lines=True)

print(df)
print(df_temp)
# merge
df = pd.merge(df, df_temp, on=["instruction", "output"], how="outer")

print(df)

text = df["instruction"].tolist()

embeddings = model.encode(text)
d = embeddings.shape[1]  # type: ignore
index_embed = faiss.IndexFlatL2(d)
index_embed.add(embeddings)
print("finished writing index")

# may want to mess with this
ncentroids = 8
niter = 40
verbose = True
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
kmeans.train(embeddings)
print("finished training")

D, I = kmeans.index.search(embeddings, 1)

df["closest_centroid"] = I.flatten()

print(df)

centers = torch.from_numpy(kmeans.centroids)

torch.save(centers, "centers.pt")
# save df without index
# this will add another row with the cluster id, from there you
# might want to split or do some other processing.
df.to_parquet("./kmeaned.parquet", index=False)

# drop text column
df.drop("text", axis=1, inplace=True)

for i in df["closest_centroid"].unique():
    # make folder
    os.mkdir(f"./cluster_{i}")
    df[df["closest_centroid"] == i].drop("closest_centroid", axis=1).to_csv(
        f"./cluster_{i}/data.csv", index=False
    )
