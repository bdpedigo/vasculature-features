# %%
import pandas as pd

root_level2_mapping = pd.read_csv(
    "/Users/ben.pedigo/code/cave-wrangler/vasculature_features/root_level2_mapping.csv"
)

# %%
root_level2_mapping["level2_id"].is_unique

# %%
root_level2_mapping[
    root_level2_mapping["level2_id"].duplicated(keep=False)
].sort_values("level2_id")

# %%
from caveclient import CAVEclient

client = CAVEclient("minnie65_public")

# %%
from caveclient import CAVEclient

client = CAVEclient("minnie65_public")

blue = 864691136973664540
print("Blue start:", client.chunkedgraph.get_root_timestamps(blue, latest=False)[0])
print("Blue end:", client.chunkedgraph.get_root_timestamps(blue, latest=True)[0])
print()
yellow = 864691134917441034
print("Yellow start:", client.chunkedgraph.get_root_timestamps(yellow, latest=False)[0])
print("Yellow end:", client.chunkedgraph.get_root_timestamps(yellow, latest=True)[0])

# %%
blue_leaves = client.chunkedgraph.get_leaves(blue, stop_layer=2)
yellow_leaves = client.chunkedgraph.get_leaves(yellow, stop_layer=2)
import numpy as np

# %%
np.setdiff1d(blue_leaves, yellow_leaves)

# %%
dup_mapping = root_level2_mapping[
    root_level2_mapping["level2_id"].duplicated(keep=False)
]
roots = dup_mapping.groupby('root_id').size()

#%%
dup_mapping.groupby('level2_id')['root_id'].max()