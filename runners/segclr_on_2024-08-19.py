# %%

import os
import time
from functools import partial
from io import BytesIO

import numpy as np
import pandas as pd
import seaborn as sns
from caveclient import CAVEclient
from cloudfiles import CloudFiles
from nglui import statebuilder
from nglui.segmentprops import SegmentProperties
from sklearn.neighbors import NearestNeighbors
from taskqueue import TaskQueue

from minniemorpho.models import load_model
from minniemorpho.query import Level2Query, SegCLRQuery


def write_dataframe(df, cf, path):
    with BytesIO() as f:
        df.to_csv(f, index=True)
        cf.put(path, f)


def load_dataframe(cf, path, **kwargs):
    bytes_out = cf.get(path)
    with BytesIO(bytes_out) as f:
        df = pd.read_csv(f, **kwargs)
    return df


def map_to_closest(source_X, target_X):
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(target_X.values)
    distances, indices = nn.kneighbors(
        source_X.values,
    )
    distances = distances.flatten()
    indices = indices.flatten()

    mapping_df = pd.DataFrame(index=source_X.index)
    mapping_df["target_id"] = target_X.index[indices]
    mapping_df["distance_to_target"] = distances
    return mapping_df


def generate_link(level2_features, box_info):
    client = CAVEclient("minnie65_public")
    client.materialize.version = 1078

    def get_level_mapping(level2_ids, levels=[4, 6]):
        mapping_df = pd.DataFrame(index=level2_ids)

        for level in levels:
            level_ids = client.chunkedgraph.get_roots(level2_ids, stop_layer=level)
            level_map = dict(zip(level2_ids, level_ids))
            mapping_df[f"level{level}_id"] = mapping_df.index.map(level_map)
        return mapping_df

    level2_ids = level2_features.index.get_level_values("level2_id").unique()
    level_mapping = get_level_mapping(level2_ids, levels=[4, 6])
    level_mapping["root_id"] = level_mapping.index.map(
        level2_features.reset_index().set_index("level2_id")["root_id"]
    )
    level_mapping = level_mapping.reset_index().set_index(
        ["root_id", "level6_id", "level4_id", "level2_id"]
    )

    level2_features_with_levels = (
        level2_features.join(level_mapping)
        .reorder_levels(["root_id", "level6_id", "level4_id", "level2_id"])
        .copy()
    )

    dummies = pd.get_dummies(level2_features_with_levels["pred_label"])
    dummies = dummies.reindex(columns=classes, fill_value=False)
    dummies = dummies.rename(columns={cl: f"n_level2_{cl}" for cl in classes})

    level2_features_with_levels = level2_features_with_levels.join(dummies)

    agg_funcs = {
        "area_nm2": "sum",
        "size_nm3": "sum",
        "n_segclr_pts": "sum",
    }
    agg_funcs.update({f"n_level2_{cl}": "sum" for cl in classes})
    agg_funcs.update({f"{cl}_posterior": "mean" for cl in classes})

    level_aggs = []
    for level in ["root_id", "level6_id", "level4_id"]:
        level_agg = level2_features_with_levels.groupby(level).agg(agg_funcs)
        level_agg["n_level2"] = level2_features_with_levels.groupby(level).size()
        level_agg["level"] = level.strip("_id")
        level_agg.index.name = "node_id"
        level_agg.sort_values("size_nm3", ascending=False, inplace=True)
        level_aggs.append(level_agg)

    mixed_level_df = pd.concat(level_aggs)
    mixed_level_df["random"] = np.random.uniform(0, 1, size=len(mixed_level_df))
    mixed_level_df["label"] = (
        mixed_level_df[[f"{cl}_posterior" for cl in classes]]
        .idxmax(axis=1)
        .str.replace("_posterior", "")
    )
    mixed_level_df["area_nm2"] = mixed_level_df["area_nm2"].astype(float)
    mixed_level_df["size_nm3"] = mixed_level_df["size_nm3"].astype(float)
    # size_threshold = 53_135_360

    colors = sns.color_palette("tab10").as_hex()
    palette = dict(
        zip(
            ["dendrite", "axon", "glia", "perivascular", "soma", "thick/myelin"], colors
        )
    )

    number_cols = ["random", "size_nm3", "area_nm2", "n_segclr_pts", "n_level2"] + [
        f"{cl}_posterior" for cl in classes
    ]
    prop_urls = {}
    for label, label_seg_df in mixed_level_df.groupby("label"):
        # label_seg_df = pd.concat(10_000 * [label_seg_df])
        if len(label_seg_df) > 10_000:
            label_seg_df = label_seg_df.iloc[:10_000]
        seg_prop = SegmentProperties.from_dataframe(
            label_seg_df.reset_index(),
            id_col="node_id",
            label_col="node_id",
            tag_value_cols=["level", "label"],
            number_cols=number_cols,
        )
        try:
            prop_id = client.state.upload_property_json(seg_prop.to_dict())
            prop_url = client.state.build_neuroglancer_url(
                prop_id, format_properties=True, target_site="mainline"
            )
            prop_urls[label] = prop_url
        except Exception as e:
            print(e)

    img = statebuilder.ImageLayerConfig(
        source=client.info.image_source(),
    )

    n_samples = 200
    base_level = 6
    seg_layers = []
    for label, prop_url in prop_urls.items():
        label_seg_df = mixed_level_df[mixed_level_df["label"] == label].reset_index()
        sample_ids = label_seg_df.query(f"level == 'level{base_level}'")["node_id"]
        if len(sample_ids) > n_samples:
            sample_ids = sample_ids.sample(n_samples)
        seg = statebuilder.SegmentationLayerConfig(
            name=label,
            source=client.info.segmentation_source(),
            segment_properties=prop_url,
            active=False,
            fixed_ids=sample_ids,
            fixed_id_colors=[palette[label]] * len(sample_ids),
            # fixed_ids=lumen_segments,
            skeleton_source="precomputed://middleauth+https://minnie.microns-daf.com/skeletoncache/api/v1/minnie65_phase3_v1/precomputed/skeleton",
            # view_kws={"visible": False},
        )
        seg_layers.append(seg)

    box_map = statebuilder.BoundingBoxMapper(
        point_column_a="point_column_a", point_column_b="point_column_b"
    )
    ann = statebuilder.AnnotationLayerConfig(
        name="box", mapping_rules=box_map, data_resolution=[1, 1, 1]
    )
    box_df = pd.DataFrame(
        [
            {
                "point_column_a": [
                    box_info["x_min"],
                    box_info["y_min"],
                    box_info["z_min"],
                ],
                "point_column_b": [
                    box_info["x_max"],
                    box_info["y_max"],
                    box_info["z_max"],
                ],
            }
        ]
    )

    sb = statebuilder.StateBuilder(
        layers=[img] + seg_layers + [ann],
        target_site="mainline",
        # view_kws={"zoom_3d": 0.001, "zoom_image": 0.0000001},
        client=client,
    )

    state_dict = sb.render_state(data=box_df, return_as="dict")

    for layer in state_dict["layers"]:
        if layer["type"] == "segmentation":
            layer["visible"] = False

    state_dict["layout"] = "3d"

    sb = statebuilder.StateBuilder(
        base_state=state_dict,
        target_site="mainline",
        client=client,
    )
    return sb.render_state(return_as="html")


client = CAVEclient("minnie65_public", version=1078)

cf = CloudFiles("gs://allen-minnie-phase3/vasculature_feature_pulls/box_info/")

targets = load_dataframe(cf, "targets_2024-08-19.csv.gz", index_col=0)

box_params = load_dataframe(cf, "box_params_2024-08-19.csv.gz", index_col=0)

# %%
out_cf = CloudFiles(
    "gs://allen-minnie-phase3/vasculature_feature_pulls/segclr/2024-08-19"
)

seg_res = np.array(client.chunkedgraph.segmentation_info["scales"][0]["resolution"])

model = load_model("segclr_logreg_bdp")
classes = model.classes_

distance_threshold = 2_000

test = True


def extract_features_for_box(box_id):
    box_info = box_params.loc[box_id]
    box_name = box_info["BranchTypeName"]
    bounds_min_cg = (box_info[["x_min", "y_min", "z_min"]].values / seg_res).astype(int)
    bounds_max_cg = (box_info[["x_max", "y_max", "z_max"]].values / seg_res).astype(int)
    bounds_cg = np.array([bounds_min_cg, bounds_max_cg])
    bounds_nm = bounds_cg * seg_res

    sub_target_df = targets[targets["box_id"] == box_id]
    query_ids = sub_target_df.index
    if test:
        query_ids = query_ids[:100]

    currtime = time.time()

    segclr_query = SegCLRQuery(
        client,
        verbose=True,
        n_jobs=8,
        continue_on_error=True,
        version=943,
        components=64,
    )
    segclr_query.set_query_ids(query_ids)
    segclr_query.set_query_bounds(bounds_nm)
    segclr_query.get_features()
    segclr_features = segclr_query.features_
    found_ids = np.unique(segclr_features.index.get_level_values("root_id"))

    level2_query = Level2Query(
        client,
        verbose=True,
        n_jobs=8,
        continue_on_error=True,
        attributes=["rep_coord_nm", "size_nm3", "area_nm2"],
    )
    level2_query.set_query_ids(found_ids)
    level2_query.set_query_bounds(bounds_nm)
    level2_query.get_features()
    level2_features = level2_query.features_

    root_features = (
        level2_features.groupby("root_id")["size_nm3"]
        .sum()
        .rename("approx_size_in_box")
        .to_frame()
    )
    root_features["n_level2_ids"] = level2_features.groupby("root_id").size()
    root_features["n_segclr_pts"] = segclr_features.groupby("root_id").size()

    print(f"{time.time() - currtime:.3f} seconds elapsed.")
    print()

    mappings = []
    for root_id in found_ids:
        root_segclr = segclr_features.loc[root_id]
        root_level2 = level2_query.features_.loc[root_id]

        segclr_X = root_segclr.reset_index()[["x", "y", "z"]]
        level2_X = root_level2[["x", "y", "z"]]

        root_mapping_df = map_to_closest(segclr_X, level2_X)
        root_mapping_df["root_id"] = root_id
        root_mapping_df[["x", "y", "z"]] = segclr_X
        mappings.append(root_mapping_df)

    mapping_df = pd.concat(mappings)
    mapping_df.rename(
        columns={"target_id": "level2_id", "distance_to_target": "distance_to_level2"},
        inplace=True,
    )

    segclr_features = segclr_features.join(
        mapping_df.set_index(["root_id", "x", "y", "z"])
    )

    feature_cols = np.arange(64)
    predictions = model.predict(segclr_features[feature_cols].values)
    predictions = pd.Series(
        predictions, index=segclr_features.index, name="pred_label"
    ).to_frame()
    posteriors = model.predict_proba(segclr_features[feature_cols].values)
    posteriors = pd.DataFrame(
        posteriors, index=segclr_features.index, columns=model.classes_
    ).add_suffix("_posterior")
    predictions = predictions.join(posteriors)
    segclr_features = segclr_features.join(predictions)

    write_dataframe(segclr_features, out_cf, f"{box_name}_segclr_features.csv.gz")

    filtered_segclr_features = segclr_features[
        segclr_features["distance_to_level2"] <= distance_threshold
    ]
    filtered_predictions = predictions.loc[filtered_segclr_features.index]
    level2_features["n_segclr_pts"] = filtered_segclr_features.groupby(
        ["root_id", "level2_id"]
    ).size()
    level2_predictions = filtered_predictions.groupby(
        filtered_segclr_features["level2_id"]
    )[filtered_predictions.columns.drop("pred_label")].mean()
    level2_predictions["pred_label"] = (
        level2_predictions[[f"{cl}_posterior" for cl in model.classes_]]
        .idxmax(axis=1)
        .str.replace("_posterior", "")
    )

    level2_features = level2_features.join(level2_predictions)

    write_dataframe(level2_features, out_cf, f"{box_name}_level2_features.csv.gz")

    return 1


tq = TaskQueue("https://sqs.us-west-2.amazonaws.com/629034007606/ben-skedit-dead")


def stop_fn(elapsed_time):
    if elapsed_time > 3600 * 3:
        return True


lease_seconds = 3 * 3600

run = bool(os.environ.get("RUN_JOBS", False))
if run:
    tq.poll(lease_seconds=lease_seconds, verbose=False, tally=False)


# %%

request = True
if request:
    tasks = (
        partial(extract_features_for_box, box_id) for box_id in box_params.index.values
    )

    tq.insert(tasks)
