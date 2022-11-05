import os
import pandas as pd
import torch.utils.data
from monai.transforms import LoadImage, AddChannel, ScaleIntensity, EnsureType, Compose
import numpy as np

from src.CONSTANTS import FILE_PATHS
from src.data_utils.CustomDataset import GeomCnnDataset


def get_image_files(data_dir="TRAIN_DATA_DIR"):
    file_names = []
    labels = []
    subject_ids = sorted(os.listdir(FILE_PATHS[data_dir]))
    scalars = FILE_PATHS["FEATURE_DIRS"]
    attr = get_attributes()
    for sub in subject_ids:
        feat_tuple = []
        sub_path = os.path.join(FILE_PATHS[data_dir], sub)
        if not os.path.isdir(sub_path):
            continue
        n_feat = [os.path.join(sub_path, f) for f in os.listdir(sub_path)
                  if os.path.isdir(os.path.join(sub_path, f))]
        if len(n_feat) == 0:
            continue
        sub_attr = attr.loc[attr["CandID"] == int(sub)]
        if sub_attr.size == 0:
            continue
        group = sub_attr["group"].values[0]
        if group == "LR-neg":
            continue
        elif group == "HR-neg":
            labels.append(0)
        else:
            labels.append(1)
        for i, s in enumerate(scalars):
            feat_tuple.append(os.path.join(sub_path, s, "left_" + s +
                                           FILE_PATHS["FILE_SUFFIX"][i]) + ".jpeg")
            feat_tuple.append(os.path.join(sub_path, s, "right_" + s +
                                           FILE_PATHS["FILE_SUFFIX"][i]) + ".jpeg")
        file_names.append(feat_tuple)
    return file_names, labels


def get_image_files_2(data_dir="TRAIN_DATA_DIR"):
    file_names = []
    labels = []
    subject_ids = sorted(os.listdir(FILE_PATHS[data_dir]))
    scalars = FILE_PATHS["FEATURE_DIRS"]
    attr = get_attributes()
    for sub in subject_ids:
        sub_path = os.path.join(FILE_PATHS[data_dir], sub)
        if not os.path.isdir(sub_path):
            continue
        n_feat = [os.path.join(sub_path, f) for f in os.listdir(sub_path)
                  if os.path.isdir(os.path.join(sub_path, f))]
        if len(n_feat) == 0:
            continue
        sub_attr = attr.loc[attr["CandID"] == int(sub)]
        if sub_attr.size == 0:
            continue
        for i, s in enumerate(scalars):
            feat_tuple = [os.path.join(sub_path, s, "left_" + s +
                                       FILE_PATHS["FILE_SUFFIX"][i]) + ".jpeg",
                          os.path.join(sub_path, s, "right_" + s +
                                       FILE_PATHS["FILE_SUFFIX"][i]) + ".jpeg"]
            file_names.append(feat_tuple)
            labels.append(i)
    return file_names, labels


def min_max_normalizer(df_column):
    return (df_column - df_column.min()) / (df_column.max() - df_column.min())


def process_attr(attr):
    attr["ASD_Ever_DSMV"] = np.where(attr["ASD_Ever_DSMV"].str.contains("No DSMV ever administered"), attr["ASD_Latest_DSMIV"], attr["ASD_Ever_DSMV"])
    attr["group"] = np.where(attr["ASD_Ever_DSMV"].str.contains("ASD-"), "HR-neg", "Noval")
    attr["group"] = np.where(attr["ASD_Ever_DSMV"].str.contains("ASD\+") | attr["ASD_Latest_DSMIV"].str.contains("ASD\+") | attr["V24-DSMIV"].str.contains("YES") | attr["V36-DSMIV"].str.contains("YES"), "HR-ASD", attr["group"])
    attr["V06 demographics,Age_at_visit_start"] = attr["V06 demographics,Age_at_visit_start"].fillna(6.0)
    attr["V06 demographics,Age_at_visit_start"] = attr["V06 demographics,Age_at_visit_start"] / 6.0 - 1.0
    attr["V12 demographics,Age_at_visit_start"] = attr["V12 demographics,Age_at_visit_start"].fillna(12.0)
    attr["V12 demographics,Age_at_visit_start"] = attr["V12 demographics,Age_at_visit_start"] / 12.0 - 1.0
    attr = attr[attr["group"].str.contains("HR-neg") | attr["group"].str.contains("HR-ASD")]
    return attr


def get_image_files_3(data_dir="TRAIN_DATA_DIR"):
    file_names = []
    subject_ids = sorted(os.listdir(FILE_PATHS[data_dir]))
    scalars = FILE_PATHS["FEATURE_DIRS"]
    time_points = FILE_PATHS["TIME_POINTS"]
    attr = get_attributes2()
    attr = process_attr(attr)
    attr["CandID"] = attr["CandID"].apply(str)
    attr = attr[attr["CandID"].isin(subject_ids)]
    attr["Gender"] = attr["Sex"].map({"Male": 0, "Female": 1})
    attr["group"] = attr["group"].map({"HR-neg": 0, "HR-ASD": 1})
    # attr["ADOS_restricted_repetitive_behavior_total"] = min_max_normalizer(attr["ADOS_restricted_repetitive_behavior_total"]).astype(float)
    # attr["ADOS_severity_score_lookup"] = min_max_normalizer(attr["ADOS_severity_score_lookup"]).astype(float)
    # attr["ADOS_social_affect_restricted_repetitive_behavior_total"] = min_max_normalizer(
    #     attr["ADOS_social_affect_restricted_repetitive_behavior_total"]).astype(float)
    cand_ids = []
    count_empty = 0
    for sub in subject_ids:
        feat_tuple = []
        sub_paths = [os.path.join(FILE_PATHS[data_dir], sub, t) for t in time_points]
        skip_sub = False
        for sub_path in sub_paths:
            if not os.path.isdir(sub_path):
                skip_sub = True
                break
        if skip_sub:
            count_empty += 1
            continue

        for sub_path in sub_paths:
            n_feat = [os.path.join(sub_path, f) for f in os.listdir(sub_path)
                      if os.path.isdir(os.path.join(sub_path, f))]
            if len(n_feat) == 0:
                continue
            for i, s in enumerate(scalars):
                feat_tuple.append(os.path.join(sub_path, s, "left_" + s +
                                               FILE_PATHS["FILE_SUFFIX"][i]) + ".jpeg")
                feat_tuple.append(os.path.join(sub_path, s, "right_" + s +
                                               FILE_PATHS["FILE_SUFFIX"][i]) + ".jpeg")
        file_names.append(tuple(feat_tuple))
        cand_ids.append(sub)
    print(f"Empty files: {count_empty}")
    file_name_df = pd.DataFrame.from_dict({"CandID": cand_ids, "FILEPATHS": file_names})
    final_df = attr.merge(file_name_df, on="CandID")
    return final_df


# def get_test_dataloader():
#     test_files, test_labels = get_image_files_3("TEST_DATA_DIR")
#     test_transform = Compose(
#         [LoadImage(image_only=True),
#          AddChannel(),
#          ScaleIntensity(),
#          EnsureType()]
#     )
#     _ds = GeomCnnDataset(test_files, test_labels, test_transform)
#     return torch.utils.data.DataLoader(_ds, batch_size=100)


def get_attributes():
    file_path = os.path.join(FILE_PATHS["TRAIN_DATA_DIR"], "DX_and_Dem.csv")
    ados_path = os.path.join(FILE_PATHS["TRAIN_DATA_DIR"], "ADOS.csv")
    attr = pd.read_csv(open(file_path))
    ados = pd.read_csv(open(ados_path))
    return attr.merge(ados, on="CandID")


def get_attributes2():
    file_path = os.path.join(FILE_PATHS["TRAIN_DATA_DIR"], "DemIBIS1-2.csv")
    attr = pd.read_csv(open(file_path))
    file_path = os.path.join(FILE_PATHS["TRAIN_DATA_DIR"], "DX_IBIS_Sept22.csv")
    attr_group = pd.read_csv(open(file_path))
    attr_group["CandID"] = attr_group["V24 demographics,CandID"]
    attr["CandID"] = attr["CandID"].astype(str)
    attr_m = attr.merge(attr_group[["CandID", "V24 demographics,DX_Subgroups","V36 demographics,DX_Subgroups"]], on="CandID", how="left")
    attr_m.rename(columns={"V24 demographics,DX_Subgroups": "V24-DSMIV", "V36 demographics,DX_Subgroups": "V36-DSMIV"}, inplace=True)
    return attr_m


def get_counts(data_dir):
    file_names = []
    labels = []
    subject_ids = sorted(os.listdir(FILE_PATHS[data_dir]))
    scalars = FILE_PATHS["FEATURE_DIRS"][0]
    time_points = FILE_PATHS["TIME_POINTS"]
    attr = get_attributes()

    for tp in ["V06", "V12", "V24"]:
        count = {"HR-neg": 0, "HR-ASD": 1}
        for sub in subject_ids:
            sub_path = os.path.join(FILE_PATHS[data_dir], sub, tp)
            if not os.path.isdir(sub_path):
                continue
            n_feat = os.path.join(sub_path, scalars,
                                  "left_" + scalars +
                                  FILE_PATHS["FILE_SUFFIX"][0] + ".jpeg")
            if not os.path.exists(n_feat):
                continue
            sub_attr = attr.loc[attr["CandID"] == int(sub)]
            if sub_attr.size == 0:
                continue
            group = sub_attr["group"].values[0]
            if group == "LR-neg":
                continue
            elif group == "HR-neg":
                labels.append(0)
            else:
                labels.append(1)
            count[group] += 1
        print(f"Time {tp}, Count {count}")


def plot_ados_scores(data_dir="TRAIN_DATA_DIR"):
    subject_ids = sorted(os.listdir(FILE_PATHS[data_dir]))
    attr = get_attributes2()
    attr["CandID"] = attr["CandID"].apply(str)
    attr = attr[attr["CandID"].isin(subject_ids)]
    attr = attr[~(attr["group"].str.contains("LR"))]
    attr["ADOS_restricted_repetitive_behavior_total"] = np.random.normal(
        attr["ADOS_restricted_repetitive_behavior_total"],
        attr["ADOS_restricted_repetitive_behavior_total"]*0.1)
    attr["ADOS_severity_score_lookup"] = np.random.normal(
        attr["ADOS_severity_score_lookup"],
        attr["ADOS_severity_score_lookup"] * 0.1)
    attr["ADOS_social_affect_restricted_repetitive_behavior_total"] = np.random.normal(
        attr["ADOS_social_affect_restricted_repetitive_behavior_total"],
        attr["ADOS_social_affect_restricted_repetitive_behavior_total"] * 0.1)
    # attr["Gender"] = attr["Gender"].map({"Male": 3, "Female": 5})
    import plotly.express as px
    fig = px.scatter(
        attr,
        x="ADOS_social_affect_restricted_repetitive_behavior_total",
        y="ADOS_severity_score_lookup",
        color="group",
        symbol="Gender"
    )
    fig.show()
    fig = px.scatter(
        attr,
        x="ADOS_restricted_repetitive_behavior_total",
        y="ADOS_severity_score_lookup",
        color="group",
        symbol="Gender"
    )
    fig.show()
    fig = px.scatter(
        attr,
        x="ADOS_restricted_repetitive_behavior_total",
        y="ADOS_social_affect_restricted_repetitive_behavior_total",
        color="group",
        symbol="Gender"
    )
    fig.show()


if __name__ == '__main__':
    get_counts("TRAIN_DATA_DIR")
    # plot_ados_scores()



