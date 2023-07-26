"""Data loading functions."""
import os
import pickle
from copy import copy

import h5py
import scanpy as sc

DEFAULT_N_SPOTS = 200000
DEFAULT_N_MARKERS = 20
DEFAULT_N_MIX = 8
DEF_ST_ID = "spatialLIBD"
DEF_SC_ID = "GSE144136"
SPLITS = ("train", "val", "test")


def get_model_rel_path(
    model_name,
    model_version,
    dset="dlpfc",
    sc_id=DEF_SC_ID,
    st_id=DEF_ST_ID,
    n_markers=DEFAULT_N_MARKERS,
    all_genes=False,
    n_mix=DEFAULT_N_MIX,
    n_spots=DEFAULT_N_SPOTS,
    st_split=False,
    samp_split=False,
    scaler_name="minmax",
    lib_seed_path="random",
    **kwargs,
):
    """Get path relative to data or results directory for a given run.

    Args:
        model_name (str): Name of the model.
        model_version (str): Version name of the model.
        dset (str): Dataset to use. Default: "dlpfc".
        sc_id (str): ID of the sc dataset to use. This the GEO accession number.
            Default: "GSE144136".
        st_id (str): ID of the st dataset to use. This the GEO accession number,
            or "spatialLIBD" for the spatialLIBD dataset. Default:
            "spatialLIBD".
        n_markers (int): Number of marker genes used per sc cluster. Default:
            20.
        all_genes (bool): Whether all genes are used. Default: False.
        n_mix (int): Number of sc samples in each spot. Default: 8.
        n_spots (int): Number of spots generated for training set. Default:
            20000.
        st_split (bool): Whether to use a train/val/test split for spatial data.
            Default: False.
        scaler_name (str): Name of the scaler to use. Default: "minmax".
        lib_seed_path: Seed used for pytorch rng. Default: "random".
        **kwargs: Catches additional unused arguments.

    Returns:
        str: Path relative to data or results directory.

    """
    selected_rel_path = get_selected_rel_path(sc_id, st_id, n_markers, all_genes)
    data_str = f"{n_mix}mix_{n_spots}spots"
    if st_split:
        data_str += "-stsplit"
    # elif samp_split:
    #     data_str += "-sampsplit"

    return os.path.join(
        model_name,
        dset,
        selected_rel_path,
        data_str,
        scaler_name,
        model_version,
        lib_seed_path,
    )


def get_dset_dir(data_dir, dset="dlpfc"):
    return os.path.join(data_dir, dset)


def get_selected_dir(
    dset_dir,
    sc_id=DEF_SC_ID,
    st_id=DEF_ST_ID,
    n_markers=DEFAULT_N_MARKERS,
    all_genes=False,
    normalize=True,
    **kwargs,
) -> str:
    """Get directory of GEx data with selected gene subset between sc and st.

    Args:
        dset_dir (str): Data directory.\
        sc_id (str): ID of the sc dataset to use. This the GEO accession number.
            Default: "GSE144136".
        st_id (str): ID of the st dataset to use. This the GEO accession number,
            or "spatialLIBD" for the spatialLIBD dataset. Default:
            "spatialLIBD".
        n_markers (int): Number of marker genes used per sc cluster. Default:
            20.
        all_genes (bool): Whether all genes are used. Default: False.
        **kwargs: Catches additional unused arguments.

    Returns:
        str: Path to directory with data containing data with gene subset
            selected.

    """
    selected_rel_path = get_selected_rel_path(
        sc_id, st_id, n_markers, all_genes, normalize=normalize
    )
    return os.path.join(dset_dir, "preprocessed", selected_rel_path)


def get_selected_rel_path(sc_id, st_id, n_markers, all_genes, normalize=True) -> str:
    """Get path of GEx data with selected gene subset between sc and st,
    relative to top-level data directory.

    Args:
        sc_id (str): ID of the sc dataset to use. This the GEO accession number.
        st_id (str): ID of the st dataset to use. This the GEO accession number,
            or "spatialLIBD" for the spatialLIBD dataset.
        n_markers (int): Number of marker genes used per sc cluster.
        all_genes (bool): Whether all genes are used.

    Returns:
        str: Path to directory with data containing data with gene subset
            selected, relative to top-level data directory.

    """

    intersec_name = f"{sc_id}_{st_id}"
    n_markers_str = "all" if all_genes else f"{n_markers}markers"
    selected_rel_path = os.path.join(intersec_name, n_markers_str)
    if normalize == False:
        selected_rel_path = os.path.join(selected_rel_path, "raw_counts")
    return selected_rel_path


def load_spatial(selected_dir, scaler_name, **kwargs):
    """Loads preprocessed spatial data.

    Args:
        selected_dir (str): Directory of selected data.
        scaler_name (str): Name of the scaler to use.
        st_split (bool): Whether to use a train/val/test split for spatial data.
        samp_split (bool): Whether to use a train/val/test split by slice.
        one_model (bool): Whether to use all spatial samples
            for training, or separate by sample. Default: False.
        **kwargs: Catches additional unused arguments. Passed to
            `load_st_spots`.

    Returns:
        Tuple::

            (`mat_sp_d`, `mat_sp_meta_d`, `st_sample_id_l`)


        `mat_sp_d` is a dict of spatial data by sample and split. If not
        `st_split`, then 'val' will not be contained and 'test' will point to
        'train'.

        `mat_sp_meta_d` dict matching the format of `mat_sp_d`, containing
        dataframes of the metadata for each sample/subsample.


        `st_sample_id_l` is a list of sample ids for spatial data.

    """
    processed_data_dir = os.path.join(selected_dir, scaler_name)

    mat_sp_d, mat_sp_meta_d = load_st_spots(processed_data_dir, **kwargs)
    st_sample_id_l = load_st_sample_names(selected_dir)

    return mat_sp_d, mat_sp_meta_d, st_sample_id_l


def load_st_spots(
    processed_data_dir,
    st_split=False,
    samp_split=False,
    one_model=False,
    **kwargs,
):
    """Loads spatial spots.

    Args:
        processed_data_dir (str): Directory to load data from.
        st_split (bool): Whether to use a train/val/test split for spatial data.
        samp_split (bool): Whether to use a train/val/test split by slice.
        one_model (bool): Whether to use all spatial samples
            for training, or separate by sample. Default: False.
        **kwargs: Catches additional unused arguments.

    Returns:
        Tuple::

            (`mat_sp_d`, `mat_sp_meta_d`)


        `mat_sp_d` is a dict of spatial data by sample and split. If not
        `st_split`, then 'val' will not be contained and 'test' will point to
        'train'.

        `mat_sp_meta_d` dict matching the format of `mat_sp_d`, containing
        dataframes of the metadata for each sample/subsample.


    """
    scaler_name = processed_data_dir.rstrip("/").split("/")[-1]  # get last dir

    if samp_split:
        fname = "mat_sp_samp_split_d"
    elif st_split:
        fname = "mat_sp_split_d"
    else:
        fname = "mat_sp_train_d"

    if one_model and scaler_name != "unscaled":
        fname = f"{fname}_one_model.h5ad"
    else:
        fname = f"{fname}.h5ad"

    in_path = os.path.join(processed_data_dir, fname)
    adata = sc.read_h5ad(in_path)
    mat_sp_d = {}
    mat_sp_meta_d = {}
    for l1 in adata.obs.iloc[:, 0].unique():
        mat_sp_d[l1] = {}
        mat_sp_meta_d[l1] = {}
        for l2 in adata.obs.iloc[:, 1].unique():
            sub_samp = adata[(adata.obs.iloc[:, 0] == l1) & (adata.obs.iloc[:, 1] == l2)]
            if len(sub_samp) > 0:
                mat_sp_d[l1][l2] = copy(sub_samp.X)
                mat_sp_meta_d[l1][l2] = sub_samp.obs.copy()
        if not st_split and not samp_split and not one_model:
            mat_sp_d[l1]["test"] = mat_sp_d[l1]["train"]
            mat_sp_meta_d[l1]["test"] = mat_sp_meta_d[l1]["train"]

    if one_model and not st_split:
        mat_sp_d["test"] = mat_sp_d["train"]
        mat_sp_meta_d["test"] = mat_sp_meta_d["train"]

    return mat_sp_d, mat_sp_meta_d


def load_st_sample_names(selected_dir):
    with open(os.path.join(selected_dir, "st_sample_id_l.pkl"), "rb") as f:
        st_sample_id_l = pickle.load(f)

    return st_sample_id_l


def load_sc(selected_dir, scaler_name, **kwargs):
    """Loads preprocessed sc data.

    Args:
        selected_dir (str): Directory of selected data.
        scaler_name (str): Name of the scaler to use.
        n_mix (int): Number of sc samples in each spot. Default: 8.
        n_spots (int): Number of spots to generate. for training set. Default:
            20000.
        **kwargs: Catches additional unused arguments. Passed to
            `load_pseudospots`.

    Returns:
        Tuple::

            (
                sc_mix_d,
                lab_mix_d,
                sc_sub_dict,
                sc_sub_dict2,
            )

         - `sc_mix_d` is a dict of sc data by split.
         - `lab_mix_d` is a dict of sc labels by split.
         - `sc_sub_dict` is a dict of 'label_id' to 'label_name' for sc data.
         - `sc_sub_dict2` is a dict of 'label_name' to 'label_id' for sc data.

    """
    processed_data_dir = os.path.join(selected_dir, scaler_name)
    sc_mix_d, lab_mix_d = load_pseudospots(processed_data_dir, **kwargs)

    # Load helper dicts / lists
    sc_sub_dict, sc_sub_dict2 = load_sc_dicts(selected_dir)

    return sc_mix_d, lab_mix_d, sc_sub_dict, sc_sub_dict2


def load_sc_dicts(selected_dir):
    with open(os.path.join(selected_dir, "sc_sub_dict.pkl"), "rb") as f:
        sc_sub_dict = pickle.load(f)
    with open(os.path.join(selected_dir, "sc_sub_dict2.pkl"), "rb") as f:
        sc_sub_dict2 = pickle.load(f)

    return sc_sub_dict, sc_sub_dict2


def _ps_fname(n_mix, seed_int=-1):
    if seed_int >= 0:
        return f"sc_{n_mix}mix_{seed_int}.hdf5"
    else:
        return f"sc_{n_mix}mix.hdf5"


def load_pseudospots(
    processed_data_dir,
    n_mix=DEFAULT_N_MIX,
    n_spots=None,
    seed_int=-1,
    **kwargs,
):
    """Loads preprocessed sc pseudospots.

    Args:
        processed_data_dir (str): Directory of processed data.
        n_mix (int): Number of sc samples in each spot. Default: 8.
        n_spots (int): Number of spots to generate. for training set. Default:
            20000.
        **kwargs: Catches additional unused arguments.

    Returns:
        Tuple::

            (sc_mix_d, lab_mix_d)

         - `sc_mix_d` is a dict of sc data by split.
         - `lab_mix_d` is a dict of sc labels by split.

    """
    sc_mix_d = {}
    lab_mix_d = {}
    with h5py.File(os.path.join(processed_data_dir, _ps_fname(n_mix, seed_int=seed_int)), "r") as f:
        for split in SPLITS:
            sc_mix_d[split] = f[f"X/{split}"][()]  # type: ignore
            lab_mix_d[split] = f[f"y/{split}"][()]  # type: ignore

    if n_spots is not None:
        sc_mix_d["train"] = sc_mix_d["train"][:n_spots]
        lab_mix_d["train"] = lab_mix_d["train"][:n_spots]
    return sc_mix_d, lab_mix_d


def save_pseudospots(lab_mix_d, sc_mix_s_d, data_dir, n_mix, seed_int=-1):
    """Saves preprocessed sc pseudospots to hdf5 files.

    Args:
        lab_mix_d (dict): sc labels by split.
        sc_mix_s_d (dict): sc data by split.
        data_dir (str): Directory to save data to.
        n_mix (int): Number of sc samples in each spot.

    """
    with h5py.File(os.path.join(data_dir, _ps_fname(n_mix, seed_int=seed_int)), "w") as f:
        grp_x = f.create_group("X")
        grp_y = f.create_group("y")
        for split in SPLITS:
            dset = grp_x.create_dataset(split, data=sc_mix_s_d[split])
            dset = grp_y.create_dataset(split, data=lab_mix_d[split])
