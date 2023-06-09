"""Utilities for preparing data for model

Adapted from: https://github.com/mexchy1000/CellDART
"""
import math

import matplotlib.pyplot as plt
import matplotlib_venn
import numpy as np
import pandas as pd
import scanpy as sc
from joblib import Parallel, delayed, effective_n_jobs
from scipy.sparse import issparse
from sklearn import preprocessing


def random_mix(X, y, nmix=5, n_samples=10000, seed=0, n_jobs=1):
    """Creates a weighted average random sampling of gene expression, and the
    corresponding weights.

    Args:
        X (:obj:, array_like of `float`): Matrix containing single cell samples
            and their GEx profiles
        y (:obj:, array_like of `int`): Array of cell type labels
        nmix (int): Number of cells per spot. Defaults to 5.
        n_samples (int): Number of training pseudospots. Defaults to 10000.
        seed (int): Random seed. Defaults to 0.
        n_jobs (int): Number of jobs to run in parallel. Defaults to 1.

    Shape:
        - X: `(N, C)`, where `N` is the number of single cell samples, and `C`
        is the number of genes
        - y: `(N,)`, where `N` is the number of single cell samples
        - pseudo_gex: :math: `(N_{out}, C)` where
        :math: `N_{out} = \text{n\_samples}` and `C` is the number of genes
        - ctps: `(N_{out}, C_{types})`
        where :math: `N_{out} = \text{n\_samples}` and :math: `C_{types}`
        is the number of cell types.

    Returns:
        A tuple of (pseudo_gex, ctps) where:
         - pseudo_gex (ndarray): Matrix containing pseudo-spot samples and their
         cell type proportion weighted averages
         - ctps (ndarray): Matrix containing pseudo-spot samples and their cell
         type proportions

    """
    # Define empty lists
    pseudo_gex, ctps = [], []

    ys_ = preprocessing.OneHotEncoder().fit_transform(y.reshape(-1, 1))
    try:
        ys_ = ys_.toarray()  # type: ignore
    except AttributeError:
        pass
    _X = X.toarray() if issparse(X) else X

    rstate = np.random.RandomState(seed)
    fraction_all = rstate.rand(n_samples, nmix)
    randindex_all = rstate.randint(len(_X), size=(n_samples, nmix))

    def _get_pseudo_sample(i):
        # fraction: random fraction across the "nmix" number of sampled cells
        fraction = fraction_all[i]
        fraction = fraction / np.sum(fraction)
        fraction = np.reshape(fraction, (nmix, 1))

        # Random selection of the single cell data by the index
        randindex = randindex_all[i]
        ymix = ys_[randindex]
        # Calculate the fraction of cell types in the cell mixture
        yy = np.sum(ymix * fraction, axis=0)
        # Calculate weighted gene expression of the cell mixture
        XX = np.asarray(_X[randindex]) * fraction
        XX_ = np.sum(XX, axis=0)

        return XX_, yy

    batch_size = math.ceil(n_samples / effective_n_jobs(n_jobs))
    pseudo_samples = Parallel(n_jobs=n_jobs, verbose=1, batch_size=batch_size)(
        delayed(_get_pseudo_sample)(i) for i in range(n_samples)
    )
    pseudo_gex, ctps = zip(*pseudo_samples)

    # Add cell type fraction & composite gene expression in the list
    # ctps.append(yy)
    # pseudo_gex.append(XX_)

    pseudo_gex = np.asarray(pseudo_gex)
    ctps = np.asarray(ctps)

    return pseudo_gex, ctps


def log_minmaxscale(arr):
    """returns log1pc and min/max normalized arr"""
    arrd = len(arr)
    arr = np.log1p(arr)

    arr_minus_min = arr - np.reshape(np.min(arr, axis=1), (arrd, 1))
    min2max = np.reshape((np.max(arr, axis=1) - np.min(arr, axis=1)), (arrd, 1))
    return arr_minus_min / min2max


def rank_genes(adata_sc):
    """Ranks genes for cell_subclasses.

    Args:
        adata_sc (:obj: AnnData): Single-cell data with cell_type.

    Returns:
        A DataFrame containing the ranked genes by cell_type.

    """
    sc.tl.rank_genes_groups(adata_sc, groupby="cell_type", method="wilcoxon")

    genelists = adata_sc.uns["rank_genes_groups"]["names"]
    df_genelists = pd.DataFrame.from_records(genelists)

    return df_genelists


def select_marker_genes(
    adata_sc,
    adata_st,
    n_markers=None,
    genelists_path=None,
    force_rerank=False,
):
    """Ranks genes for cell_subclasses, finds set of top genes and selects those
    features in adatas.

    Args:
        adata_sc (:obj: AnnData): Single-cell data with cell_type.
        adata_st (:obj: AnnData): Spatial transcriptomic data.
        n_markers (Any): Number of top markers to include for each
            cell_type. If not provided, or not truthy, all genes will be
            included.
        genelists_path (Any): Path to save/load ranked genes. If not provided,
            will not save/load. Defaults to None.
        force_rerank (bool): If True, will force rerank of genes and will not
            save. Defaults to False.

    Returns:
        A tuple of a tuple of (adata_sc, adata_st) with the reduced set of
        marker genes, and a DataFrame containing the ranked genes by
        cell_type.

    """

    # Load or rank genes
    if force_rerank or not genelists_path:
        df_genelists = rank_genes(adata_sc)
        if genelists_path:
            df_genelists.to_pickle(genelists_path)
    else:
        try:
            df_genelists = pd.read_pickle(genelists_path)
        except FileNotFoundError as e:
            print(e)
            df_genelists = rank_genes(adata_sc)
            if genelists_path:
                df_genelists.to_pickle(genelists_path)

    all_sc_genes = set(adata_sc.var.index)
    all_st_genes = set(adata_st.var.index)

    if n_markers:
        # Get set of all top genes for cluster
        res_genes = []
        for column in df_genelists.head(n_markers):
            res_genes.extend(df_genelists.head(n_markers)[column].tolist())
        res_genes_ = list(set(res_genes))

        top_genes_sc = set(res_genes)

        # Find gene intersection
        inter_genes = [val for val in adata_st.var.index if val in res_genes_]

        fig, ax = plt.subplots()
        matplotlib_venn.venn3_unweighted(
            [all_sc_genes, all_st_genes, top_genes_sc],
            set_labels=(
                "SC genes",
                "ST genes",
                f"Union of top {n_markers} genes for all clusters",
            ),
            ax=ax,
        )
    else:
        inter_genes_set = list(all_sc_genes.intersection(all_st_genes))
        inter_genes = [val for val in adata_st.var.index if val in inter_genes_set]

        fig, ax = plt.subplots()
        matplotlib_venn.venn2_unweighted(
            [all_sc_genes, all_st_genes],
            set_labels=(
                "SC genes",
                "ST genes",
            ),
            ax=ax,
        )

    print("Selected Feature Gene number", len(inter_genes))

    # Return adatas with subset of genes
    return (adata_sc[:, inter_genes], adata_st[:, inter_genes]), df_genelists, (fig, ax)


def qc_sc(
    adata,
    min_cells=3,
    min_genes=200,
    pct_counts_mt=5,
    remove_mt=False,
):
    """Performs QC on single-cell data.

    Args:
        adata (:obj: AnnData): Single-cell data.
        min_cells (int): Minimum number of cells a gene must be expressed in.
            Defaults to 3.
        min_genes (int): Minimum number of genes a cell must express. Defaults
            to 200.
        pct_counts_mt (int): Maximum percentage of mitochondrial genes a cell
            can express. Defaults to 5.
        remove_mt (bool): Remove mitochondrial genes. Defaults to False.
    """
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, min_genes=min_genes)

    # annotate the group of mitochondrial genes as 'mt'
    adata.var["mt"] = adata.var_names.str.lower().str.startswith("mt-")
    print(f"{adata.var['mt'].sum()} mitochondrial genes")
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    adata = adata[adata.obs.pct_counts_mt < pct_counts_mt, :]

    if remove_mt:
        adata = adata[:, ~adata.var["mt"]]


def safe_stratify(stratify):
    """Makes stratify arg for sklearn splits safe when there is only one class.

    Args:
        stratify (array-like): Array to stratify.

    Returns:
        `stratify` if there is more than one unique value, else None.

    """
    if len(np.unique(stratify)) > 1:
        return stratify

    return None
