"""Utilities for preparing data for model

Adapted from: https://github.com/mexchy1000/CellDART
"""

import math
import os
import subprocess
import warnings

import gffutils
import matplotlib.pyplot as plt
import matplotlib_venn
import numpy as np
import pandas as pd
import scanpy as sc
from joblib import Parallel, delayed, effective_n_jobs
from scipy.sparse import issparse
from sklearn import preprocessing

from ._utils import deprecated_to_sc_utils


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
    """returns log1pc and min/max normalized arr

    Args:
        arr (:obj:, array_like of `float`): Matrix to be normalized

    Returns:
        ndarray: Log1p-transformed and min-max scaled array
    """
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


def deprecated_to_sc_utils(func):
    """Decorator for functions moved to sc_utils.data_utils module."""

    def dep_warning(*args, **kwargs):
        warnings.warn(
            f"{func.__name__} was moved to sc_utils.data_utils @ c7929da from "
            "da_utils.data_processing @ 83173f1; "
            "this implementation has been deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
        func(*args, **kwargs)

    return dep_warning


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

    Returns:
        None: The input AnnData object is modified in-place.
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


@deprecated_to_sc_utils
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


@deprecated_to_sc_utils
def download_gtf(dir, url):
    """Downloads and extracts a GTF file from a URL.

    Args:
        dir (str): Directory to download the file to.
        url (str): URL to download the GTF file from.

    Returns:
        str: Path to the extracted GTF file.
    """
    os.makedirs(dir, exist_ok=True)

    gtf_gz = os.path.basename(url)
    gtf_gz_path = os.path.join(dir, gtf_gz)
    gtf_path = os.path.splitext(gtf_gz_path)[0]

    if not os.path.exists(gtf_path):
        if not os.path.exists(gtf_gz_path):
            print(f"Downloading {url} to {gtf_gz_path}")
            subprocess.run(
                [
                    "curl",
                    "--create-dirs",
                    "-o",
                    gtf_gz_path,
                    url,
                ]
            )
        print(f"Unzipping {gtf_gz_path}")
        subprocess.run(
            [
                "gunzip",
                gtf_gz_path,
            ]
        )

    return gtf_path


@deprecated_to_sc_utils
def get_reference_genome_db(
    dir,
    gtf_fname="Homo_sapiens.GRCh38.84.gtf",
    cache_fname=None,
    use_cache=True,
):
    """Creates or loads a gffutils database from a GTF file.

    Adapted from Ryan Dale's (https://www.biostars.org/u/528/) comment at
    https://www.biostars.org/p/152517/.

    Args:
        dir (str): Directory containing the GTF file.
        gtf_fname (str): Filename of the GTF file. Defaults to
            "Homo_sapiens.GRCh38.84.gtf".
        cache_fname (str, optional): Filename for the database cache. If None,
            will use the GTF filename with a .db extension. Defaults to None.
        use_cache (bool): Whether to use an existing cache file. Defaults to True.

    Returns:
        gffutils.FeatureDB: The database of genomic features.
    """
    if cache_fname is None:
        cache_fname = os.path.splitext(gtf_fname)[0] + ".db"

    gtf_path = os.path.join(dir, gtf_fname)
    cache_path = os.path.join(dir, cache_fname)
    id_spec = {
        "exon": "exon_id",
        "gene": "gene_id",
        "transcript": "transcript_id",
        # # [1] These aren't needed for speed, but they do give nicer IDs.
        # 'CDS': [subfeature_handler],
        # 'stop_codon': [subfeature_handler],
        # 'start_codon': [subfeature_handler],
        # 'UTR':  [subfeature_handler],
    }
    if os.path.exists(cache_path) and use_cache:
        db = gffutils.FeatureDB(cache_path)
    else:
        db = gffutils.create_db(
            gtf_path,
            cache_path,
            # Since Ensembl GTF files now come with genes and transcripts already in
            # the file, we don't want to spend the time to infer them (which we would
            # need to do in an on-spec GTF file)
            disable_infer_genes=True,
            disable_infer_transcripts=True,
            # Here's where we provide our custom id spec
            id_spec=id_spec,
            # "create_unique" runs a lot faster than "merge"
            # See https://pythonhosted.org/gffutils/database-ids.html#merge-strategy
            # for details.
            merge_strategy="create_unique",
            verbose=True,
            force=True,
        )

        for f in db.featuretypes():
            if f == "gene":
                continue

            db.delete(db.features_of_type(f), make_backup=False)

    return db


@deprecated_to_sc_utils
def populate_vars_from_ref(adata, db):
    """Populates the var attribute of an AnnData object with information from a reference genome.

    Args:
        adata (:obj: AnnData): AnnData object to populate.
        db (gffutils.FeatureDB): Reference genome database.

    Returns:
        None: The input AnnData object is modified in-place.
    """
    if adata.var_names.name == "gene_ids":
        gene_ids = adata.var_names.to_series()
    else:
        gene_ids = adata.var["gene_ids"]

    attrs = gene_ids.map(lambda x: dict(db[x].attributes))
    attrs = pd.DataFrame.from_records(attrs, index=attrs.index).agg(lambda x: x.str[0])

    if "gene_ids" in attrs.columns:
        attrs = attrs.drop(columns=["gene_ids"])
    if "gene_id" in attrs.columns:
        attrs = attrs.drop(columns=["gene_id"])

    adata.var = adata.var.join(attrs, how="left", validate="one_to_one")
