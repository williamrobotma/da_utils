"""Evaluator for models."""

import glob
import itertools
import logging
import math
import os
import pickle
import re
import shutil
import tarfile
from collections import OrderedDict, defaultdict
import warnings

import harmonypy as hm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import yaml
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from joblib import Parallel, delayed, effective_n_jobs, parallel_backend
from sklearn import metrics, model_selection
from sklearn.decomposition import PCA
from sklearn.metrics import RocCurveDisplay

from src.da_models.model_utils.utils import (
    ModelWrapper,
    dict_to_lib_config,
    get_best_params_file,
)
from src.da_utils import data_loading, evaluation
from src.da_utils.output_utils import TempFolderHolder
from src.da_utils.scripts.data.preprocessing_mouse_GSE115746 import (
    cell_cluster_cell_type_to_spot_composition,
    cell_subclass_to_spot_composition,
)
from src.da_utils.scripts.data.preprocessing_spotless import get_st_sub_map

logger = logging.getLogger(__name__)

Ex_to_L_d = {
    1: {5, 6},
    2: {5},
    3: {4, 5},
    4: {6},
    5: {5},
    6: {4, 5, 6},
    7: {4, 5, 6},
    8: {5, 6},
    9: {5, 6},
    10: {2, 3, 4},
}


class Evaluator:
    def __init__(self, args_dict, metric_ctp):
        self.args_dict = args_dict
        self.metric_ctp = metric_ctp

        self.lib_config = dict_to_lib_config(self.args_dict)
        ModelWrapper.configurate(self.lib_config)

        print(f"Evaluating {self.args_dict['modelname']} on with {self.args_dict['njobs']} jobs")
        print("Using library config:")
        print(self.lib_config)

        if self.args_dict["reverse_val"]:
            print("Searching for best model through reverse validation")

            # use basic non-specific config provided first
            with open(
                os.path.join(
                    self.args_dict["configs_dir"],
                    self.args_dict["modelname"],
                    self.args_dict["config_fname"],
                ),
                "r",
            ) as f:
                basic_config = yaml.safe_load(f)
                basic_data_params = basic_config["data_params"]

            config_fname, rv_df, best_hp = get_best_params_file(
                self.args_dict["modelname"],
                basic_data_params["dset"],
                basic_data_params["sc_id"],
                basic_data_params["st_id"],
                self.args_dict["configs_dir"],
            )
            print("Best RV config:")
            print(rv_df.loc[best_hp])
        else:
            config_fname = self.args_dict["config_fname"]

        print(f"Loading config {config_fname} ... ")

        with open(
            os.path.join(
                self.args_dict["configs_dir"],
                self.args_dict["modelname"],
                config_fname,
            ),
            "r",
        ) as f:
            self.config = yaml.safe_load(f)

        print(yaml.dump(self.config))

        self.lib_params = self.config["lib_params"]
        self.data_params = self.config["data_params"]
        self.model_params = self.config["model_params"]
        self.train_params = self.config["train_params"]

        lib_seed_path = str(self.lib_params.get("manual_seed", "random"))

        model_rel_path = data_loading.get_model_rel_path(
            self.args_dict["modelname"],
            self.model_params["model_version"],
            lib_seed_path=lib_seed_path,
            **self.data_params,
        )

        model_folder = os.path.join("model", model_rel_path)

        if self.args_dict["tmpdir"]:
            real_model_folder = model_folder
            model_folder = os.path.join(self.args_dict["tmpdir"], "model")

            shutil.copytree(real_model_folder, model_folder, dirs_exist_ok=True)

        # Check to make sure config file matches config file in model folder
        with open(os.path.join(model_folder, "config.yml"), "r") as f:
            config_model_folder = yaml.safe_load(f)
        if config_model_folder != self.config:
            raise ValueError("Config file does not match config file in model folder")

        self.pretraining = self.args_dict["pretraining"] or self.train_params.get(
            "pretraining", False
        )
        if self.args_dict["reverse_val"]:
            model_rel_path_l = model_rel_path.split(os.sep)
            results_folder = os.path.join(
                "results", os.sep.join(model_rel_path_l[:3]), "reverse_val"
            )
        else:
            results_folder = os.path.join("results", model_rel_path)

        self.temp_folder_holder = TempFolderHolder()
        temp_results_folder = (
            os.path.join(self.args_dict["tmpdir"], "results") if self.args_dict["tmpdir"] else None
        )
        self.results_folder = self.temp_folder_holder.set_output_folder(
            temp_results_folder, results_folder
        )

        if self.args_dict["reverse_val"]:
            rv_df.to_csv(os.path.join(self.results_folder, "rv_df.csv"))

        sc.set_figure_params(facecolor="white", figsize=(8, 8))
        sc.settings.verbosity = 3

        self.selected_dir = data_loading.get_selected_dir(
            data_loading.get_dset_dir(
                self.data_params["data_dir"],
                dset=self.data_params.get("dset", "dlpfc"),
            ),
            **self.data_params,
        )

        print("Loading Data")
        # Load spatial data
        mat_sp_d, self.mat_sp_meta_d, st_sample_id_l = data_loading.load_spatial(
            self.selected_dir,
            **self.data_params,
        )

        if self.args_dict["test"]:
            self.splits = ("train", "val", "test")
        else:
            self.splits = ("train", "val")

        if self.data_params.get("samp_split", False):
            self.st_sample_id_d = {}
            for split in self.splits:
                if split == "val":
                    continue
                self.st_sample_id_d[split] = [
                    sid for sid in st_sample_id_l if sid in mat_sp_d[split].keys()
                ]

            self.mat_sp_d = {}
            for split in self.splits:
                if split == "val":
                    continue
                for sid in self.st_sample_id_d[split]:
                    self.mat_sp_d[sid] = mat_sp_d[split][sid]

        else:
            self.st_sample_id_d = {"": st_sample_id_l}
            self.mat_sp_d = {k: v["test"] for k, v in mat_sp_d.items()}

        # Load sc data
        (
            self.sc_mix_d,
            self.lab_mix_d,
            self.sc_sub_dict,
            self.sc_sub_dict2,
        ) = data_loading.load_sc(self.selected_dir, **self.data_params)

        self.pretrain_folder = os.path.join(model_folder, "pretrain")
        self.advtrain_folder = os.path.join(model_folder, "advtrain")
        self.samp_split_folder = os.path.join(self.advtrain_folder, "samp_split")

        self.model_fname = "final_model"
        # self.pretrain_model_path = os.path.join(pretrain_folder, f"final_model.pth")

    def gen_pca(self, sample_id, split, y_dis, emb, emb_noda=None):
        n_cols = 2 if emb_noda is not None else 1
        logger.debug("Generating PCA plots")
        fig, axs = plt.subplots(1, n_cols, figsize=(n_cols * 10, 10), squeeze=False)
        logger.debug("Calculating DA PCA")
        pca_da_df = pd.DataFrame(
            fit_pca(emb, n_components=2).transform(emb), columns=["PC1", "PC2"]
        )

        pca_da_df["domain"] = ["source" if x == 0 else "target" for x in y_dis]
        sns.scatterplot(
            data=pca_da_df,
            x="PC1",
            y="PC2",
            hue="domain",
            ax=axs[0][0],
            marker=".",
        )
        axs.flat[0].set_title("DA")

        if emb_noda is not None:
            logger.debug("Calculating no DA PCA")
            pca_noda_df = pd.DataFrame(
                fit_pca(emb_noda, n_components=2).transform(emb_noda),
                columns=["PC1", "PC2"],
            )
            pca_noda_df["domain"] = pca_da_df["domain"]

            sns.scatterplot(
                data=pca_noda_df,
                x="PC1",
                y="PC2",
                hue="domain",
                ax=axs[0][1],
                marker=".",
            )
            axs.flat[1].set_title("No DA")

        for ax in axs.flat:
            ax.set_aspect("equal", "box")
        fig.suptitle(f"{sample_id} {split}")
        fig.savefig(
            os.path.join(self.results_folder, f"PCA_{sample_id}_{split}.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    def rf50_score(self, emb_train, emb_test, y_dis_train, y_dis_test):
        logger.info("Running RF50")
        logger.debug(f"emb_train dtype: {emb_train.dtype}")
        logger.debug("fitting pca 50")
        pca = fit_pca(emb_train, n_components=min(50, emb_train.shape[1]))

        emb_train_50 = pca.transform(emb_train)
        logger.debug("transforming pca 50 test")
        emb_test_50 = pca.transform(emb_test)

        logger.debug("initialize brfc")
        n_jobs = effective_n_jobs(int(self.args_dict["njobs"]))
        clf = BalancedRandomForestClassifier(random_state=145, n_jobs=n_jobs)
        logger.debug("fit brfc")
        clf.fit(emb_train_50, y_dis_train)
        logger.debug("predict brfc")
        y_pred_test = clf.predict(emb_test_50)

        logger.debug("eval brfc")
        return metrics.balanced_accuracy_score(y_dis_test, y_pred_test)

    def evaluate_embeddings(self):
        random_states = np.asarray([225, 53, 92])

        self.rf50_d = {"da": {}, "noda": {}}

        for split in self.splits:
            for k in self.rf50_d:
                self.rf50_d[k][split] = {}

        if self.args_dict["milisi"]:
            self.miLISI_d = {"da": {}}
            if self.pretraining:
                self.miLISI_d["noda"] = {}
            for split in self.splits:
                for k in self.miLISI_d:
                    self.miLISI_d[k][split] = {}

        if self.pretraining:
            model_noda = ModelWrapper(self.pretrain_folder, name="final_model")
        else:
            model_noda = None

        splits, _ = self._get_splits_sids()

        if self.data_params.get("samp_split", False):
            model = ModelWrapper(self.samp_split_folder, name=self.model_fname)
        elif self.data_params.get("one_model", False):
            model = ModelWrapper(self.advtrain_folder, name=self.model_fname)
        else:
            model = None

        for split in splits:
            for sample_id in self.st_sample_id_d[split]:
                print(f"Calculating domain shift for {sample_id}:", end=" ")
                random_states = random_states + 1

                self._eval_embeddings_1samp(
                    sample_id, random_states, model=model, model_noda=model_noda
                )

    def _eval_embeddings_1samp(self, sample_id, random_states, model=None, model_noda=None):
        if model is None:
            model = ModelWrapper(
                os.path.join(self.advtrain_folder, sample_id), name=self.model_fname
            )

        n_jobs = effective_n_jobs(int(self.args_dict["njobs"]))

        for split, rs in zip(self.splits, random_states):
            print(split.upper(), end=" |")
            Xs, Xt = (self.sc_mix_d[split], self.mat_sp_d[sample_id])

            source_emb = model.get_embeddings(Xs, source_encoder=True)
            target_emb = model.get_embeddings(Xt, source_encoder=False)

            emb = np.concatenate([source_emb, target_emb])
            if self.pretraining:
                source_emb_noda = model_noda.get_embeddings(Xs, source_encoder=True)
                target_emb_noda = model_noda.get_embeddings(Xt, source_encoder=True)

                emb_noda = np.concatenate([source_emb_noda, target_emb_noda])
            else:
                emb_noda = None

            y_dis = np.concatenate(
                [
                    np.zeros((source_emb.shape[0],), dtype=np.int_),
                    np.ones((target_emb.shape[0],), dtype=np.int_),
                ]
            )

            if self.pretraining:
                all_embs = np.concatenate([emb, emb_noda], axis=1)
            else:
                all_embs = emb

            # undersample majority class between source and target for milisi, pca
            all_embs_bal, y_dis_bal = RandomUnderSampler(
                random_state=int.from_bytes(sample_id.encode("utf8"), "big") % 2**32
            ).fit_resample(all_embs, y_dis)

            if self.pretraining:
                emb_bal = all_embs_bal[:, : all_embs_bal.shape[1] // 2]
                emb_noda_bal = all_embs_bal[:, all_embs_bal.shape[1] // 2 :]
            else:
                emb_bal = all_embs_bal
                emb_noda_bal = None

            # pca
            self.gen_pca(sample_id, split, y_dis_bal, emb_bal, emb_noda=emb_noda_bal)

            # milisi
            if self.args_dict["milisi"]:
                if "spotless" in self.data_params.get("st_id", set()):
                    self.milisi_perplexity = 5
                else:
                    self.milisi_perplexity = 30
                self._run_milisi(
                    sample_id,
                    n_jobs,
                    split,
                    emb_bal,
                    emb_noda_bal,
                    y_dis_bal,
                )

            # rf50
            print("rf50", end=" ")
            if self.pretraining:
                embs = (emb, emb_noda)
            else:
                embs = (emb,)
            split_data = model_selection.train_test_split(
                y_dis,
                *embs,
                test_size=0.2,
                random_state=rs,
                stratify=y_dis,
            )
            logger.debug(f"split shapes: {[x.shape for x in split_data]}")

            logger.debug("rf50 da")
            y_dis_train, y_dis_test = split_data[:2]
            emb_train, emb_test = split_data[2:4]
            self.rf50_d["da"][split][sample_id] = self.rf50_score(
                emb_train, emb_test, y_dis_train, y_dis_test
            )
            if self.pretraining:
                logger.debug("rf50 noda")
                emb_noda_train, emb_noda_test = split_data[4:]
                self.rf50_d["noda"][split][sample_id] = self.rf50_score(
                    emb_noda_train, emb_noda_test, y_dis_train, y_dis_test
                )

            print("|", end=" ")
            # newline at end of split

        print("")

    def _run_milisi(self, sample_id, n_jobs, split, emb, emb_noda, y_dis):
        logger.debug(f"Using {n_jobs} jobs with parallel backend \"{'threading'}\"")

        print(" milisi", end=" ")
        meta_df = pd.DataFrame(y_dis, columns=["Domain"])
        score = self._milisi_parallel(n_jobs, emb, meta_df)

        self.miLISI_d["da"][split][sample_id] = np.median(score)
        logger.debug(f"miLISI da: {self.miLISI_d['da'][split][sample_id]}")
        if self.pretraining:
            score = self._milisi_parallel(n_jobs, emb_noda, meta_df)
            self.miLISI_d["noda"][split][sample_id] = np.median(score)
            logger.debug(f"miLISI noda: {self.miLISI_d['da'][split][sample_id]}")

    def _milisi_parallel(self, n_jobs, emb, meta_df):
        if n_jobs > 1:
            with parallel_backend("threading", n_jobs=n_jobs):
                return hm.compute_lisi(
                    emb,
                    meta_df,
                    ["Domain"],
                    perplexity=self.milisi_perplexity,
                )

        return hm.compute_lisi(
            emb,
            meta_df,
            ["Domain"],
            perplexity=self.milisi_perplexity,
        )

    def _plot_cellfraction(self, visnum, adata, pred_sp, ax=None):
        """Plot predicted cell fraction for a given visnum"""
        try:
            cell_name = self.sc_sub_dict[visnum]
        except TypeError:
            cell_name = "Other"
        logger.debug(f"plotting cell fraction for {cell_name}")
        y_pred = pred_sp[:, visnum].squeeze()
        if y_pred.ndim > 1:
            y_pred = y_pred.sum(axis=1)
        adata.obs["Pred_label"] = y_pred

        sc.pl.spatial(
            adata,
            img_key="hires",
            color="Pred_label",
            palette="Set1",
            # size=1.5,
            legend_loc=None,
            title=cell_name,
            spot_size=1 if self.data_params.get("dset") == "pdac" else 150,
            show=False,
            ax=ax,
        )

    def _plot_roc(
        self,
        visnum,
        adata,
        pred_sp,
        name,
        num_name_exN_l,
        numlist,
        ax=None,
    ):
        """Plot ROC for a given visnum"""

        logging.debug(f"plotting ROC for {self.sc_sub_dict[visnum]} and {name}")
        Ex_l = [t[2] for t in num_name_exN_l]
        num_to_ex_d = dict(zip(numlist, Ex_l))

        def layer_to_layer_number(x):
            """Converts a string of layers to a list of layer numbers"""
            for char in x:
                if char.isdigit():
                    # if in (ordinal -> ex number -> layers)
                    if int(char) in Ex_to_L_d[num_to_ex_d[visnum]]:
                        return 1
            return 0

        y_pred = pred_sp[:, visnum]
        y_true = adata.obs["spatialLIBD"].map(layer_to_layer_number).fillna(0)
        # print(y_true)
        # print(y_true.isna().sum())
        RocCurveDisplay.from_predictions(y_true=y_true, y_pred=y_pred, name=name, ax=ax)

        return metrics.roc_auc_score(y_true, y_pred)

    def _plot_roc_pdac(self, visnum, adata, pred_sp, name, sc_to_st_celltype, ax=None):
        """Plot ROC for a given visnum (PDAC)"""
        try:
            cell_name = self.sc_sub_dict[visnum]
        except TypeError:
            cell_name = "Other"
        logging.debug(f"plotting ROC for {cell_name} and {name}")

        def st_sc_bin(cell_type):
            return int(cell_type in sc_to_st_celltype.get(cell_name, set()))

        y_pred = pred_sp[:, visnum].squeeze()
        if y_pred.ndim > 1:
            y_pred = y_pred.sum(axis=1)
        y_true = adata.obs["cell_type"].map(st_sc_bin).fillna(0)

        if y_true.sum() > 0:
            if ax is not None:
                RocCurveDisplay.from_predictions(y_true=y_true, y_pred=y_pred, name=name, ax=ax)
            return metrics.roc_auc_score(y_true, y_pred)

        return np.nan

    def _color_dict_from_layers(self, adata_st):
        """"""
        cmap = mpl.cm.get_cmap("Accent_r")

        color_range = list(
            np.linspace(
                0.125,
                1,
                len(adata_st.obs.spatialLIBD.cat.categories),
                endpoint=True,
            )
        )
        colors = [cmap(x) for x in color_range]

        color_dict = defaultdict(lambda: "lightgrey")
        for cat, color in zip(adata_st.obs.spatialLIBD.cat.categories, colors):
            color_dict[cat] = color

        color_dict["NA"] = "lightgrey"

        return color_dict

    def _plot_spatial(self, adata_st_d, color_dict, color="spatialLIBD", fname="layers.png"):
        splits, sids = self._get_splits_sids()
        fig, ax = plt.subplots(
            nrows=1,
            ncols=len(sids),
            figsize=(3 * len(sids), 3),
            squeeze=False,
            constrained_layout=True,
            dpi=50,
        )
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=_color,
                markerfacecolor=color_dict[_color],
                markersize=10,
            )
            for _color in color_dict
        ]
        fig.legend(
            bbox_to_anchor=(0, 0.5),
            handles=legend_elements,
            loc="center right",
        )

        i = 0
        for split in splits:
            for sample_id in self.st_sample_id_d[split]:
                sc.pl.spatial(
                    adata_st_d[sample_id],
                    img_key=None,
                    color=color,
                    palette=color_dict,
                    size=1,
                    title=f"{split}: {sample_id}" if split else sample_id,
                    legend_loc=4,
                    na_color="lightgrey",
                    spot_size=1 if self.data_params.get("dset") == "pdac" else 100,
                    show=False,
                    ax=ax[0][i],
                )
                _square_and_strip(ax[0][i])
                i += 1

        fig.savefig(
            os.path.join(self.results_folder, fname),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    def _plot_samples_pdac(self, sample_id, adata_st, pred_sp, pred_sp_noda=None, no_output=False):
        if not no_output:
            logging.debug(f"Plotting {sample_id}")

        if self.data_params.get("sc_id") == "CA001063":
            sc_to_st_celltype = {
                # Peng et al., 2019: Taken together, these results show that
                # type 2 ductal cells are the major source of malignant cells in
                # PDACs.
                "Ductal cell type 2": {"Cancer region"},
                "T cell": {"Cancer region", "Stroma"},
                "Macrophage cell": {"Cancer region", "Stroma"},
                "Fibroblast cell": {"Cancer region", "Stroma"},
                "B cell": {"Cancer region", "Stroma"},
                "Ductal cell type 1": {"Duct epithelium"},
                "Endothelial cell": {"Interstitium"},
                "Stellate cell": {"Stroma", "Pancreatic tissue"},
                "Acinar cell": {"Pancreatic tissue"},
                "Endocrine cell": {"Pancreatic tissue"},
            }
        else:
            sc_to_st_celltype = {
                # As expected, we found that all ductal subpopulations in PDAC-A
                # were enriched in the duct region of the tissue. In contrast,
                # only the hypoxic and terminal ductal cell populations were
                # significantly enriched in the cancer region
                "Ductal - MHC Class II": {"Duct epithelium"},
                "Ductal - CRISP3 high/centroacinar like": {"Duct epithelium"},
                "Ductal - terminal ductal like": {"Duct epithelium", "Cancer region"},
                "Ductal - APOL1 high/hypoxic": {"Duct epithelium", "Cancer region"},
                "Cancer clone": {"Cancer region"},
                "mDCs": {"Cancer region", "Stroma"},
                "Macrophages": {"Cancer region", "Stroma"},
                "T cells & NK cells": {"Cancer region", "Stroma"},
                "Tuft cells": {"Pancreatic tissue"},
                "Monocytes": {"Cancer region", "Stroma"},
                # "RBCs": 15,
                "Mast cells": {"Cancer region", "Stroma"},
                "Acinar cells": {"Pancreatic tissue"},
                "Endocrine cells": {"Pancreatic tissue"},
                "pDCs": {"Cancer region", "Stroma"},
                "Endothelial cells": {"Interstitium"},
            }

        celltypes = list(sc_to_st_celltype.keys()) + ["Other"]
        n_celltypes = len(celltypes)
        n_rows = int(math.ceil(n_celltypes / 5))

        numlist = [self.sc_sub_dict2.get(t) for t in celltypes[:-1]]
        numlist.extend([v for k, v in self.sc_sub_dict2.items() if k not in celltypes[:-1]])

        if not no_output:
            logging.debug(f"Plotting Cell Fractions")
            logging.debug(f"numlist: {numlist}")
            logging.debug(f"celltypes: {celltypes}")
            logging.debug(f"sc_sub_dict2: {self.sc_sub_dict2}")
            fig, ax = plt.subplots(
                n_rows, 5, figsize=(20, 4 * n_rows), constrained_layout=True, dpi=10
            )
            for i, num in enumerate(numlist):
                self._plot_cellfraction(num, adata_st, pred_sp, ax.flat[i])
                ax.flat[i].axis("equal")
                ax.flat[i].set_xlabel("")
                ax.flat[i].set_ylabel("")
            for i in range(n_celltypes, n_rows * 5):
                ax.flat[i].axis("off")
            fig.suptitle(sample_id)

            logging.debug(f"Saving Cell Fractions Figure")
            fig.savefig(
                os.path.join(self.results_folder, f"{sample_id}_cellfraction.png"),
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()

            logging.debug(f"Plotting ROC")

        n_rows = int(math.ceil(len(sc_to_st_celltype) / 5))
        fig, ax = plt.subplots(
            n_rows,
            5,
            figsize=(20, 4 * n_rows),
            constrained_layout=True,
            sharex=True,
            sharey=True,
            dpi=10,
        )

        da_aucs = []
        if self.pretraining and pred_sp_noda is not None:
            noda_aucs = []
        for i, num in enumerate(numlist[:-1]):
            ax_ = ax.flat[i] if not no_output else None
            da_aucs.append(
                self._plot_roc_pdac(
                    num,
                    adata_st,
                    pred_sp,
                    self.args_dict["modelname"],
                    sc_to_st_celltype,
                    ax_,
                )
            )
            if self.pretraining and pred_sp_noda is not None:
                noda_aucs.append(
                    self._plot_roc_pdac(
                        num,
                        adata_st,
                        pred_sp_noda,
                        f"{self.args_dict['modelname']}_wo_da",
                        sc_to_st_celltype,
                        ax_,
                    )
                )

            if not no_output:
                ax.flat[i].plot([0, 1], [0, 1], transform=ax.flat[i].transAxes, ls="--", color="k")
                ax.flat[i].set_aspect("equal")
                ax.flat[i].set_xlim([0, 1])
                ax.flat[i].set_ylim([0, 1])
                try:
                    cell_name = self.sc_sub_dict[num]
                except TypeError:
                    cell_name = "Other"
                ax.flat[i].set_title(cell_name)

                if i >= len(numlist) - 5:
                    ax.flat[i].set_xlabel("FPR")
                else:
                    ax.flat[i].set_xlabel("")
                if i % 5 == 0:
                    ax.flat[i].set_ylabel("TPR")
                else:
                    ax.flat[i].set_ylabel("")

        if not no_output:
            for i in range(len(numlist[:-1]), n_rows * 5):
                ax.flat[i].axis("off")
            fig.suptitle(sample_id)

        logging.debug(f"Saving ROC Figure")
        fig.savefig(
            os.path.join(self.results_folder, f"{sample_id}_roc.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

        return (
            np.nanmean(da_aucs),
            np.nanmean(noda_aucs) if self.pretraining and pred_sp_noda is not None else None,
        )

    def _plot_samples(self, sample_id, adata_st_d, pred_sp_d, pred_sp_noda_d=None, no_output=False):
        if not no_output:
            logging.debug(f"Plotting {sample_id}")
            fig, ax = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True, dpi=10)

        num_name_exN_l = []
        for k, v in self.sc_sub_dict.items():
            if "Ex" in v:
                # (clust_ordinal, clust_name, Ex_clust_num)
                num_name_exN_l.append((k, v, int(v.split("_")[1])))

        num_name_exN_l.sort(key=lambda a: a[2])

        numlist = [t[0] for t in num_name_exN_l]  # clust ordinals

        if not no_output:
            logging.debug(f"Plotting Cell Fractions")
            for i, num in enumerate(numlist):
                self._plot_cellfraction(
                    num, adata_st_d[sample_id], pred_sp_d[sample_id], ax.flat[i]
                )
                ax.flat[i].axis("equal")
                ax.flat[i].set_xlabel("")
                ax.flat[i].set_ylabel("")
            fig.suptitle(sample_id)

            logging.debug(f"Saving Cell Fractions Figure")
            fig.savefig(
                os.path.join(self.results_folder, f"{sample_id}_cellfraction.png"),
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()

            logging.debug(f"Plotting ROC")
            fig, ax = plt.subplots(
                2,
                5,
                figsize=(20, 8),
                constrained_layout=True,
                sharex=True,
                sharey=True,
                dpi=10,
            )

        da_aucs = []
        if self.pretraining and pred_sp_noda_d is not None:
            noda_aucs = []
        for i, num in enumerate(numlist):
            ax_ = ax.flat[i] if not no_output else None
            da_aucs.append(
                self._plot_roc(
                    num,
                    adata_st_d[sample_id],
                    pred_sp_d[sample_id],
                    self.args_dict["modelname"],
                    num_name_exN_l,
                    numlist,
                    ax_,
                )
            )
            if self.pretraining and pred_sp_noda_d is not None:
                noda_aucs.append(
                    self._plot_roc(
                        num,
                        adata_st_d[sample_id],
                        pred_sp_noda_d[sample_id],
                        f"{self.args_dict['modelname']}_wo_da",
                        num_name_exN_l,
                        numlist,
                        ax_,
                    )
                )

            if not no_output:
                ax.flat[i].plot(
                    [0, 1],
                    [0, 1],
                    transform=ax.flat[i].transAxes,
                    ls="--",
                    color="k",
                )
                ax.flat[i].set_aspect("equal")
                ax.flat[i].set_xlim([0, 1])
                ax.flat[i].set_ylim([0, 1])

                ax.flat[i].set_title(f"{self.sc_sub_dict[num]}")

                if i >= len(numlist) - 5:
                    ax.flat[i].set_xlabel("FPR")
                else:
                    ax.flat[i].set_xlabel("")
                if i % 5 == 0:
                    ax.flat[i].set_ylabel("TPR")
                else:
                    ax.flat[i].set_ylabel("")

        if not no_output:
            fig.suptitle(sample_id)
            logging.debug(f"Saving ROC Figure")
            fig.savefig(
                os.path.join(self.results_folder, f"{sample_id}_roc.png"),
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()

        return (
            np.nanmean(da_aucs),
            np.nanmean(noda_aucs) if self.pretraining and pred_sp_noda_d is not None else None,
        )

    def _plot_ax_scatterpie(self, xs, ys, dists, **kwargs):
        for x, y, dist in zip(xs, ys, dists):
            evaluation.draw_pie(x, y, dist, **kwargs)

    def _plot_spatial_scatterpie(
        self,
        adata_st_d,
        pred_sp_d,
        pred_sp_noda_d=None,
        hue="relative_spot_composition",
        fname="st_cell_types.png",
        no_output=False,
    ):
        if (
            self.data_params.get("sc_id") == "GSE115746"
            and self.data_params.get("st_id") == "spotless_mouse_cortex"
        ):
            st_sub_map = get_st_sub_map()
            cell_type_index = sorted(list(st_sub_map.keys())) + ["Other"]

            # create a mapping from spotless cell types to sc cell types
            merged_to_sc = {k: [] for k in cell_type_index}

            for k, v in itertools.chain(
                cell_cluster_cell_type_to_spot_composition.items(),
                cell_subclass_to_spot_composition.items(),
            ):
                if k != "keep_the_rest":
                    if len(v) > 0:
                        merged_to_sc["/".join(sorted(list(v)))].append(k)
                    else:
                        merged_to_sc["Other"].append(k)

            new_pred_sp_d = self._merge_sc_preds(pred_sp_d, cell_type_index, merged_to_sc)
            new_pred_sp_noda_d = (
                self._merge_sc_preds(pred_sp_noda_d, cell_type_index, merged_to_sc)
                if pred_sp_noda_d is not None
                else None
            )
        else:
            cell_type_index = [self.sc_sub_dict[i] for i in range(len(self.sc_sub_dict))]
            new_pred_sp_d = pred_sp_d
            new_pred_sp_noda_d = pred_sp_noda_d

        # set up plot
        if not no_output:
            splits, sids = self._get_splits_sids()

            # get colour codes
            cmap = mpl.cm.get_cmap("viridis")
            color_range = list(
                np.linspace(
                    0.125,
                    1,
                    len(cell_type_index),
                    endpoint=True,
                )
            )
            colors = [cmap(x) for x in color_range]

            color_dict = {}
            for cat, color in zip(cell_type_index, colors):
                color_dict[cat] = color

            # create figure
            nrows = 2 if new_pred_sp_noda_d is None else 3
            fig = plt.figure(
                figsize=(3 * len(sids), 3 * nrows),
                constrained_layout=True,
                dpi=50,
            )
            subfigs = fig.subfigures(nrows=nrows, ncols=1)
            axs = [subfig.subplots(nrows=1, ncols=len(sids)) for subfig in subfigs]

            legend_elements = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=cell_type,
                    markerfacecolor=color_dict[cell_type],
                    markersize=10,
                )
                for cell_type in cell_type_index
            ]
            fig.legend(
                bbox_to_anchor=(0, 0.5),
                handles=legend_elements,
                loc="center right",
            )

            subfigs[0].suptitle("Ground Truth")
            if new_pred_sp_noda_d is not None:
                subfigs[1].suptitle(f"{self.args_dict['modelname']}_wo_da")
            subfigs[-1].suptitle(self.args_dict["modelname"])

            colors = [color_dict[name] for name in cell_type_index]
        else:
            splits = ["train"]
            sids = self.st_sample_id_d["train"]

        st_cell_types_to_sc = {re.sub("( |\/)", ".", name): name for name in cell_type_index}

        ctps = OrderedDict([(sid, [None, None]) for sid in sids if sid in new_pred_sp_d])

        i = 0
        for split in splits:
            for sample_id in self.st_sample_id_d[split]:
                if sample_id not in new_pred_sp_d:
                    continue
                dists_true = (
                    adata_st_d[sample_id]
                    .obsm[hue]
                    .rename(columns=st_cell_types_to_sc)
                    # this adds an "Other" column with all 0s
                    .reindex(columns=cell_type_index, fill_value=0.0)
                    .to_numpy()
                )
                ctps[sample_id][0] = self.metric_ctp(new_pred_sp_d[sample_id], dists_true)
                if new_pred_sp_noda_d is not None:
                    ctps[sample_id][1] = self.metric_ctp(new_pred_sp_noda_d[sample_id], dists_true)

                # plot
                if not no_output:
                    sp_kws = dict(
                        xs=adata_st_d[sample_id].obs["X"].to_numpy(),
                        ys=adata_st_d[sample_id].obs["Y"].to_numpy(),
                        colors=colors,
                        s=1000,
                    )

                    self._plot_ax_scatterpie(dists=dists_true, ax=axs[0][i], **sp_kws)
                    axs[0][i].set_title(f"{split}: {sample_id}" if split else sample_id)
                    _square_and_strip(axs[0][i])

                    if new_pred_sp_noda_d is not None:
                        self._plot_ax_scatterpie(
                            dists=new_pred_sp_noda_d[sample_id], ax=axs[1][i], **sp_kws
                        )

                        ctps[sample_id][1] = self.metric_ctp(
                            new_pred_sp_noda_d[sample_id], dists_true
                        )

                        axs[1][i].set_title(f"JSD: {ctps[sample_id][1]}")
                        _square_and_strip(axs[1][i])

                    self._plot_ax_scatterpie(
                        dists=new_pred_sp_d[sample_id], ax=axs[-1][i], **sp_kws
                    )

                    axs[-1][i].set_title(f"JSD: {ctps[sample_id][0]}")
                    _square_and_strip(axs[-1][i])

                i += 1

        if not no_output:
            fig.savefig(
                os.path.join(self.results_folder, fname),
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()

        return [ctps[sid] for sid in sids if sid in ctps]

    def _merge_sc_preds(self, pred_sp_d, cell_type_index, merged_to_sc):
        new_pred_dict = {}
        for sid, pred in pred_sp_d.items():
            new_pred_dict[sid] = np.empty((pred.shape[0], len(cell_type_index)), dtype=pred.dtype)
            for i, merged_cell_type in enumerate(cell_type_index):
                if merged_cell_type in merged_to_sc:
                    old_idxs = [
                        self.sc_sub_dict2[cell_type] for cell_type in merged_to_sc[merged_cell_type]
                    ]
                    new_pred_dict[sid][:, i] = pred[:, old_idxs].sum(axis=1)
                else:
                    # no sc cell types map to this st cell type
                    new_pred_dict[sid][:, i] = 0

        return new_pred_dict

    # %%
    def eval_spots(self):
        """Evaluates spots."""

        _, sids = self._get_splits_sids()

        # %%
        print("Loading ST adata: ")

        adata_st = sc.read_h5ad(os.path.join(self.selected_dir, "st.h5ad"))

        adata_st_d = {}

        # Get samples and coordinates
        for sid in sids:
            adata_st_d[sid] = adata_st[adata_st.obs.sample_id == sid]
            adata_st_d[sid].obsm["spatial"] = adata_st_d[sid].obs[["X", "Y"]].values

        print("Getting predictions: ")
        if self.data_params.get("samp_split", False):
            path = self.samp_split_folder
        elif self.data_params.get("one_model", False):
            path = self.advtrain_folder
        else:
            path = None

        if self.args_dict.get("early_stopping") and self.data_params.get("samp_split"):
            val_sids = self.st_sample_id_d["train"]

            if self.temp_folder_holder.is_temp():
                new_path = os.path.join(self.results_folder, "curr_models")
                os.makedirs(new_path, exist_ok=True)
                for name in glob.glob(os.path.join(path, "*")):
                    if not name.endswith("tar.gz"):
                        shutil.copy(name, new_path)
            else:
                new_path = path

            with tarfile.open(os.path.join(path, "checkpts.tar.gz"), "r:gz") as tar:
                tar.extractall(new_path)

            self.samp_split_folder = path = new_path

            pred_sp_chkpt_d = {}
            for name in glob.glob(os.path.join(path, "checkpt-*")):
                model_fname = os.path.basename(name).split(".")[0]
                epoch = int(model_fname[len("checkpt-") :])  # strip "checkpt-"
                epoch_model = ModelWrapper(path, name=model_fname)
                pred_sp_chkpt_d[epoch] = {
                    sid: epoch_model.get_predictions(self.mat_sp_d[sid]) for sid in val_sids
                }

            # early stopping using train as validation set
            epochs = sorted(list(pred_sp_chkpt_d.keys()))

            if "spotless" in self.data_params.get("st_id", set()):
                n_jobs_samples = min(
                    effective_n_jobs(int(self.args_dict["njobs"])),
                    len(epochs),
                )

                logging.debug(f"n_jobs_samples: {n_jobs_samples}")

                aucs = Parallel(n_jobs=n_jobs_samples, verbose=100)(
                    delayed(self._plot_spatial_scatterpie)(
                        adata_st_d,
                        pred_sp_chkpt_d[epoch],
                        hue="relative_spot_composition",
                        no_output=True,
                    )
                    for epoch in epochs
                )
                aucs_df = pd.DataFrame(aucs, columns=val_sids, index=epochs)
                aucs_df = aucs_df.applymap(lambda x: x[0])

                # min because divergence metric
                best_epoch = aucs_df.mean(axis=1).idxmin()
                # print(aucs_df)
                # print(aucs_df.mean(axis=1))
            else:
                epochs_sids = list(itertools.product(epochs, val_sids))
                n_jobs_samples = min(
                    effective_n_jobs(int(self.args_dict["njobs"])),
                    len(epochs_sids),
                )

                logging.debug(f"n_jobs_samples: {n_jobs_samples}")

                if self.data_params.get("dset") == "pdac":
                    aucs = Parallel(n_jobs=n_jobs_samples, verbose=100)(
                        delayed(self._plot_samples_pdac)(
                            sid,
                            adata_st_d[sid],
                            pred_sp_chkpt_d[epoch][sid],
                            no_output=True,
                        )
                        for epoch, sid in epochs_sids
                    )

                else:
                    aucs = Parallel(n_jobs=n_jobs_samples, verbose=100)(
                        delayed(self._plot_samples)(
                            sid,
                            adata_st_d,
                            pred_sp_chkpt_d[epoch],
                            no_output=True,
                        )
                        for epoch, sid in epochs_sids
                    )

                aucs_df = pd.DataFrame.from_records(epochs_sids, columns=["epoch", "sample_id"])
                aucs_df["aucs"] = aucs
                aucs_df["aucs"] = aucs_df["aucs"].map(lambda x: x[0])

                aucs_df = aucs_df.pivot(index="epoch", columns="sample_id", values="aucs")

                # max because auc
                best_epoch = aucs_df.mean(axis=1).idxmax()

            print(f"Best epoch: {best_epoch}")

            self.model_fname = f"checkpt-{best_epoch}"

        if path is not None:
            inputs = [self.mat_sp_d[sid] for sid in sids]
            outputs = iter_preds(inputs, path, name=self.model_fname)
            pred_sp_d = dict(zip(sids, outputs))

        else:
            pred_sp_d = {}
            for sample_id in sids:
                path = os.path.join(self.advtrain_folder, sample_id)
                input = self.mat_sp_d[sample_id]
                pred_sp_d[sample_id] = ModelWrapper(path, name=self.model_fname).get_predictions(
                    input
                )

        if self.pretraining:
            inputs = [self.mat_sp_d[sid] for sid in sids]
            outputs = iter_preds(
                inputs, self.pretrain_folder, name="final_model", source_encoder=True
            )
            pred_sp_noda_d = dict(zip(sids, outputs))
        else:
            pred_sp_noda_d = None

        realspots_d = {"da": {}}
        if self.pretraining:
            realspots_d["noda"] = {}

        if self.data_params.get("dset") == "pdac":
            aucs = self.eval_pdac_spots(pred_sp_d, pred_sp_noda_d, adata_st_d)
        elif "spotless" in self.data_params.get("st_id", set()):
            aucs = self.eval_spotless_gs_spots(pred_sp_d, pred_sp_noda_d, adata_st_d)
        else:
            aucs = self.eval_dlpfc_spots(pred_sp_d, pred_sp_noda_d, adata_st, adata_st_d)

        for sample_id, auc in zip(sids, aucs):
            realspots_d["da"][sample_id] = auc[0]
            if self.pretraining:
                realspots_d["noda"][sample_id] = auc[1]
        self.realspots_d = realspots_d

    def eval_dlpfc_spots(self, pred_sp_d, pred_sp_noda_d, adata_st, adata_st_d):
        _, sids = self._get_splits_sids()

        color_dict = self._color_dict_from_layers(adata_st)

        self._plot_spatial(adata_st_d, color_dict, color="spatialLIBD", fname="layers.png")

        print("Plotting Samples")
        n_jobs_samples = min(effective_n_jobs(int(self.args_dict["njobs"])), len(sids))
        logging.debug(f"n_jobs_samples: {n_jobs_samples}")
        aucs = Parallel(n_jobs=n_jobs_samples, verbose=100)(
            delayed(self._plot_samples)(sid, adata_st_d, pred_sp_d, pred_sp_noda_d) for sid in sids
        )
        return aucs

    def eval_pdac_spots(self, pred_sp_d, pred_sp_noda_d, adata_st_d):
        _, sids = self._get_splits_sids()

        ## get colour codes
        raw_pdac_dir = os.path.join(self.data_params["data_dir"], "pdac", "st_adata")
        ctr_fname = f"{self.data_params['st_id']}-cluster_to_rgb.pkl"
        with open(os.path.join(raw_pdac_dir, ctr_fname), "rb") as f:
            cluster_to_rgb = pickle.load(f)

        self._plot_spatial(adata_st_d, cluster_to_rgb, color="cell_type", fname="st_cell_types.png")

        print("Plotting Samples")
        n_jobs_samples = min(effective_n_jobs(int(self.args_dict["njobs"])), len(sids))
        logging.debug(f"n_jobs_samples: {n_jobs_samples}")
        if n_jobs_samples < 4:
            print("n_jobs_samples < 4, no parallelization")
            aucs = [
                self._plot_samples_pdac(
                    sid,
                    adata_st_d[sid],
                    pred_sp_d[sid],
                    pred_sp_noda_d[sid] if pred_sp_noda_d else None,
                )
                for sid in sids
            ]
        else:
            aucs = Parallel(n_jobs=n_jobs_samples, verbose=100)(
                delayed(self._plot_samples_pdac)(
                    sid,
                    adata_st_d[sid],
                    pred_sp_d[sid],
                    pred_sp_noda_d[sid] if pred_sp_noda_d else None,
                )
                for sid in sids
            )
        return aucs

    def eval_spotless_gs_spots(self, pred_sp_d, pred_sp_noda_d, adata_st_d):
        ctps = self._plot_spatial_scatterpie(
            adata_st_d,
            pred_sp_d,
            pred_sp_noda_d,
            hue="relative_spot_composition",
            fname="st_cell_types.png",
        )

        return ctps

    def eval_sc(self):
        self.jsd_d = {"da": {}}
        if self.pretraining:
            self.jsd_d["noda"] = {}

        for k in self.jsd_d:
            self.jsd_d[k] = {split: {} for split in self.splits}

        _, sids = self._get_splits_sids()

        if self.pretraining:
            model = ModelWrapper(self.pretrain_folder, name="final_model")

            self._calc_jsd(
                sids[0],
                model=model,
                da="noda",
            )
            for split in self.splits:
                score = self.jsd_d["noda"][split][sids[0]]
                for sample_id in sids[1:]:
                    self.jsd_d["noda"][split][sample_id] = score

        if self.data_params.get("samp_split", False):
            model = ModelWrapper(self.samp_split_folder, name=self.model_fname)
        elif self.data_params.get("one_model", False):
            model = ModelWrapper(self.advtrain_folder, name=self.model_fname)
        else:
            model = None

        for sample_id in sids:
            if model is None:
                model_samp = ModelWrapper(
                    os.path.join(self.advtrain_folder, sample_id), name=self.model_fname
                )
                self._calc_jsd(sample_id, model=model_samp, da="da")
            else:
                self._calc_jsd(sample_id, model=model, da="da")

    def _calc_jsd(self, sample_id, model=None, da="da"):
        for split in self.splits:
            input = self.sc_mix_d[split]
            pred = model.get_predictions(input, source_encoder=True)
            score = self.metric_ctp(pred, self.lab_mix_d[split])

            self.jsd_d[da][split][sample_id] = score

    def gen_l_dfs(self, da):
        """Generate a list of series for a given da"""
        df = pd.DataFrame.from_dict(self.jsd_d[da], orient="columns")
        df.columns.name = "SC Split"
        yield df
        df = pd.DataFrame.from_dict(self.rf50_d[da], orient="columns")
        df.columns.name = "SC Split"
        yield df
        if self.args_dict["milisi"]:
            df = pd.DataFrame.from_dict(self.miLISI_d[da], orient="columns")
            df.columns.name = "SC Split"
            yield df
        yield pd.Series(self.realspots_d[da])
        return

    def produce_results(self):
        if self.args_dict.get("early_stopping", False) and self.data_params.get(
            "samp_split", False
        ):
            print("Cleaning up ... ")
            if self.temp_folder_holder.is_temp():
                # clean up temp folder before copying back
                if os.path.basename(os.path.norm(self.samp_split_folder)) == "curr_models":
                    shutil.rmtree(self.samp_split_folder)
                else:
                    warnings.warn("temp folder not found, skipping clean up")
            else:
                # clean up local folder
                for name in glob.glob(os.path.join(self.samp_split_folder, "checkpt-*.pth")):
                    os.remove(name)

        if self.data_params.get("dset") == "pdac":
            real_spot_header = "Real Spots (Mean AUC celltype)"
        elif "spotless" in self.data_params.get("st_id", set()):
            real_spot_header = "Real Spots (Cosine Distance)"
        else:
            real_spot_header = "Real Spots (Mean AUC Ex1-10)"

        df_keys = [
            "Pseudospots (Cosine Distance)",
            "RF50",
            real_spot_header,
        ]

        if self.args_dict["milisi"]:
            df_keys.insert(2, f"miLISI (perplexity={self.milisi_perplexity})")

        da_dict_keys = ["da"]
        if self.args_dict.get("early_stopping", False):
            epoch = int(self.model_fname[len("checkpt-") :])  # strip "checkpt-"
            da_df_keys = [f"After DA (epoch {epoch})"]
        else:
            da_df_keys = ["After DA (final model)"]

        if self.pretraining:
            da_dict_keys.insert(0, "noda")
            da_df_keys.insert(0, "Before DA")

        results_df = [
            pd.concat(list(self.gen_l_dfs(da)), axis=1, keys=df_keys) for da in da_dict_keys
        ]

        sid_to_split = {sid: split for split, sids_ in self.st_sample_id_d.items() for sid in sids_}
        for df in results_df:
            df.index = df.index.map(lambda x: (sid_to_split[x], x))
            df.index.set_names(["SC Split", "Sample ID"], inplace=True)

        results_df = pd.concat(results_df, axis=0, keys=da_df_keys)
        if self.args_dict.get("early_stopping", False):
            results_fname = f"results_{self.model_fname}.csv"
        else:
            results_fname = "results.csv"
        results_df.to_csv(os.path.join(self.results_folder, results_fname))
        print(results_df)

        with open(os.path.join(self.results_folder, "config.yml"), "w") as f:
            yaml.dump(self.config, f)

        self.temp_folder_holder.copy_out()

    def _get_splits_sids(self):
        if "" in self.st_sample_id_d:
            splits = [""]
        else:
            splits = [split for split in self.splits if split in self.st_sample_id_d]

        sids = [sid for split in splits for sid in self.st_sample_id_d[split]]
        return splits, sids


def iter_preds(inputs, model_dir, name, source_encoder=False):
    model = ModelWrapper(model_dir, name)
    inputs_iter = listify(inputs)

    for input in inputs_iter:
        yield model.get_predictions(input, source_encoder=source_encoder)


def iter_embs(inputs, model_dir, name, source_encoder=False):
    model = ModelWrapper(model_dir, name)
    inputs_iter = listify(inputs)
    logger.debug(f"Embeddings input length: {len(inputs_iter)}")

    for input in inputs_iter:
        yield model.get_embeddings(input, source_encoder=source_encoder)


def listify(inputs):
    try:
        inputs.shape
    except AttributeError:
        inputs_iter = inputs
    else:
        logger.debug(f"Input is single array with shape {inputs.shape}")
        inputs_iter = [inputs]
    return inputs_iter


def fit_pca(X, *args, **kwargs):
    return PCA(*args, **kwargs).fit(X)


def _square_and_strip(ax):
    ax.axis("equal")
    ax.set_xlabel("")
    ax.set_ylabel("")


# %%
