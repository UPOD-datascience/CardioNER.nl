import csv
import logging
from typing import Dict
from typing import Tuple
import re 
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Constants
START_SPAN_TAG = "start_span"
END_SPAN_TAG = "end_span"
ENTITY_NAME_TAG = "text"
LABEL_TAG = "label"
FILE_NAME = "filename"


def parse_tsv_file(datapath: str, entities_to_evaluate: list) -> pd.DataFrame:
    """
    Parse a TSV file into a DataFrame and perform basic formatting and deduplication.

    Parameters:
    -----------
    datapath : str
        Path to the TSV file.
    entities_to_evaluate: list
        List of entities to evaluate. If none, take all entities

    Returns:
    --------
    pd.DataFrame
        Formatted and deduplicated DataFrame.
    """
    try:
        # Load the TSV file
        df = pd.read_csv(datapath, sep='\t', header=0, quoting=csv.QUOTE_NONE, keep_default_na=False, dtype=str)

        if entities_to_evaluate:
            df = df.loc[df[LABEL_TAG].isin(entities_to_evaluate), :].copy()

        # Format DataFrame
        df['offset'] = df[START_SPAN_TAG].astype(str) + ' ' + df[END_SPAN_TAG].astype(str)
        df = df[~df[LABEL_TAG].isna() & (df[LABEL_TAG].str.strip() != "")]
        df[LABEL_TAG] = df[LABEL_TAG].str.upper()

        # Check for duplicated entries
        if df.duplicated(subset=[FILE_NAME, LABEL_TAG, 'offset']).any():
            df = df.drop_duplicates(subset=[FILE_NAME, LABEL_TAG, 'offset']).copy()
            logger.warning("Duplicated entries found and removed.")

        return df

    except Exception as e:
        logger.error(f"Error parsing TSV file: {e}")
        raise


def calculate_metrics_strict(gs: pd.DataFrame, pred: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], Dict[str, float]]:
    """
    Calculates strict matching metrics (exact span and label match) including precision, recall, and F1-score
    for each label, along with micro and macro-averaged scores.

    This function assumes the input dataframes contain an `offset` field, a `label` field, and a `filename` column.
    Strict evaluation considers a prediction correct only if the filename, offset, and label all match.

    Args:
        gs (pd.DataFrame): Ground truth mentions with columns ['filename', 'offset', 'label'].
        pred (pd.DataFrame): Predicted mentions with the same required columns.

    Returns:
        Tuple containing:
            - Dict[str, Dict[str, float]]: Per-label metrics with precision, recall, and F1-score.
            - Dict[str, float]: Micro-averaged precision, recall, and F1-score.
            - Dict[str, float]: Macro-averaged precision, recall, and F1-score.
    """

    #gs = gs.drop_duplicates(subset=[FILE_NAME, "offset", LABEL_TAG])
    #pred = pred.drop_duplicates(subset=[FILE_NAME, "offset", LABEL_TAG])

    labels = sorted(set(gs[LABEL_TAG].unique()) | set(pred[LABEL_TAG].unique()))
    result_by_cat = {}

    total_tp = total_fp = total_fn = 0
    precision_list = []
    recall_list = []
    f1_list = []

    for label in labels:
        gs_label = gs[gs[LABEL_TAG] == label]
        pred_label = pred[pred[LABEL_TAG] == label]

        GS_Pos = gs_label.shape[0]
        Pred_Pos = pred_label.shape[0]

        merged = pd.merge(pred_label, gs_label, how="inner", on=[FILE_NAME, START_SPAN_TAG, END_SPAN_TAG, LABEL_TAG])
        TP = merged.shape[0]
        FP = Pred_Pos - TP
        FN = GS_Pos - TP

        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        result_by_cat[label] = {
            "Precision": round(precision, 2),
            "Recall": round(recall, 2),
            "F1": round(f1, 2),
        }

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        total_tp += TP
        total_fp += FP
        total_fn += FN

    # Micro-averaged scores
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) else 0.0

    # Macro-averaged scores
    macro_precision = sum(precision_list) / len(precision_list) if precision_list else 0.0
    macro_recall = sum(recall_list) / len(recall_list) if recall_list else 0.0
    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0

    summary_micro = {
        "Precision": round(micro_precision, 2),
        "Recall": round(micro_recall, 2),
        "F1": round(micro_f1, 2),
    }

    summary_macro = {
        "Precision": round(macro_precision, 2),
        "Recall": round(macro_recall, 2),
        "F1": round(macro_f1, 2),
    }

    return result_by_cat, summary_micro, summary_macro


def calculate_metrics_relaxed(gs: pd.DataFrame, pred: pd.DataFrame) -> (
        Tuple)[Dict[str, Dict[str, float]], Dict[str, float], Dict[str, float]]:
    """
    Compute relaxed precision, recall, and F1-score for entity recognition.

    This function compares predicted and ground truth entity spans using a relaxed (interval overlap) strategy,
    where a match is valid if the spans overlap and the labels match. Each prediction can match at most one gold entity.

    Args:
        gs (pd.DataFrame): Ground truth mentions. Must contain columns ['filename', 'start_span', 'end_span', 'label'].
        pred (pd.DataFrame): Predicted mentions with the same column structure.

    Returns:
        Tuple containing:
            - result_by_cat (Dict[str, Dict[str, float]]): Per-label scores with precision, recall, and F1.
            - summary_micro (Dict[str, float]): Micro-averaged precision, recall, and F1.
            - summary_macro (Dict[str, float]): Macro-averaged precision, recall, and F1.

    Notes:
        - Intervals with missing 'start_span' or 'end_span' are ignored.
        - Micro scores are calculated by summing true positives, false positives, and false negatives.
        - Macro scores are calculated by averaging the per-label scores.
        - All scores are rounded to 4 decimal places.
    """

    # Clean and prepare intervals
    for df in [gs, pred]:
        df["start_span"] = pd.to_numeric(df["start_span"], errors="coerce")
        df["end_span"] = pd.to_numeric(df["end_span"], errors="coerce")

    gs_mentions = gs.dropna(subset=["start_span", "end_span"]).copy()
    preds_mentions = pred.dropna(subset=["start_span", "end_span"]).copy()

    gs_mentions["interval"] = pd.arrays.IntervalArray.from_arrays(
        gs_mentions["start_span"], gs_mentions["end_span"], closed="both"
    )
    preds_mentions["interval"] = pd.arrays.IntervalArray.from_arrays(
        preds_mentions["start_span"], preds_mentions["end_span"], closed="both"
    )

    labels = sorted(gs_mentions["label"].unique())
    result_by_cat = {}

    relaxed_TP_total = relaxed_FP_total = relaxed_FN_total = 0
    precision_list = []
    recall_list = []
    f1_list = []

    gs_grouped = gs_mentions.groupby(["label", "filename"])
    preds_grouped = preds_mentions.groupby(["label", "filename"])

    for label in labels:
        tp = fp = fn = 0
        filenames = set(gs_mentions[gs_mentions["label"] == label]["filename"]) | \
                    set(preds_mentions[preds_mentions["label"] == label]["filename"])

        for filename in filenames:
            gs_filtered = gs_grouped.get_group((label, filename)) if (label,
                                                                      filename) in gs_grouped.groups else pd.DataFrame()
            preds_filtered = preds_grouped.get_group((label, filename)) if (label,
                                                                            filename) in preds_grouped.groups else pd.DataFrame()

            gs_rows = list(gs_filtered.itertuples())
            pred_rows = list(preds_filtered.itertuples())

            gs_used = set()
            preds_used = set()

            for gs_idx, gs_row in enumerate(gs_rows):
                for pred_idx, pred_row in enumerate(pred_rows):
                    if pred_idx in preds_used:
                        continue
                    if gs_row.interval.overlaps(pred_row.interval):
                        tp += 1
                        gs_used.add(gs_idx)
                        preds_used.add(pred_idx)
                        break

            fp += len(pred_rows) - len(preds_used)
            fn += len(gs_rows) - len(gs_used)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        result_by_cat[label] = {
            "Precision": round(precision, 2),
            "Recall": round(recall, 2),
            "F1": round(f1, 2),
        }

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        relaxed_TP_total += tp
        relaxed_FP_total += fp
        relaxed_FN_total += fn

    # Micro scores
    micro_precision = relaxed_TP_total / (relaxed_TP_total + relaxed_FP_total) if (
                relaxed_TP_total + relaxed_FP_total) else 0.0
    micro_recall = relaxed_TP_total / (relaxed_TP_total + relaxed_FN_total) if (
                relaxed_TP_total + relaxed_FN_total) else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (
                micro_precision + micro_recall) else 0.0

    # Macro scores
    macro_precision = sum(precision_list) / len(precision_list) if precision_list else 0.0
    macro_recall = sum(recall_list) / len(recall_list) if recall_list else 0.0
    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0

    summary_micro = {
        "Precision": round(micro_precision, 2),
        "Recall": round(micro_recall, 2),
        "F1": round(micro_f1, 2)
    }

    summary_macro = {
        "Precision": round(macro_precision, 2),
        "Recall": round(macro_recall, 2),
        "F1": round(macro_f1, 2)
    }

    return result_by_cat, summary_micro, summary_macro
