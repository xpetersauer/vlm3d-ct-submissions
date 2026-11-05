#!/usr/bin/env python3
"""Metric-alignment CLI for VLM3D Tasks 1-3.

Every command writes machine-readable evidence and prints the same evidence in
plain language.  Exact challenge metrics are never conflated with surrogates.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np

from vlm3d_reward_model import fit_ridge


PATHOLOGIES = {
    "medical material": ("catheter", "line", "tube", "pacemaker", "stent", "prosthe"),
    "arterial wall calcification": ("aortic calc", "arterial calc", "atherosclero"),
    "cardiomegaly": ("cardiomegaly", "enlarged heart", "cardiac enlargement"),
    "pericardial effusion": ("pericardial effusion",),
    "coronary artery wall calcification": ("coronary calc",),
    "hiatal hernia": ("hiatal hernia",),
    "lymphadenopathy": ("lymphadenopathy", "adenopathy", "enlarged lymph node"),
    "emphysema": ("emphysema", "emphysematous"),
    "atelectasis": ("atelectasis", "atelectatic"),
    "lung nodule": ("pulmonary nodule", "lung nodule", "nodular opacity"),
    "lung opacity": ("lung opacity", "pulmonary opacity", "ground glass", "ground-glass"),
    "pulmonary fibrotic sequela": ("fibrosis", "fibrotic", "scarring"),
    "pleural effusion": ("pleural effusion",),
    "mosaic attenuation pattern": ("mosaic attenuation",),
    "peribronchial thickening": ("peribronchial thick", "bronchial wall thick"),
    "consolidation": ("consolidation", "airspace disease", "pneumonia"),
    "bronchiectasis": ("bronchiectasis",),
    "interlobular septal thickening": ("interlobular septal thick", "septal thick"),
}
NEGATIONS = ("no ", "without ", "absent ", "negative for ", "not seen", "not identified")


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def sigmoid(value):
    value = np.asarray(value, dtype=np.float64)
    return np.where(value >= 0, 1.0 / (1.0 + np.exp(-value)), np.exp(value) / (1.0 + np.exp(value)))


def logit(value):
    value = np.clip(np.asarray(value, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(value / (1.0 - value))


def binary_f1(truth, prediction) -> float:
    truth = np.asarray(truth, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    tp = np.logical_and(truth, prediction).sum()
    fp = np.logical_and(~truth, prediction).sum()
    fn = np.logical_and(truth, ~prediction).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def binary_auc(truth, probability) -> float:
    truth = np.asarray(truth, dtype=bool)
    probability = np.asarray(probability, dtype=np.float64)
    positive = probability[truth]
    negative = probability[~truth]
    if not len(positive) or not len(negative):
        return float("nan")
    # Pairwise definition handles ties and avoids a sklearn dependency.
    differences = positive[:, None] - negative[None, :]
    return float((np.count_nonzero(differences > 0) + 0.5 * np.count_nonzero(differences == 0)) / differences.size)


def load_task1_table(path: Path):
    if path.suffix.lower() == ".npz":
        archive = np.load(path, allow_pickle=True)
        truth = np.asarray(archive["y_true"], dtype=np.float64)
        probability = np.asarray(archive["probabilities"], dtype=np.float64)
        labels = [str(x) for x in archive.get("labels", np.arange(truth.shape[1]))]
        groups = np.asarray(archive.get("groups", np.arange(len(truth)))).astype(str)
        return truth, probability, labels, groups
    import pandas as pd
    frame = pd.read_csv(path)
    true_columns = [column for column in frame if column.startswith("true_")]
    labels = [column[5:] for column in true_columns]
    probability_columns = [f"prob_{label}" for label in labels]
    missing = [column for column in probability_columns if column not in frame]
    if not labels or missing:
        raise ValueError(
            "Task-1 CSV needs matching true_<label> and prob_<label> columns; "
            f"missing={missing[:5]}"
        )
    group_column = next((x for x in ("patient_id", "group", "VolumeName", "input_image_name") if x in frame), None)
    if group_column:
        raw_groups = frame[group_column].astype(str).to_numpy()
        if group_column in ("VolumeName", "input_image_name"):
            groups = np.asarray([
                "_".join(Path(value).name.split("_")[:2])
                if "_" in Path(value).name else Path(value).stem
                for value in raw_groups
            ])
        else:
            groups = raw_groups
    else:
        groups = np.arange(len(frame)).astype(str)
    return (
        frame[true_columns].to_numpy(dtype=np.float64),
        frame[probability_columns].to_numpy(dtype=np.float64),
        labels,
        groups,
    )


def grouped_split(groups, fraction: float, seed: int):
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    count = min(max(int(round(len(unique) * fraction)), 1), max(len(unique) - 1, 1))
    calibration_groups = set(unique[:count])
    calibration = np.asarray([value in calibration_groups for value in groups], dtype=bool)
    if calibration.all() or (~calibration).all():
        order = rng.permutation(len(groups))
        calibration = np.zeros(len(groups), dtype=bool)
        calibration[order[: max(1, len(groups) // 2)]] = True
    return calibration, ~calibration


def fit_logistic_calibrator(truth, probability):
    raw = logit(probability)
    truth = np.asarray(truth, dtype=np.float64)
    if len(np.unique(truth)) < 2:
        return 1.0, 0.0
    try:
        from scipy.optimize import minimize
        def objective(parameters):
            temperature = math.exp(float(parameters[0]))
            calibrated = sigmoid(raw / temperature + float(parameters[1]))
            nll = -(truth * np.log(calibrated + 1e-9) + (1 - truth) * np.log(1 - calibrated + 1e-9)).mean()
            return float(nll + 1e-4 * np.square(parameters).sum())
        result = minimize(objective, np.zeros(2), method="L-BFGS-B", bounds=((-3.0, 3.0), (-10.0, 10.0)))
        return float(math.exp(result.x[0])), float(result.x[1])
    except Exception as error:
        print(f"[task1] scipy calibrator unavailable ({error}); using identity temperature", flush=True)
        return 1.0, 0.0


def best_threshold(truth, probability):
    candidates = np.unique(np.concatenate([np.linspace(0.01, 0.99, 197), probability]))
    scored = [(binary_f1(truth, probability >= threshold), float(threshold)) for threshold in candidates]
    scored.sort(key=lambda item: (-item[0], abs(item[1] - 0.5), item[1]))
    return scored[0][1], scored[0][0]


def task1_fit(args) -> int:
    truth, probability, labels, groups = load_task1_table(args.input)
    if truth.shape != probability.shape or truth.shape[1] != len(labels):
        raise ValueError(f"Mismatched Task-1 arrays: truth={truth.shape} probability={probability.shape} labels={len(labels)}")
    calibration_mask, locked_mask = grouped_split(groups, args.calibration_fraction, args.seed)
    parameters = {}
    before_f1 = []
    locked_f1 = []
    locked_auc = []
    rows = []
    for index, label in enumerate(labels):
        temperature, bias = fit_logistic_calibrator(truth[calibration_mask, index], probability[calibration_mask, index])
        calibrated = sigmoid(logit(probability[:, index]) / temperature + bias)
        threshold, calibration_f1 = best_threshold(truth[calibration_mask, index], calibrated[calibration_mask])
        deployed = sigmoid(logit(calibrated) - logit(threshold))
        raw_f1 = binary_f1(truth[locked_mask, index], probability[locked_mask, index] >= 0.5)
        new_f1 = binary_f1(truth[locked_mask, index], deployed[locked_mask] >= 0.5)
        auc = binary_auc(truth[locked_mask, index], deployed[locked_mask])
        before_f1.append(raw_f1)
        locked_f1.append(new_f1)
        locked_auc.append(auc)
        parameters[label] = {"temperature": temperature, "bias": bias, "threshold": threshold}
        rows.append({
            "label": label,
            "raw_locked_f1": raw_f1,
            "calibrated_locked_f1": new_f1,
            "locked_auroc": auc,
            "calibration_f1": calibration_f1,
            **parameters[label],
        })
    mean_before = float(np.nanmean(before_f1))
    mean_after = float(np.nanmean(locked_f1))
    mean_auc = float(np.nanmean(locked_auc))
    accepted = mean_after >= mean_before + args.minimum_gain
    config = {
        "schema": "vlm3d-task1-metric-calibration-v1",
        "source": str(args.input.resolve()),
        "selection": "patient/group-held-out macro-F1; AUROC ranking is preserved by monotonic transforms",
        "official_crg": False,
        "warning": "This improves a CRG-aligned component. Run the official evaluator before calling it a CRG improvement.",
        "accepted": accepted,
        "locked_rows": int(locked_mask.sum()),
        "calibration_rows": int(calibration_mask.sum()),
        "raw_locked_macro_f1": mean_before,
        "calibrated_locked_macro_f1": mean_after,
        "locked_macro_auroc": mean_auc,
        "parameters": parameters,
        "per_label": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "task1_metric_calibration.json", config)
    print("[task1] held-out calibration result", flush=True)
    print(json.dumps({key: config[key] for key in ("accepted", "raw_locked_macro_f1", "calibrated_locked_macro_f1", "locked_macro_auroc")}, indent=2))
    if not accepted:
        print("[task1] REJECTED: keeping the original probabilities is safer", flush=True)
        return 0
    print(f"[task1] ACCEPTED: {args.output / 'task1_metric_calibration.json'}", flush=True)
    return 0


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", str(text).lower())


def case_group(value: str) -> str:
    name = Path(str(value)).name
    pieces = name.split("_")
    return "_".join(pieces[:2]) if len(pieces) >= 2 else Path(name).stem


def ngrams(tokens: list[str], size: int) -> Counter:
    return Counter(tuple(tokens[index:index + size]) for index in range(max(0, len(tokens) - size + 1)))


def bleu(reference: str, candidate: str) -> float:
    ref = tokenize(reference)
    hyp = tokenize(candidate)
    if not hyp:
        return 0.0
    precisions = []
    for size in range(1, 5):
        ref_counts = ngrams(ref, size)
        hyp_counts = ngrams(hyp, size)
        overlap = sum(min(count, ref_counts[key]) for key, count in hyp_counts.items())
        total = sum(hyp_counts.values())
        precisions.append((overlap + 1.0) / (total + 1.0))
    brevity = 1.0 if len(hyp) >= len(ref) else math.exp(1.0 - len(ref) / max(len(hyp), 1))
    cumulative = [
        brevity * math.exp(sum(math.log(value) for value in precisions[:order]) / order)
        for order in range(1, 5)
    ]
    # The challenge dashboard's BLEU(mean) is the mean of BLEU-1..BLEU-4.
    return float(np.mean(cumulative))


def rouge_l(reference: str, candidate: str) -> float:
    a, b = tokenize(reference), tokenize(candidate)
    if not a or not b:
        return 0.0
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0]
        for index, other in enumerate(b, start=1):
            current.append(previous[index - 1] + 1 if token == other else max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(b)
    recall = lcs / len(a)
    return float(2 * precision * recall / max(precision + recall, 1e-12))


def meteor_lite(reference: str, candidate: str) -> float:
    ref, hyp = tokenize(reference), tokenize(candidate)
    if not ref or not hyp:
        return 0.0
    ref_counts, hyp_counts = Counter(ref), Counter(hyp)
    overlap = sum(min(ref_counts[key], count) for key, count in hyp_counts.items())
    precision = overlap / len(hyp)
    recall = overlap / len(ref)
    if not precision or not recall:
        return 0.0
    return float(10 * precision * recall / (recall + 9 * precision))


def report_labels(text: str) -> set[str]:
    lower = str(text).lower()
    positive = set()
    for label, phrases in PATHOLOGIES.items():
        for phrase in phrases:
            position = lower.find(phrase)
            if position < 0:
                continue
            prefix = lower[max(0, position - 36):position]
            if not any(cue in prefix[-24:] for cue in NEGATIONS):
                positive.add(label)
                break
    return positive


def clinical_f1(reference: str, candidate: str) -> float:
    truth = report_labels(reference)
    predicted = report_labels(candidate)
    tp = len(truth & predicted)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    return float(2 * tp / max(2 * tp + fp + fn, 1)) if truth or predicted else 1.0


def cider_idf(references: Iterable[str]):
    documents = list(references)
    result = {}
    for size in range(1, 5):
        document_frequency = Counter()
        for text in documents:
            document_frequency.update(set(ngrams(tokenize(text), size)))
        result[size] = {key: math.log((len(documents) + 1.0) / (count + 1.0)) + 1.0 for key, count in document_frequency.items()}
    return result


def cider(reference: str, candidate: str, idf) -> float:
    values = []
    for size in range(1, 5):
        ref_counts = ngrams(tokenize(reference), size)
        hyp_counts = ngrams(tokenize(candidate), size)
        keys = set(ref_counts) | set(hyp_counts)
        if not keys:
            values.append(0.0)
            continue
        ref = np.asarray([ref_counts[key] * idf[size].get(key, 1.0) for key in keys])
        hyp = np.asarray([hyp_counts[key] * idf[size].get(key, 1.0) for key in keys])
        denominator = np.linalg.norm(ref) * np.linalg.norm(hyp)
        values.append(float(ref @ hyp / denominator) if denominator else 0.0)
    return float(np.mean(values))


def read_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def task2_fit(args) -> int:
    cases = read_jsonl(args.input)
    if not cases:
        raise ValueError("Task-2 capture is empty")
    references = [str(case.get("reference", "")) for case in cases]
    if any(not value.strip() for value in references):
        raise ValueError("Every Task-2 case must include its held-out reference report")
    idf = cider_idf(references)
    rows = []
    per_case = []
    for case in cases:
        candidates = case.get("candidates") or []
        features = np.asarray(case.get("features"), dtype=np.float64)
        if len(candidates) != len(features) or len(candidates) < 2:
            raise ValueError(f"{case.get('case_id')}: candidates/features mismatch or fewer than 2 candidates")
        rewards = []
        metrics = []
        for candidate in candidates:
            text = str(candidate)
            values = {
                "bleu": bleu(case["reference"], text),
                "rouge_l": rouge_l(case["reference"], text),
                "meteor_lite": meteor_lite(case["reference"], text),
                "cider": cider(case["reference"], text, idf),
                "clinical_f1": clinical_f1(case["reference"], text),
            }
            reward = (
                args.clinical_weight * values["clinical_f1"]
                + args.bleu_weight * values["bleu"]
                + args.rouge_weight * values["rouge_l"]
                + args.meteor_weight * values["meteor_lite"]
                + args.cider_weight * values["cider"]
            )
            rewards.append(float(reward))
            metrics.append({**values, "reward": float(reward), "text": text})
        rows.append({**case, "features_array": features, "rewards": np.asarray(rewards), "metrics": metrics})
        per_case.append(case_group(str(case.get("case_id"))))

    unique_ratio = float(np.mean([
        len({re.sub(r"\s+", " ", str(value).strip().lower()) for value in row["candidates"]}) / len(row["candidates"])
        for row in rows
    ]))
    unique_groups = np.unique(per_case)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_groups)
    split = max(1, int(round(len(unique_groups) * 0.8)))
    train_groups = set(unique_groups[:split])
    train = [row for row in rows if case_group(str(row.get("case_id"))) in train_groups]
    locked = [row for row in rows if case_group(str(row.get("case_id"))) not in train_groups]
    if not locked:
        locked = rows[-max(1, len(rows) // 5):]
        train = rows[:-len(locked)] or rows
    x_train = np.concatenate([row["features_array"] for row in train], axis=0)
    y_train = np.concatenate([row["rewards"] for row in train], axis=0)
    model = fit_ridge(x_train, y_train, alpha=args.ridge_alpha)

    original_rewards, learned_rewards, oracle_rewards = [], [], []
    for row in locked:
        original_scores = np.asarray(
            row.get("original_scores", np.arange(len(row["rewards"]), 0, -1)),
            dtype=np.float64,
        )
        prediction = model.predict(row["features_array"])
        original_rewards.append(float(row["rewards"][int(np.argmax(original_scores))]))
        learned_rewards.append(float(row["rewards"][int(np.argmax(prediction))]))
        oracle_rewards.append(float(np.max(row["rewards"])))
    baseline = float(np.mean(original_rewards))
    learned = float(np.mean(learned_rewards))
    oracle = float(np.mean(oracle_rewards))
    accepted = unique_ratio >= args.minimum_unique_ratio and learned >= baseline + args.minimum_gain
    final_model = fit_ridge(
        np.concatenate([row["features_array"] for row in rows], axis=0),
        np.concatenate([row["rewards"] for row in rows], axis=0),
        alpha=args.ridge_alpha,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(final_model, args.output / "reranker.joblib")
    metadata = {
        "schema": "vlm3d-task2-reference-reward-distillation-v1",
        "accepted": accepted,
        "selection": "group-held-out learned reward over inference-available original reranker features",
        "reference_metrics_used_for_labels": ["BLEU(mean of 1-4)", "ROUGE-L", "METEOR-lite", "CIDEr-like", "rule-based clinical F1"],
        "official_crg": False,
        "warning": "Token metrics label validation candidates only; no hidden-test reference is used at inference.",
        "feature_count": int(final_model.n_features_in_),
        "candidate_unique_ratio": unique_ratio,
        "minimum_unique_ratio": args.minimum_unique_ratio,
        "locked_original_reward": baseline,
        "locked_learned_reward": learned,
        "locked_oracle_reward": oracle,
        "reward_weights": {
            "clinical_f1": args.clinical_weight,
            "bleu": args.bleu_weight,
            "rouge_l": args.rouge_weight,
            "meteor_lite": args.meteor_weight,
            "cider": args.cider_weight,
        },
    }
    atomic_json(args.output / "reranker_metric_alignment.json", metadata)
    print("[task2] held-out reward distillation result")
    print(json.dumps(metadata, indent=2))
    if not accepted:
        print("[task2] REJECTED: do not replace the existing reranker", flush=True)
        return 0
    print(f"[task2] ACCEPTED: {args.output / 'reranker.joblib'}", flush=True)
    return 0


def frechet_distance(real, generated) -> float:
    from scipy.linalg import sqrtm
    real = np.asarray(real, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    if real.ndim != 2 or generated.ndim != 2 or real.shape[1] != generated.shape[1]:
        raise ValueError(f"Feature shapes must be [cases, features], got {real.shape} and {generated.shape}")
    mu_real, mu_generated = real.mean(0), generated.mean(0)
    cov_real = np.cov(real, rowvar=False)
    cov_generated = np.cov(generated, rowvar=False)
    product_root = sqrtm(cov_real @ cov_generated)
    if np.iscomplexobj(product_root):
        if np.max(np.abs(product_root.imag)) > 1e-5:
            raise ValueError("FVD covariance square root has a material imaginary component")
        product_root = product_root.real
    difference = mu_real - mu_generated
    return float(difference @ difference + np.trace(cov_real + cov_generated - 2.0 * product_root))


def task3_fvd(args) -> int:
    real_archive = np.load(args.real, allow_pickle=False)
    generated_archive = np.load(args.generated, allow_pickle=False)
    real_key = args.real_key or ("features" if "features" in real_archive else real_archive.files[0])
    generated_key = args.generated_key or ("features" if "features" in generated_archive else generated_archive.files[0])
    value = frechet_distance(real_archive[real_key], generated_archive[generated_key])
    evidence = {
        "FVD": value,
        "feature_model": args.feature_model,
        "official_FVD_CTNet": bool(args.official_ctnet),
        "real_features": str(args.real.resolve()),
        "generated_features": str(args.generated.resolve()),
        "selection_allowed": bool(args.official_ctnet),
    }
    if not args.official_ctnet:
        evidence["warning"] = "This is exact Frechet math on non-official features, not leaderboard FVD_CTNet."
    atomic_json(args.output, evidence)
    print(json.dumps(evidence, indent=2))
    return 0


def task3_select(args) -> int:
    rows = json.loads(args.metrics.read_text())
    if isinstance(rows, dict):
        rows = rows.get("candidates", rows.get("results", []))
    eligible = []
    for row in rows:
        reasons = []
        if row.get("official_FVD_CTNet") is not True:
            reasons.append("not official CT-Net FVD")
        if row.get("FVD_CTNet") is None:
            reasons.append("missing FVD_CTNet")
        if row.get("complete_contract") is not True:
            reasons.append("output contract not passed")
        if row.get("projected_hours") is not None and float(row["projected_hours"]) > args.max_hours:
            reasons.append("runtime cap")
        row = {**row, "eligible": not reasons, "rejection_reasons": reasons}
        if not reasons:
            eligible.append(row)
    if not eligible:
        raise SystemExit("No Task-3 candidate has official FVD_CTNet + contract + runtime evidence")
    eligible.sort(key=lambda row: (float(row["FVD_CTNet"]), -float(row.get("CTCLIP", -1e30))))
    winner = eligible[0]
    result = {
        "winner": winner["name"],
        "reason": "lowest official held-out FVD_CTNet; CTCLIP breaks ties",
        "winner_evidence": winner,
        "all_candidates": rows,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


def self_test(args) -> int:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(29)
    labels = np.asarray(["a", "b", "c"])
    truth = rng.binomial(1, [0.15, 0.35, 0.55], size=(240, 3))
    probability = np.clip(0.08 + 0.55 * truth + rng.normal(0, 0.18, truth.shape), 0.001, 0.999)
    np.savez(output / "task1.npz", y_true=truth, probabilities=probability, labels=labels, groups=np.arange(240))
    task1_args = argparse.Namespace(
        input=output / "task1.npz", output=output / "task1_out",
        calibration_fraction=0.6, seed=29, minimum_gain=-1.0,
    )
    assert task1_fit(task1_args) == 0
    capture = output / "task2.jsonl"
    with capture.open("w") as handle:
        for index in range(40):
            reference = "No pleural effusion. Mild right lower lobe atelectasis."
            candidates = [
                "No acute abnormality.",
                "Mild right lower lobe atelectasis without pleural effusion.",
                "Large left pleural effusion and pneumonia.",
            ]
            features = [[0, 0, 1], [1, 1, 0], [-1, 0, 1]]
            handle.write(json.dumps({
                "case_id": f"case_{index:03d}", "reference": reference,
                "candidates": candidates, "features": features, "original_scores": [0.9, 0.3, 0.2],
            }) + "\n")
    task2_args = argparse.Namespace(
        input=capture, output=output / "task2_out", seed=29, ridge_alpha=1.0,
        minimum_unique_ratio=0.75, minimum_gain=-1.0, clinical_weight=0.45,
        bleu_weight=0.15, rouge_weight=0.15, meteor_weight=0.10, cider_weight=0.15,
    )
    assert task2_fit(task2_args) == 0
    real = rng.normal(size=(64, 8))
    generated = real + 0.1
    np.savez(output / "real.npz", features=real)
    np.savez(output / "generated.npz", features=generated)
    task3_args = argparse.Namespace(
        real=output / "real.npz", generated=output / "generated.npz",
        real_key=None, generated_key=None, feature_model="synthetic", official_ctnet=False,
        output=output / "fvd.json",
    )
    task3_fvd(task3_args)
    atomic_json(output / "SELF_TEST_PASS.json", {"status": "PASS"})
    print("METRIC ALIGNMENT SELF-TEST PASS")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    item = commands.add_parser("task1-fit")
    item.add_argument("--input", type=Path, required=True)
    item.add_argument("--output", type=Path, required=True)
    item.add_argument("--calibration-fraction", type=float, default=0.60)
    item.add_argument("--seed", type=int, default=29)
    item.add_argument("--minimum-gain", type=float, default=0.002)
    item.set_defaults(func=task1_fit)
    item = commands.add_parser("task2-fit")
    item.add_argument("--input", type=Path, required=True)
    item.add_argument("--output", type=Path, required=True)
    item.add_argument("--seed", type=int, default=29)
    item.add_argument("--ridge-alpha", type=float, default=3.0)
    item.add_argument("--minimum-unique-ratio", type=float, default=0.75)
    item.add_argument("--minimum-gain", type=float, default=0.002)
    item.add_argument("--clinical-weight", type=float, default=0.45)
    item.add_argument("--bleu-weight", type=float, default=0.15)
    item.add_argument("--rouge-weight", type=float, default=0.15)
    item.add_argument("--meteor-weight", type=float, default=0.10)
    item.add_argument("--cider-weight", type=float, default=0.15)
    item.set_defaults(func=task2_fit)
    item = commands.add_parser("task3-fvd")
    item.add_argument("--real", type=Path, required=True)
    item.add_argument("--generated", type=Path, required=True)
    item.add_argument("--real-key")
    item.add_argument("--generated-key")
    item.add_argument("--feature-model", required=True)
    item.add_argument("--official-ctnet", action="store_true")
    item.add_argument("--output", type=Path, required=True)
    item.set_defaults(func=task3_fvd)
    item = commands.add_parser("task3-select")
    item.add_argument("--metrics", type=Path, required=True)
    item.add_argument("--output", type=Path, required=True)
    item.add_argument("--max-hours", type=float, default=11.5)
    item.set_defaults(func=task3_select)
    item = commands.add_parser("self-test")
    item.add_argument("--output", type=Path, required=True)
    item.set_defaults(func=self_test)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    raise SystemExit(arguments.func(arguments))
