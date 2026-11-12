"""Six-class Spark ML baselines with train-only preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

from src.features.eeg import feature_names
from src.spark_jobs.features import TARGET_COLS


LABEL_NAMES = ("seizure", "lpd", "gpd", "lrda", "grda", "other")


def _metrics(prediction):
    from pyspark.sql import functions as F
    from pyspark.ml.functions import vector_to_array
    counts = prediction.groupBy("class_id", "prediction").count().collect()
    confusion = [[0 for _ in range(6)] for _ in range(6)]
    for row in counts:
        confusion[int(row["class_id"])][int(row["prediction"])] = row["count"]
    total = sum(map(sum, confusion))
    recall, precision, f1 = [], [], []
    for i in range(6):
        tp = confusion[i][i]
        p = tp / max(sum(confusion[j][i] for j in range(6)), 1)
        r = tp / max(sum(confusion[i]), 1)
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / max(p + r, 1e-12))
    probabilities = vector_to_array(F.col("probability"))
    targets = F.array(*[F.col(key) for key in TARGET_COLS])
    scored = prediction.withColumn("target_array", targets).withColumn("probability_array", probabilities)
    terms = F.zip_with("target_array", "probability_array",
        lambda y, p: F.when(y > 0, y * (F.log(y) - F.log(F.greatest(p, F.lit(1e-12))))).otherwise(F.lit(0.0)))
    scored = scored.withColumn("kl", F.aggregate(terms, F.lit(0.0), lambda acc, term: acc + term))
    kl_window = scored.agg(F.avg("kl")).first()[0]
    kl_patient = scored.groupBy("patient_id").agg(F.avg("kl").alias("patient_kl")).agg(F.avg("patient_kl")).first()[0]
    record_summary = (scored.groupBy("eeg_id").agg(F.avg("kl").alias("record_kl"))
                      .agg(F.avg("record_kl").alias("equal_kl"),
                           F.count("*").alias("n_records")).first())
    return {"n": total, "accuracy": sum(confusion[i][i] for i in range(6)) / max(total, 1),
            "kl_window_mean": kl_window, "kl_patient_equal_mean": kl_patient,
            "kl_eeg_record_equal_mean": record_summary["equal_kl"],
            "n_eeg_records": record_summary["n_records"],
            "macro_f1": sum(f1) / 6, "macro_recall": sum(recall) / 6,
            "per_class_precision": precision, "per_class_recall": recall,
            "per_class_f1": f1, "confusion": confusion}


def train_baselines(spark, features_path, output_dir, seed=42, rf_trees=100,
                    lr_max_iter=100, overwrite=False, experiment_status="unverified",
                    expected_windows=None):
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
    from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler
    from pyspark.sql import functions as F
    from pyspark.ml.functions import vector_to_array
    source = spark.read.parquet(features_path)
    cols = feature_names()
    missing = set(cols + ["window_id", "eeg_id", "patient_id", "split",
                          "class_id", "quality_ok"]) - set(source.columns)
    if missing:
        raise ValueError("feature table missing: %s" % sorted(missing))
    data = source.filter(F.col("quality_ok") & F.col("class_id").between(0, 5))
    if data.filter(F.col("eeg_id").isNull()).limit(1).count():
        raise ValueError("Spark baseline windows require EEG recording IDs")
    target_available = set(TARGET_COLS).issubset(source.columns)
    if target_available:
        target_sum = sum((F.col(key) for key in TARGET_COLS), F.lit(0.0))
        invalid_target = F.isnan(target_sum) | (F.abs(target_sum - 1.0) > 1e-4)
        for key in TARGET_COLS:
            invalid_target = (invalid_target | F.col(key).isNull() | F.isnan(F.col(key)) |
                              (F.col(key) < 0) | (F.col(key) > 1))
        target_available = data.filter(invalid_target).limit(1).count() == 0
    if not target_available:
        if experiment_status == "formal":
            raise ValueError("Formal Spark baselines require six soft target columns")
        for i, key in enumerate(TARGET_COLS):
            data = data.withColumn(key, F.when(F.col("class_id") == i, F.lit(1.0)).otherwise(F.lit(0.0)))
    counts = {r["split"]: r["count"] for r in data.groupBy("split").count().collect()}
    if experiment_status == "formal":
        if expected_windows is None or sum(counts.values()) != int(expected_windows):
            raise ValueError("Formal Spark features do not cover every audited valid window")
    if counts.get("train", 0) == 0:
        raise ValueError("no valid train windows")
    train_classes = {int(r["class_id"]) for r in data.filter(
        F.col("split") == "train").select("class_id").distinct().collect()}
    if train_classes != set(range(6)):
        raise ValueError("train split must contain all six classes; found %s" %
                         sorted(train_classes))
    out = Path(output_dir)
    if out.exists() and not overwrite:
        raise FileExistsError(str(out))
    out.mkdir(parents=True, exist_ok=True)
    results = {"features_path": features_path, "counts": counts,
               "experiment_status": experiment_status,
               "target_source": "expert_votes" if target_available else "hard_label_fallback",
               "spark_version": spark.version, "seed": seed,
               "rf_trees": rf_trees, "lr_max_iter": lr_max_iter,
               "label_order": LABEL_NAMES, "feature_names": cols, "models": {}}
    for kind in ("lr", "rf"):
        imputed = ["imputed_" + c for c in cols]
        stages = [Imputer(inputCols=cols, outputCols=imputed,
                          strategy="median"),
                  VectorAssembler(inputCols=imputed, outputCol="raw_features")]
        if kind == "lr":
            stages += [StandardScaler(inputCol="raw_features", outputCol="features",
                                      withMean=True, withStd=True),
                       LogisticRegression(featuresCol="features", labelCol="class_id",
                                          family="multinomial", maxIter=lr_max_iter,
                                          regParam=0.01)]
        else:
            stages += [RandomForestClassifier(featuresCol="raw_features",
                                              labelCol="class_id", seed=seed,
                                              numTrees=rf_trees, maxDepth=8)]
        model = Pipeline(stages=stages).fit(data.filter(F.col("split") == "train"))
        model_path = str(out / (kind + "_model"))
        model.write().overwrite().save(model_path)
        evaluations = {}
        for split in ("train", "val", "test"):
            if counts.get(split, 0) == 0:
                continue
            pred = model.transform(data.filter(F.col("split") == split))
            evaluations[split] = _metrics(pred)
            selected = pred.select("window_id", "eeg_id", "patient_id", "split",
                "label_id", "class_id", F.col("prediction").cast("int").alias("prediction"),
                *[F.col(key) for key in TARGET_COLS],
                vector_to_array("probability").alias("probabilities"))
            selected.write.mode("overwrite").parquet(str(out / (kind + "_" + split + "_predictions")))
        results["models"][kind] = {"model_path": model_path, "splits": evaluations}
    (out / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return results
