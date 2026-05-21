# Script Index

Scripts are CLI entrypoints. Reusable logic should live under `src/eyenet/`.

## Dataset Audit and Conversion

```text
audit_ems_dataset.py
build_ems_events.py
build_ems_qc_filtered_tables.py
build_encoder_ready_table.py
build_filtered_subject_splits.py
build_subject_qc_report.py
inspect_pymovements_dataset.py
download_pymovements_dataset.py
validate_dataset_schema.py
```

## Feature Extraction

```text
extract_ems_features.py
extract_ems_segment_features.py
build_ems_event_temporal_sequences.py
```

## Baselines and Fixed-Split Models

```text
train_ems_baseline.py
train_ems_baseline_fixed_split.py
train_ems_segment_baseline.py
train_ems_segment_sequence.py
train_ems_segment_sequence_fixed_split.py
train_ems_event_temporal_sequence.py
train_ems_event_temporal_sequence_fixed_split.py
train_ems_dual_stream_concat_fixed_split.py
train_ems_dual_stream_gated_fixed_split.py
```

## Encoder Pretraining

```text
check_encoder_dataloader.py
train_masked_event_modeling_smoke.py
train_supervised_encoder_smoke.py
summarize_encoder_transfer_results.py
```

## Analysis and Diagnostics

```text
analyze_attention_segments.py
analyze_dual_stream_fusion.py
analyze_ems_baseline.py
analyze_ems_qc_errors.py
analyze_multiseed_predictions.py
analyze_thresholds.py
compare_models_bootstrap.py
diagnose_ems_fold_distribution.py
diagnose_event_temporal_sequence.py
summarize_fixed_split_results.py
summarize_qc_variant_results.py
validate_attention_statistics.py
```

## Paper Drafting

```text
create_paper_draft_docx.py
```

## Current Main Next Command

After HBN adapter is implemented, the expected pipeline should be:

```powershell
$env:PYTHONPATH="D:\CodeProjects\Python\eyenet\src"
python scripts/<build_hbn_events>.py
python scripts/validate_dataset_schema.py --events data/processed/HBN/hbn_events.csv
python scripts/build_subject_qc_report.py --events data/processed/HBN/hbn_events.csv --output-dir data/processed/HBN/qc
python scripts/build_encoder_ready_table.py --events data/processed/HBN/hbn_events.csv --output-dir data/processed/HBN/encoder_ready/no_position
python scripts/train_masked_event_modeling_smoke.py --events data/processed/HBN/encoder_ready/no_position/hbn_encoder_events.csv
```
