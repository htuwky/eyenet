# Script Index

Scripts are CLI entrypoints. Reusable logic should live under `src/eyenet/`.

## Dataset Audit and Conversion

```text
audit_ems_dataset.py
build_ems_events.py
build_hbn_fixation_events.py
build_ems_qc_filtered_tables.py
build_encoder_ready_table.py
combine_encoder_ready_tables.py
filter_encoder_ready_by_subject_qc.py
build_filtered_subject_splits.py
build_subject_qc_report.py
create_self_supervised_subject_split.py
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
train_mem_pretrain.py
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

The package should be installed once in editable mode:

```powershell
conda activate eyenet
cd D:\CodeProjects\Python\eyenet
python -m pip install -e .
```

Current next dataset task is Saliency4ASD adapter screening. HBN and GazeBase have already completed the shared path:

```text
raw/public dataset
  -> shared fixation-event schema
  -> validate_dataset_schema.py
  -> build_subject_qc_report.py
  -> build_encoder_ready_table.py
  -> combine_encoder_ready_tables.py with EMS
  -> create_self_supervised_subject_split.py
  -> check_encoder_dataloader.py
  -> train_mem_pretrain.py
```
