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
build_ems_subject_summary_features.py
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
train_ems_encoder_dual_stream_fixed_split.py
train_ems_subject_summary_baseline.py
train_ems_summary_encoder_dual_stream_fixed_split.py
```

## Encoder Pretraining

```text
check_encoder_dataloader.py
train_masked_event_modeling_smoke.py
train_mem_pretrain.py
train_supervised_encoder_smoke.py
summarize_encoder_transfer_results.py
summarize_phase1_encoder_results.py
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
summarize_old_encoder_dual_stream.py
summarize_ems_subject_summary_baseline.py
summarize_ems_summary_encoder_dual_stream.py
validate_attention_statistics.py
```

## Paper Drafting

```text
create_paper_draft_docx.py
create_phase1_docx_deliverables.py
```

## Current Main Next Command

The package should be installed once in editable mode:

```powershell
conda activate eyenet
cd D:\CodeProjects\Python\eyenet
python -m pip install -e .
```

The current project priority is phase-1 documentation and reproducibility cleanup. Do not start long training jobs without an explicit run decision.

Current source-of-truth summaries:

```text
experiments/encoder_downstream/phase1_encoder_summary.csv
experiments/encoder_downstream/phase1_encoder_split_leakage_audit.csv
experiments/ems_encoder_dual_stream/old_encoder_dual_stream_summary.csv
experiments/ems_subject_summary_baseline_strict/summary.csv
experiments/ems_summary_encoder_dual_stream/summary.csv
docs/current_experiment_summary.md
docs/encoder_model_selection_summary.md
docs/old_encoder_dual_stream_closure.md
docs/new_summary_encoder_dual_stream_closure.md
```

Transformer, Saliency4ASD, and additional dual-stream design are deferred until the phase-1 report and paper draft are stable.
