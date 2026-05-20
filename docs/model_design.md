# Model Design

## Project Goal

Build a hardware-aware but hardware-robust eye movement screening framework for adolescent mental health risk.

## Core Modeling Idea

All datasets are converted to event-level eye movement representations before modeling.

```text
raw gaze or fixation table
-> shared event representation
-> spatial stream + temporal stream
-> fusion
-> subject-level prediction
```

## Spatial Stream

The spatial stream models fixation locations and transitions.

- Input: fixation graph or graph/statistical spatial features.
- Nodes: fixations.
- Edges: adjacent transitions or spatial neighbors.
- Candidate models: GAT, GCN, graph statistics + MLP.

## Temporal Stream

The temporal stream models event dynamics.

- Input: duration, amplitude, angle, velocity sequences.
- Candidate models: TCN, GRU with attention, 1D-CNN.

## Fusion

Use vector gated fusion when deep models are used:

```text
g = sigmoid(W[h_spatial; h_temporal] + b)
h = g * h_spatial + (1 - g) * h_temporal
```

## Clinical Deployment Considerations

- Output calibrated risk probabilities.
- Include data-quality rejection criteria.
- Avoid reliance on image/video content or optional pupil fields.
