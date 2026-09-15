# GLCFormer

Official cleaned code release for **GLCFormer**, a graph-enhanced land-cover fusion model for greenhouse/barn segmentation from remote-sensing imagery.

This repository contains code only. Datasets, generated graph files, experiment logs, and model checkpoints are intentionally excluded.

## Overview

GLCFormer combines an image segmentation backbone with a land-cover graph branch. The main configuration uses:

- image backbone: Swin-B
- decoder: UPerNet
- graph branch: GeoLink-style land-cover graph fusion
- task: binary greenhouse/barn segmentation

The primary experiment config is:

```bash
configs/glcformer_swin_b_upernet_geolink_v3.py
```

The original internal config path is also preserved for traceability:

```bash
configs/final_250906/spatial_split/Dual_Branch/Swin_B_UperNet_LandCoverGraph_GeoLink_v3_train_semseg.py
```

## Repository layout

```text
configs/                         # GLCFormer training configuration
mmgeo/
  datasets/                       # dataset, image/mask loading, graph loading
  evaluation/                     # segmentation metrics
  hooks/                          # optional training hooks
  models/                         # SlideEncoderDecoder and graph fusion modules
  visualization/                  # visualization helpers
tools/                            # train / inference / sanity-check scripts
utils/satellite_graph/            # land-cover graph construction utilities
```

## Key implementation files

- `mmgeo/models/segmentors/slide_encoder_decoder.py`
- `mmgeo/models/swin_geolink_graph_fusion_v3.py`
- `mmgeo/models/swin_geolink_graph_fusion.py`
- `mmgeo/models/v3_graph_fusion.py`
- `mmgeo/datasets/semseg/transforms/loading_graph.py`
- `mmgeo/datasets/semseg/transforms/formatting_aerial.py`

## Installation

This project is based on OpenMMLab-style training code.

```bash
pip install -r requirements.txt
```

You also need compatible versions of PyTorch, MMCV, MMEngine, MMSegmentation, and MMDetection for your CUDA environment.

## Data preparation

The config expects the following user-provided assets under `data/exp3_250818` by default:

```text
images/recent/
masks/recent/binary_inmap_roi/
labels/
data_list_final/
graphs_hier/
```

Land-cover graph files are loaded through `LoadLandCoverGraph` and passed through sample metadata as `landcover_graph`.

No dataset files are included in this repository.

## Training

```bash
python tools/train_semseg.py configs/glcformer_swin_b_upernet_geolink_v3.py
```

Update `data_root`, `data_list_path`, and graph directory paths in the config before training on a new machine.

## Sanity checks

Shape-level checks for the graph fusion modules are included:

```bash
python tools/test_swin_geolink_graph_fusion_shape.py
python tools/test_swin_geolink_graph_fusion_v2_shape.py
python tools/test_v3_graph_fusion_shape.py
```

## Notes

- This release is code-only.
- Paths in the config are relative placeholders and should be adjusted for your environment.
- Pretrained Swin weights are referenced through the public OpenMMLab checkpoint URL in the config.
