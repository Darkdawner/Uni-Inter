# Uni-Inter: Unifying 3D Human Motion Synthesis Across Diverse Interaction Contexts

<p align="center">
  <a href="https://arxiv.org/abs/2511.13032"><img src="https://img.shields.io/badge/arXiv-2511.13032-b31b1b.svg" alt="arXiv"></a>
  <a href="https://dl.acm.org/doi/10.1145/3757377.3763954"><img src="https://img.shields.io/badge/SIGGRAPH%20Asia-2025-blue.svg" alt="SIGGRAPH Asia 2025"></a>
  <a href="https://doi.org/10.1145/3757377.3763954"><img src="https://img.shields.io/badge/DOI-10.1145%2F3757377.3763954-green.svg" alt="DOI"></a>
</p>

> **Uni-Inter: Unifying 3D Human Motion Synthesis Across Diverse Interaction Contexts**
>
> Sheng Liu, Yuanzhi Liang, Jiepeng Wang, Sidan Du, Chi Zhang, Xuelong Li
>
> *SIGGRAPH Asia 2025 Conference Papers*

## Overview

Uni-Inter is a unified framework for human motion generation that supports a wide range of interaction scenarios—including human-human, human-object, and human-scene—within a single, task-agnostic architecture. It introduces the **Unified Interactive Volume (UIV)**, a volumetric representation that encodes heterogeneous interactive entities into a shared spatial field, enabling consistent relational reasoning and compound interaction modeling. Motion generation is formulated as joint-wise probabilistic prediction over the UIV using a **4D UNet diffusion model** conditioned on **CLIP text features** and **interaction context voxels**.

### Supported Interaction Scenarios

| Dataset | Interaction Type | Description |
|---------|-----------------|-------------|
| BEHAVE | Human-Object | Human interacting with rigid objects |
| NTU | Human-Human | Action-label-driven two-person interaction |
| CHI3D | Human-Human | Close-range human-human interaction |
| TRUMANS | Human-Scene+Object | Human navigating and interacting in scenes |
| MANIP (OMOMO) | Human-Object | Human manipulating articulated objects |

## Installation

```bash
# Clone the repository
git clone https://github.com/xxx/Uni-Inter.git
cd Uni-Inter

# Create conda environment
conda create -n uniinter python=3.9 -y
conda activate uniinter

# Install dependencies
pip install -r requirements.txt
```

### Download CLIP Model

Download [ViT-B/32](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt) and place it as `./ViT-B-32.pt`

## Data Preparation

Each dataset should be preprocessed following its **original official pipeline**. Additionally, all human motion data must be **inverse-fitted to SMPL/SMPL-X format** to extract body mesh vertices, which are used as conditional voxel inputs during training.

Set your data root path via environment variable or modify `configs/config.py`:

```bash
export UNIINTER_DATA_ROOT=/path/to/your/datasets
```

Expected data structure under `UNIINTER_DATA_ROOT`:

```
datasets/
├── behave_t2m/
│   ├── new_joints_local/     # Human joint positions
│   ├── object_joints_local/  # Object point clouds
│   └── texts/                # Text descriptions
├── ntu/
│   ├── train/                # person1/, person2/, person1_smplx/, person2_smplx/
│   └── test/
├── chi3d/
│   ├── train/                # person1/, person2/, person1_smplx/, person2_smplx/
│   └── test/
├── trumans/
│   ├── subject/              # Human motion
│   └── object/               # object/, scene/
└── processed_data/           # MANIP/OMOMO data
    ├── cano_train_diffusion_manip_window_120_joints24.p
    ├── cano_test_diffusion_manip_window_120_joints24.p
    ├── object_motion/        # train/, test/
    └── omomo_text_anno_json_data/
```

## Training

```bash
# 4-GPU distributed training
bash train.sh
```

Or manually:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 torchrun \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:29505 \
    --nproc_per_node=4 train.py
```

Key training configurations can be modified in `configs/config.py`.

## Inference

After training, modify `checkpoint_path` in `configs/config.py` to point to your trained checkpoint, then run:

```bash
python infer.py
```

This generates GIF animations (front view + top view) for both ground truth and predictions, saved to `./tmp_results/`.

## Evaluation

Generate predictions for specific datasets:

```bash
python scripts/gen_manip.py     # MANIP/OMOMO evaluation
python scripts/gen_ntu.py       # NTU evaluation
python scripts/gen_trumans.py   # TRUMANS evaluation
```

Results are saved to `./eval_{dataset}/pred/`.

## Project Structure

```
Uni-Inter/
├── configs/                # Configuration
│   └── config.py           # Global configuration
├── models/                 # Model architecture
│   ├── UNet4D.py           # 4D UNet architecture
│   └── diffusion.py        # Diffusion process (noise schedules + DDIM sampling)
├── datasets/               # Dataset loaders
│   ├── mixed_dataset.py    # Unified multi-dataset loader
│   ├── behave.py           # BEHAVE dataset
│   ├── ntu.py              # NTU dataset
│   ├── chi3d.py            # CHI3D dataset
│   ├── trumans.py          # TRUMANS dataset
│   └── manip.py            # MANIP/OMOMO dataset
├── utils/                  # Utilities
│   ├── helpers.py          # Optimizer, scheduler, checkpoint loading
│   ├── quaternion.py       # Quaternion operations
│   ├── skeleton.py         # Forward/Inverse kinematics
│   └── visualization.py    # Skeleton visualization (3D animation)
├── scripts/                # Evaluation generation scripts
│   ├── gen_manip.py        # MANIP evaluation generation
│   ├── gen_ntu.py          # NTU evaluation generation
│   └── gen_trumans.py      # TRUMANS evaluation generation
├── data_splits/            # Train/test split JSON files
├── train.py                # Distributed training script
├── train.sh                # Training launcher
├── infer.py                # Inference with visualization
├── requirements.txt        # Python dependencies
└── .gitignore
```

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{10.1145/3757377.3763954,
  author = {Liu, Sheng and Liang, Yuanzhi and Wang, Jiepeng and Du, Sidan and Zhang, Chi and Li, Xuelong},
  title = {Uni-Inter: Unifying 3D Human Motion Synthesis Across Diverse Interaction Contexts},
  year = {2025},
  isbn = {9798400721373},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3757377.3763954},
  doi = {10.1145/3757377.3763954},
  booktitle = {Proceedings of the SIGGRAPH Asia 2025 Conference Papers},
  articleno = {186},
  numpages = {11},
  series = {SA Conference Papers '25}
}
```

## Acknowledgements

- [CLIP](https://github.com/openai/CLIP) for text encoding
- Quaternion utilities from [Facebook Research](https://github.com/facebookresearch/QuaterNet)

## License

TBD
