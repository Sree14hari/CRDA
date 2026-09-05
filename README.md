# CRDA Research Notebook Compendium

This repository contains a sequence of Jupyter notebooks focused on audio-classification experiments, ranging from dataset preparation to multiple deep-learning model variants and uncertainty-driven CRDA analysis.

## Research Scope

- **Domain**: Spectrogram-based audio classification
- **Pipeline style**: Notebook-first experimental workflow
- **Model families explored**:
  - ResNet baselines (`resnet18`, `resnet34`)
  - DeiT-based transformer modeling
  - CNN-Transformer hybrids
  - CoAtNet + CaiT hybrid variants
  - CRDA-style disagreement/uncertainty analysis

## Notebook Map

| Notebook | Role in pipeline | Key methods/components | Reported validation result* |
|---|---|---|---|
| `Data.ipynb` | Dataset preparation and preprocessing | Raw-to-processed conversion (`RawDataset` → `ProcessedDataset_Balanced`), spectrogram creation, augmentation helpers (`add_noise`, `change_pitch`, `change_speed`) | N/A (preprocessing notebook) |
| `Resnet/Resnet34.ipynb` | Strong CNN baseline training/evaluation | ResNet34, checkpointed training loop, `ReduceLROnPlateau`, cached loading utilities | **Best Val Acc: 91.93%** |
| `resnet18/resnet18.ipynb` | Lightweight CNN baseline | ResNet18, mixed precision (`autocast` + `GradScaler`), cosine scheduler | **Best Val Acc: 91.48%** |
| `Diet/Diet.ipynb` | Transformer baseline | DeiT-small (`timm`), `AdamW`, cosine annealing schedule, checkpointing | **Best Val Acc: 91.93%** |
| `Hybgem/hyb.ipynb` | Hybrid CNN-Transformer architecture | `ResNetDiETHybrid`, transformer encoder on ResNet tokens, cosine schedule | **Best Val Acc: 91.93%** |
| `hyb2/Hybgpt.ipynb` | Updated hybrid experimentation | `ConvDiETHybrid` / `ConvDiETHybrid95`, token generator from ResNet stages, cosine scheduling | **Peak Val Acc in logs: 91.93%** |
| `CiatCoatnet/CoatnetCait.ipynb` | Advanced hybrid optimization experiments | CoAtNet + CaiT hybrid blocks, enhanced architecture variants, label smoothing / focal-loss components | **Peak logged Val: 93.27%** |
| `CRDA.ipynb` | CRDA-oriented post-hoc analysis | `EnhancedHybrid` definition + disagreement/uncertainty plotting and analysis workflow | No explicit training metrics logged in notebook outputs |

\*Validation numbers are taken from notebook output cells currently stored in the repository and reflect logged experiment snapshots.

## Cross-Notebook Findings

1. **Performance ceiling in current logs**  
   The highest recorded validation value appears in the CoAtNet+CaiT line (`93.27%`), suggesting the most benefit from richer hybrid feature fusion.

2. **Stable baseline cluster around ~91.5–91.9%**  
   ResNet34, DeiT, and hybrid variants repeatedly converge around `~91.93%`, indicating a strong but shared plateau under current preprocessing/training settings.

3. **CRDA notebook emphasizes reliability analysis**  
   Beyond top-1 accuracy, `CRDA.ipynb` introduces disagreement-based analysis to study confidence/error behavior, which is useful for trust-aware deployment decisions.

## End-to-End Research Flow

1. Prepare balanced processed spectrogram data (`Data.ipynb`).
2. Train baseline models (`Resnet34`, `resnet18`, `Diet`).
3. Train/iterate hybrid architectures (`Hybgem`, `hyb2`, `CiatCoatnet`).
4. Run CRDA-style disagreement and uncertainty analysis (`CRDA.ipynb`).
5. Compare performance and confidence trends to select deployment candidates.

## Suggested Reproducibility Order

For a clean rerun of the experiments, execute notebooks in this order:

1. `Data.ipynb`
2. `Resnet/Resnet34.ipynb`
3. `resnet18/resnet18.ipynb`
4. `Diet/Diet.ipynb`
5. `Hybgem/hyb.ipynb`
6. `hyb2/Hybgpt.ipynb`
7. `CiatCoatnet/CoatnetCait.ipynb`
8. `CRDA.ipynb`

## Notes

- Some notebooks contain multiple experimental reruns; reported values represent the best visible validation values in committed outputs.
- To make future comparisons stronger, consider exporting each run’s metrics to a single structured table (CSV/JSON) with run ID, seed, split, and checkpoint path.
