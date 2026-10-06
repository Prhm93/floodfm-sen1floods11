# Stage 3 report: U-Net on radar and on optical images

**Project:** FloodFM-Sen1Floods11
**Writing standard:** ASD-STE100 (Simplified Technical English)
**Status:** complete

**Source of the numbers:** all numbers in this report come from the notebook `03_unet.ipynb`. The notebook read them from the results files on Drive (`content/results/`): `otsu_vh.json`, `fixed_vh.json`, `unet_s1_seed0.json` to `unet_s2_seed2.json`, and `stage3_per_run.csv`. The differences in section 5.2 are simple subtractions of the means in section 5.1.

---

## 1. Summary

1. We trained a U-Net 6 times: on radar and on optical images, with 3 seeds each.
2. **Both U-Nets are better than both baselines on the test split.**
3. **Optical is better than radar** on test and on Bolivia. But the labels were made with the optical images. Thus this comparison favours optical (section 7.1).
4. **The radar U-Net is stable on test, but not stable on Bolivia.** One radar seed is below the fixed threshold on Bolivia.
5. The bad radar seed **missed water**. It did not make more false alarms. Three wet Bolivia chips cause most of the difference.
6. **The validation score did not predict the Bolivia result.** The validation chips come from the training events.

---

## 2. What we did

### 2.1 Model and training

| Setting | Value |
|---|---|
| Model | U-Net (segmentation-models-pytorch) |
| Encoder | ResNet-34, **no pretrained weights** |
| Input | Radar: 2 bands (VV, VH). Optical: 13 bands. |
| Output | 2 classes (not water, water) |
| Loss | Cross-entropy + Dice. Label -1 is skipped. |
| Optimiser | AdamW, learning rate 0.001, weight decay 0.0001 |
| Schedule | Cosine, from 0.001 to near 0 over 100 epochs |
| Batch size | 8 (31 steps in each epoch) |
| Epochs | 100 |
| Augmentation | Random horizontal and vertical flips (training only) |
| Mixed precision | Yes (16-bit for most calculations) |
| Seeds | 0, 1, 2 |
| Selected checkpoint | Best validation IoU |

### 2.2 Evaluation

- The same `metrics.py` as the baselines: TP, FP, FN and TN added over all chips, then one IoU and one F1.
- The same missing-input rule as the baselines: where the input has no data, the prediction is "not water".
  - Radar: VH is NaN.
  - Optical: all 13 bands are 0.
- The same results format as the baselines. Each run has a results file with the counts of each chip, each event and each split.
- The Bolivia predictions of each run are saved in `content/predictions/` for the visual demo.

### 2.3 Hardware and time

| Item | Value |
|---|---|
| GPU | Tesla T4, 15.6 GB |
| Peak GPU memory (6-step test, batch 8) | Radar: 2.4 GB. Optical: 2.5 GB. |
| Time for one epoch, with validation | Radar: about 10 s. Optical: about 12.7 s. |
| Data copy from Drive to the Colab disk | 335 s (1,342 files, one time for each session) |

---

## 3. Checks before the long runs

| Check | Result |
|---|---|
| Model fits in GPU memory | Yes, with a large margin |
| Loss after 6 steps is a normal number | Yes (about 1.3), not `nan` |
| Short run of 3 epochs | Correct. Validation IoU: 0.287, 0.583, 0.615. |
| Resume after a stop | The test run continued from its last checkpoint |
| Local copy complete | 446 files in each layer folder, 4 split files |

---

## 4. Training curves

### 4.1 Radar

- The validation IoU goes up fast in the first 2 to 5 epochs.
- After about epoch 50, it stops improving. It stays between about 0.58 and 0.65.
- The training loss continues to go down slowly.
- **The best epochs were 56, 64 and 66.**
- Thus the radar model learns the training chips a little better after epoch 50, but it does not become better on new chips. This is a mild sign of overfitting. It does not change the result, because we keep the best validation epoch.

### 4.2 Optical

- The validation IoU reaches about 0.80 after about 25 epochs.
- It continues to improve slowly until the end.
- **The best epochs were 100, 88 and 81.**
- Thus the optical model could possibly improve with more epochs. We did not change the plan. All runs use the same number of epochs.

### 4.3 Both

The validation IoU jumps a lot from epoch to epoch, mainly in the first half of the training. With only 89 validation chips and a high learning rate at the start, this is normal.

---

## 5. Results

### 5.1 Each run

| Method | Seed | Best epoch | Valid IoU | Test IoU | Test F1 | Bolivia IoU | Bolivia F1 |
|---|---|---|---|---|---|---|---|
| Otsu (VH) | – | – | 0.220 | 0.230 | 0.374 | 0.403 | 0.575 |
| Fixed VH threshold | – | – | 0.492 | 0.524 | 0.688 | 0.531 | 0.694 |
| U-Net radar | 0 | 56 | 0.648 | 0.663 | 0.798 | 0.688 | 0.815 |
| U-Net radar | 1 | 64 | 0.643 | 0.673 | 0.805 | 0.676 | 0.807 |
| U-Net radar | 2 | 66 | 0.648 | 0.665 | 0.799 | **0.507** | 0.672 |
| U-Net optical | 0 | 100 | 0.837 | 0.828 | 0.906 | 0.777 | 0.875 |
| U-Net optical | 1 | 88 | 0.830 | 0.816 | 0.899 | 0.740 | 0.850 |
| U-Net optical | 2 | 81 | 0.828 | 0.817 | 0.899 | 0.781 | 0.877 |

### 5.2 Mean, minimum and maximum of the 3 seeds

| Method | Valid IoU (mean) | Test IoU: mean (min to max) | Bolivia IoU: mean (min to max) |
|---|---|---|---|
| Fixed VH threshold | 0.492 | 0.524 | 0.531 |
| U-Net radar | 0.647 | **0.667** (0.663 to 0.673) | **0.624** (0.507 to 0.688) |
| U-Net optical | 0.831 | **0.820** (0.816 to 0.828) | **0.766** (0.740 to 0.781) |

**Difference from the fixed threshold (mean of 3 seeds):**

| Method | Test | Bolivia |
|---|---|---|
| U-Net radar | +0.143 | +0.093 |
| U-Net optical | +0.296 | +0.235 |

**Caution:** these are means of 3 seeds only. Stage 5 gives the 95% bootstrap intervals. Those intervals show how much the numbers can change by chance.

---

## 6. Findings

### 6.1 On test, both U-Nets are stable

The range of the 3 seeds on test is small: 0.010 for radar and 0.012 for optical.

### 6.2 On Bolivia, the radar U-Net is not stable

The radar Bolivia IoU goes from 0.507 to 0.688. **Seed 2 (0.507) is below the fixed threshold (0.531).**

With one seed only, we could have reported 0.688 or 0.507. Both numbers would be misleading. This is the reason for 3 seeds.

### 6.3 Seed 2 missed water. It did not make more false alarms.

| Radar seed | TP | FP | FN |
|---|---|---|---|
| 0 | 344,348 | 45,787 | 110,516 |
| 1 | 341,533 | 50,159 | 113,331 |
| 2 | 251,547 | 41,711 | **203,317** |

Seed 2 has almost **2 times more missed water** (FN) than seeds 0 and 1. It has **fewer** false alarms. Thus on Bolivia, seed 2 is more "careful": it calls less water.

### 6.4 Three wet chips cause most of the difference

| Chip | Water share | Seed 0 IoU | Seed 1 IoU | Seed 2 IoU |
|---|---|---|---|---|
| Bolivia_129334 | 65.9% | 0.878 | 0.874 | 0.668 |
| Bolivia_314919 | 44.3% | 0.699 | 0.717 | 0.476 |
| Bolivia_432776 | 46.0% | 0.671 | 0.566 | 0.190 |

On the other chips, the three seeds are similar. Thus the failure of seed 2 is concentrated in a few chips with much water.

### 6.5 Validation did not warn us

The best validation IoU of the three radar seeds was almost the same: 0.648, 0.643 and 0.648. But the Bolivia IoU was very different.

**The reason:** the validation chips come from the same 10 events as the training chips. Thus a good validation score shows that the model knows those places. It does not show that the model works on a new place.

**Importance for the project:** this is a direct, small-scale example of the problem of generalisation to a new place.

### 6.6 Some Bolivia chips are difficult for all methods

- **Bolivia_242570** (15.6% water): the fixed threshold gets 0.309. The radar U-Nets get only 0.160, 0.042 and 0.023. Thus on this chip, the U-Net is much worse than one simple threshold. The reason is not known yet.
- **Bolivia_312675** (6.1% water): seeds 0 and 1 get about 0.04. The fixed threshold gets 0.189.
- **Bolivia_76104** and **Bolivia_233925** have no water in the label. All methods get IoU = 0 on them. Thus all methods make some false alarms there.
- **Three chips with very little water** (Bolivia_195474, 0.6%; Bolivia_360519, 1.1%; Bolivia_379434, 4.1%) have low IoU for all methods.

### 6.7 Optical is better than radar on every split

The optical U-Net is better than the radar U-Net on validation, test and Bolivia, for all seeds. On Bolivia, the worst optical seed (0.740) is better than the best radar seed (0.688).

---

## 7. Limits

### 7.1 The labels favour optical

The analysts made the labels with the Sentinel-1 VH band, two Sentinel-2 false-colour images and a Sentinel-2 water map. They corrected the Sentinel-2 water map to make the final label. This has two effects:

1. The labels "see" what Sentinel-2 sees. Thus an optical model has an advantage that comes partly from how the labels were made.
2. Under cloud, the analysts usually marked "no data". We do not score those pixels. Thus **the main weakness of optical images (cloud) is hidden from the scores.**

Thus "optical is better than radar" is correct **only for this dataset and these scores**. In a real flood with heavy cloud, radar can still see. Optical cannot.

### 7.2 Other limits

1. **One training setup.** We did not tune the settings. Other settings can give different results for both inputs.
2. **Fixed number of epochs.** The optical model was still improving at epoch 100.
3. **Bolivia has 15 chips from one event.** A few chips can change the score a lot (section 6.4).
4. **The test split shares events with training.** Only Bolivia tests a new event.
5. **Our numbers are not comparable with the dataset paper.** The paper uses a different model and calculates the mean IoU of chips. We add the pixels first, then calculate one IoU.
6. **A resumed run is not exactly identical to a run with no stop.** No run in Stage 3 stopped. Thus this limit had no effect here.

---

## 8. Decisions for Stage 4

1. **The bars for Prithvi:**
   - Radar U-Net (mean): test 0.667, Bolivia 0.624
   - Optical U-Net (mean): test 0.820, Bolivia 0.766
   - Fixed threshold: test 0.524, Bolivia 0.531
2. **Prithvi uses optical bands.** Thus the fair comparison for Prithvi is the **optical U-Net**.
3. **Prithvi uses the same evaluation:** the same metrics, the same missing-input rule, the same results format and 3 seeds.
4. **Report the Bolivia range, not only the mean.** Section 6.2 shows why.
5. **In Stage 5, look at the difficult Bolivia chips** (section 6.6) for all methods.

---

## 9. Files from this stage

| File | Location |
|---|---|
| `03_unet.ipynb` | `codes/notebooks/` |
| `train.py`, `evaluate.py` | `codes/src/floodfm/` |
| `unet_s1_seed0.json` to `unet_s2_seed2.json` | Drive, `content/results/` |
| `stage3_per_run.csv` | Drive, `content/results/` |
| `unet_<input>_seed<n>/` (last.pt, best.pt, history.json) | Drive, `content/checkpoints/` |
| `unet_<input>_seed<n>_bolivia.npz` | Drive, `content/predictions/` |
| `_test/` (the 3-epoch test run) | Drive, `content/checkpoints/` (can be deleted) |
