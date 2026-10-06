# Stage 4 report: Prithvi-EO-2.0 and a band-matched U-Net

**Project:** FloodFM-Sen1Floods11
**Writing standard:** ASD-STE100 (Simplified Technical English)
**Status:** complete (including the label-fraction test)

**Source of the numbers:** all scores in this report come from the results files on Drive (`content/results/`): `prithvi_frozen_seed0.json` to `prithvi_frozen_seed2.json`, `prithvi_full_seed0.json`, `unet_s2_6_seed0.json` to `unet_s2_6_seed2.json`, and the label-fraction files `unet_s2_6_frac10_seed0.json` to `unet_s2_6_frac10_seed2.json` and `prithvi_frozen_frac10_seed0.json` to `prithvi_frozen_frac10_seed2.json`. Three label-fraction runs were trained again after a Drive sync loss (section 10.9); the numbers come from the rerun files. The training times and memory values come from the notebook output and from `history.json` in each checkpoint folder. The means in section 6.2 are simple means of the three seeds.

---

## 1. Summary

1. We tested the foundation model **Prithvi-EO-2.0-300M-TL** through **TerraTorch**, with the backbone frozen (3 seeds) and fully fine-tuned (1 seed).
2. We also trained a **U-Net on the same 6 bands** as Prithvi (3 seeds). This makes the comparison fair.
3. **With all training labels, Prithvi did not give a higher score than the band-matched U-Net,** on the test split or on the unseen Bolivia event.
4. **Prithvi trained faster at the start and more stably,** but its final scores were not higher.
5. **Full fine-tuning did not improve on the frozen backbone,** and it needed two times more time and gradient accumulation.
6. **With only 10% of the training chips (26 of 252), the U-Net was clearly better than the frozen Prithvi on test, and about equal on Bolivia.** Thus, in this setup, the foundation model did not need fewer labels (section 10).
7. **These are results before the bootstrap intervals.** Stage 5 decides which differences are larger than chance.

---

## 2. Facts from the official sources

### 2.1 Model card

Source: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11

- The pretrained Prithvi-EO-2.0-300M-TL model is fine-tuned to segment floods on Sentinel-2 images from Sen1Floods11.
- It uses six bands: Blue, Green, Red, Narrow NIR, SWIR and SWIR 2.
- Labels: 0 = no water, 1 = water or flood, -1 = no data or cloud. These are the same labels as in our project.
- The model was pretrained with 4 time steps. For Sen1Floods11, it uses one time step.
- The card points to the NASA configuration file for fine-tuning (section 2.2).

### 2.2 NASA configuration file

Source: https://github.com/NASA-IMPACT/Prithvi-EO-2.0/blob/main/configs/sen1floods11.yaml

| Setting | Value |
|---|---|
| Image values | Multiplied by 0.0001 |
| No-data input | Replaced with 0 |
| Image size | Resized to 224 × 224 |
| Augmentation | Horizontal and vertical flips |
| Decoder | UperNet, 256 channels, head dropout 0.1 |
| Backbone layers for the 300M model | 5, 11, 17, 23 |
| Loss | Cross-entropy, label -1 ignored |
| Optimiser | AdamW, learning rate 0.00005, weight decay 0.05 |
| Schedule | Cosine over 50 epochs |
| Epochs, batch | 50 epochs (early stopping after 20 checks), batch 16 |
| Precision | 16-bit mixed |
| Backbone frozen | No |

### 2.3 TerraTorch documentation

Source: TerraTorch quick start (ibm.github.io/terratorch)

- TerraTorch is tested for Python 3.10 to 3.12.
- The quick start lists the Sen1Floods11 bands as positions 1, 2, 3, 8, 11 and 12. In our 13-band files, these are B2, B3, B4, B8A, B11 and B12.

### 2.4 The NASA file does not run on the current TerraTorch

- Colab gave TerraTorch **1.2.13**.
- The NASA file uses the UperNet setting `scale_modules`. TerraTorch 1.2.13 does not accept this setting. The model build stopped with an error.
- On the model card discussion page, a member of the Prithvi team says that some configuration keywords changed with TerraTorch 1.0. He points to his own example configurations for TerraTorch 1.x.

### 2.5 TerraTorch 1.x example configuration (used in this project)

Source: https://github.com/blumenstiel/TerraTorch-Examples/blob/main/configs/prithvi_v2_eo_300_tl_unet_sen1floods11.yaml

| Setting | Value |
|---|---|
| Backbone | prithvi_eo_v2_300_tl, pretrained |
| Image size | 512 × 512 (no resize) |
| Location and time inputs | Not used |
| Necks | SelectIndices [5, 11, 17, 23], ReshapeTokensToImage, LearnedInterpolateToPyramidal |
| Decoder | UNet decoder, channels 512, 256, 128, 64 |
| Head dropout | 0.1 |
| Loss | Dice, label -1 ignored |
| Optimiser | AdamW, learning rate 0.0001, weight decay 0.1 |
| Schedule | Learning rate halved after 5 epochs with no improvement in validation loss |
| Epochs, batch | 100 epochs, batch 8 |
| Augmentation | D4 (flips and 90° rotations) |
| Normalisation | Six means and six standard deviations given in the file |
| Precision | 16-bit mixed |

**Difference in band order:** this file lists 12 dataset bands. Our files have 13 bands. We selected the six bands by their positions in our files (1, 2, 3, 8, 11, 12), as the dataset page and the TerraTorch quick start say.

---

## 3. What we did

### 3.1 Software on Colab

| Item | Version |
|---|---|
| Python | 3.13 |
| TerraTorch | 1.2.13 |
| PyTorch | 2.11.0 (CUDA 13.0) |
| numpy | 2.5.3 |

The installation gave one warning: the library numba wants an older numpy. The project does not use numba. The warning had no effect.

### 3.2 Runs

| Run | Seeds | Input | Trainable parameters |
|---|---|---|---|
| Prithvi frozen | 0, 1, 2 | 6 bands | 20.3 M of 324 M |
| Prithvi full | 0 | 6 bands | 324.2 M of 324 M |
| U-Net, 6 bands | 0, 1, 2 | 6 bands | about 24 M |

### 3.3 Project rules that we kept for all runs

- Best checkpoint by **validation IoU**.
- Evaluation on full **512 × 512** chips.
- **Missing-input rule:** where all 13 Sentinel-2 bands are 0, the prediction is "not water".
- The same scores (`metrics.py`) and the same results format as all other methods.

### 3.4 The 6-band U-Net

The 6-band U-Net uses the **same settings as the 13-band optical U-Net** from Stage 3. Only the input bands change. Thus:

- **6-band U-Net against 13-band U-Net** shows the effect of the number of bands.
- **6-band U-Net against Prithvi** compares two models with the same input.

---

## 4. Memory and time on the T4

| Run | Peak memory (memory test, batch 8) | Time for one epoch (real run) |
|---|---|---|
| Prithvi frozen | 2.8 GB | about 32 s |
| Prithvi full | 14.5 GB | about 65 s (with accumulation) |
| U-Net, 6 bands | – | about 17 s |

### 4.1 The full run needed gradient accumulation

- The memory test with random input fitted at batch 8 (14.5 GB of 15.6 GB).
- The real training run at batch 8 stopped with "out of memory", also on a GPU with 3 MiB in use before the start.
- **Solution:** gradient accumulation. The model takes 4 chips at a time and updates its weights after 2 groups of 4. Thus each weight update still uses 8 chips (31 updates for each epoch).
- **One small difference:** the Dice loss is calculated on each group of 4 chips, not on all 8 together.

### 4.2 Disconnections

- Two times, VS Code lost its connection to the Colab server during a run.
- Both times, the server continued the run. The checkpoints and `history.json` on Drive showed the real progress.
- No run had to start again from epoch 1.

---

## 5. Training curves

- **Prithvi frozen:** validation IoU about 0.60 to 0.63 after epoch 1. The curve was smooth. Best epochs: 78, 91 and 97 (best valid IoU 0.832, 0.830 and 0.831).
- **Prithvi full:** validation IoU 0.66 after epoch 1. Best epoch: 82 (best valid IoU 0.836).
- **U-Net, 6 bands:** validation IoU 0.50 to 0.73 after epoch 1. The curve jumped a lot from epoch to epoch (for example, seed 0 fell to 0.519 at epoch 13). Best valid IoU: 0.829, 0.835 and 0.835.
- **The learning-rate schedule of Prithvi worked as designed.** It halved the learning rate several times, each time after 5 epochs with no improvement in validation loss.

---

## 6. Results

### 6.1 Each run

| Method | Seed | Best epoch | Valid IoU | Test IoU | Test F1 | Bolivia IoU | Bolivia F1 |
|---|---|---|---|---|---|---|---|
| Prithvi frozen | 0 | 78 | 0.832 | 0.826 | 0.905 | 0.729 | 0.843 |
| Prithvi frozen | 1 | 91 | 0.830 | 0.825 | 0.904 | 0.760 | 0.864 |
| Prithvi frozen | 2 | 97 | 0.831 | 0.822 | 0.902 | 0.749 | 0.857 |
| Prithvi full | 0 | 82 | 0.836 | 0.812 | 0.896 | 0.759 | 0.863 |
| U-Net, 6 bands | 0 | 96 | 0.829 | 0.827 | 0.905 | 0.760 | 0.864 |
| U-Net, 6 bands | 1 | 82 | 0.835 | 0.829 | 0.907 | 0.791 | 0.883 |
| U-Net, 6 bands | 2 | 87 | 0.835 | 0.832 | 0.908 | 0.795 | 0.886 |

### 6.2 All methods (mean of the seeds, min to max)

| Method | Input | Seeds | Test IoU | Bolivia IoU |
|---|---|---|---|---|
| Otsu | Radar VH | – | 0.230 | 0.403 |
| Fixed VH threshold | Radar VH | – | 0.524 | 0.531 |
| U-Net | Radar, 2 bands | 3 | 0.667 (0.663 to 0.673) | 0.624 (0.507 to 0.688) |
| U-Net | Optical, 13 bands | 3 | 0.820 (0.816 to 0.828) | 0.766 (0.740 to 0.781) |
| U-Net | Optical, 6 bands | 3 | **0.829** (0.827 to 0.832) | **0.782** (0.760 to 0.795) |
| Prithvi frozen | Optical, 6 bands | 3 | 0.824 (0.822 to 0.826) | 0.746 (0.729 to 0.760) |
| Prithvi full | Optical, 6 bands | 1 | 0.812 | 0.759 |

---

## 7. Findings

### 7.1 With the same 6 bands, the U-Net is not worse than Prithvi

- **Test:** the U-Net mean (0.829) is 0.005 higher than the frozen Prithvi mean (0.824).
- **Bolivia:** the U-Net mean (0.782) is 0.036 higher. The worst U-Net seed (0.760) equals the best frozen Prithvi seed (0.760).

Thus, **in this setup, with all training labels, the pretraining did not give a higher score.** **Update after Stage 5:** the paired bootstrap shows that these differences are **not larger than chance** (test +0.005, interval -0.002 to +0.012; Bolivia +0.036, interval -0.014 to +0.155). The correct statement is: no evidence that Prithvi is better, and no evidence that it is worse. See the Stage 5 report.

### 7.2 Fewer bands did not make the U-Net worse

The 6-band U-Net is a little better than the 13-band U-Net (test 0.829 against 0.820; Bolivia 0.782 against 0.766).

**Hypothesis (not tested):** some of the 7 extra bands carry information about the air, not the ground (for example B9, water vapour, and B10, cirrus). They can add noise.

**Consequence:** the 13-band U-Net did **not** have an unfair advantage over Prithvi because of its extra bands.

### 7.3 Prithvi trains faster at the start and more stably

- After epoch 1, the frozen Prithvi had a validation IoU of about 0.60 to 0.63. The 6-band U-Net had 0.50 to 0.73, and its curve jumped a lot.
- By about epoch 15, both reached about 0.79.
- The Prithvi curve stayed smooth until the end.

This is a real advantage of pretraining, but it did not give a higher final score with all labels.

### 7.4 Full fine-tuning did not improve on the frozen backbone

- On test, the full run (0.812) is below all three frozen seeds (0.822 to 0.826).
- On Bolivia, the full run (0.759) is at the top of the frozen range (0.729 to 0.760).
- It took about two times longer for each epoch, and it needed gradient accumulation.
- **It has one seed only.** Thus we cannot say whether its lower test score is real or a less lucky seed.

### 7.5 One difficult test chip: Ghana_866994

On test, the full Prithvi has a low IoU for Ghana (0.482). Most of this comes from one chip, `Ghana_866994`: the model missed 59,845 water pixels and found 46,126. Stage 5 compares this chip across all methods.

---

## 8. Limits

1. **The main runs used all training labels.** The label-fraction test (section 10) tests the claim about "minimal supervision", but only at one fraction (10%) and with one subset of chips.
2. **Prithvi and the U-Net use different training settings,** as given by their sources: loss (Dice against cross-entropy + Dice), augmentation (D4 against flips), schedule (plateau against cosine) and normalisation values.
3. **The full fine-tuning run has one seed.**
4. **The full run used gradient accumulation** (2 × 4 chips). The Dice loss was calculated on groups of 4 chips.
5. **We did not tune any settings.** Other settings can change the results for all models.
6. **The labels were made with the Sentinel-2 images.** All optical models get an advantage from this, compared with radar (see the Stage 3 report).
7. **Bolivia has 15 chips from one event.** Its scores are less certain than the test scores.
8. **The NASA configuration file could not run on TerraTorch 1.2.13.** We used the newer example configuration from the Prithvi team instead.

---

## 9. Decisions for Stage 5

1. Calculate **95% bootstrap intervals** for all methods, from the counts of each chip in the results files.
2. Test whether these differences are larger than chance:
   - 6-band U-Net against Prithvi frozen (test and Bolivia)
   - 6-band U-Net against 13-band U-Net
3. Compare the difficult chips across all methods: `Ghana_866994`, `Bolivia_242570`, `Bolivia_312675` and the three wet Bolivia chips from the Stage 3 report.
4. Report the full Prithvi run as **one seed**, without an interval over seeds.
5. **The label-fraction test at 10% is complete** (section 10). The test at 25% is dropped because of the free GPU limit. The report states this.

---

## 10. The label-fraction test (10% of the training chips)

### 10.1 The question

A common claim is that foundation models adapt "with minimal supervision". That means: with few labels. This test asks: **with few labelled training chips, does Prithvi do better than a U-Net from zero?**

### 10.2 The design

| Item | Value |
|---|---|
| Training chips | **26 of 252** (about 10%) |
| Selection | Stratified by event: about 10% of the chips of each event, at least one chip for each event. Subset seed 0. |
| Same chips for all runs | Yes. Both models and all 3 seeds use the same 26 chips. |
| List of the chips | `content/results/subset_frac10.json` on Drive |
| Samples in one epoch | **252**, the same as with all chips. The 26 chips are drawn again and again, each time with a new random flip or turn. |
| Learning steps | 3,100 (100 epochs × 31 steps), **the same as the full runs** |
| Validation, test, Bolivia | Unchanged (89, 90 and 15 chips) |
| Models | U-Net on 6 bands, and Prithvi frozen |
| Settings of each model | Unchanged from the full runs |
| Seeds | 0, 1, 2 |

**Why the same number of learning steps:** with 26 chips and "100 passes", each model would get only about 300 learning steps instead of 3,100. Then a low score could mean "too few steps", not "too few labels".

**Expected chips for each event, from the rule:**

| Event | Training chips | Chips kept |
|---|---|---|
| Ghana | 31 | 3 |
| India | 40 | 4 |
| Mekong | 18 | 2 |
| Nigeria | 10 | 1 |
| Pakistan | 16 | 2 |
| Paraguay | 39 | 4 |
| Somalia | 15 | 2 |
| Spain | 18 | 2 |
| Sri-Lanka | 24 | 2 |
| USA | 41 | 4 |
| **Total** | **252** | **26** |

The total of 26 was confirmed by the notebook. The exact list of chips is in `subset_frac10.json`.

### 10.3 Time on the T4

| Run | Time for one epoch | One run (100 epochs and evaluation) |
|---|---|---|
| U-Net, 10% | about 16 to 19 s | about 32 min |
| Prithvi frozen, 10% | about 28 to 33 s | about 55 min |

The epoch times are the same as in the full runs. This confirms that the "same number of steps" rule worked.

### 10.4 Each run

| Model | Seed | Best epoch | Best valid IoU | Test IoU | Test F1 | Bolivia IoU | Bolivia F1 |
|---|---|---|---|---|---|---|---|
| U-Net, 6 bands, 10% | 0 | 48 | 0.828 | 0.833 | 0.909 | 0.750 | 0.857 |
| U-Net, 6 bands, 10% | 1 | 20 | 0.824 | 0.827 | 0.905 | 0.760 | 0.864 |
| U-Net, 6 bands, 10% | 2 | 17 | 0.822 | 0.825 | 0.904 | 0.791 | 0.883 |
| Prithvi frozen, 10% | 0 | 46 | 0.763 | 0.788 | 0.881 | 0.758 | 0.862 |
| Prithvi frozen, 10% | 1 | 58 | 0.762 | 0.793 | 0.884 | 0.753 | 0.859 |
| Prithvi frozen, 10% | 2 | 40 | 0.757 | 0.792 | 0.884 | 0.755 | 0.860 |

Prithvi seeds 1 and 2 and U-Net seed 2 come from the rerun (section 10.9).

### 10.5 All chips against 10% of the chips (mean of 3 seeds, min to max)

| Model | Training chips | Test IoU | Bolivia IoU |
|---|---|---|---|
| U-Net, 6 bands | 252 | 0.829 (0.827 to 0.832) | 0.782 (0.760 to 0.795) |
| Prithvi frozen | 252 | 0.824 (0.822 to 0.826) | 0.746 (0.729 to 0.760) |
| **U-Net, 6 bands** | **26** | **0.828** (0.825 to 0.833) | **0.767** (0.750 to 0.791) |
| **Prithvi frozen** | **26** | **0.791** (0.788 to 0.793) | **0.755** (0.753 to 0.758) |

**Change from 252 chips to 26 chips (difference of the means):**

| Model | Test | Bolivia |
|---|---|---|
| U-Net, 6 bands | -0.001 | -0.015 |
| Prithvi frozen | -0.033 | +0.009 |

### 10.6 Findings

**1. On test, with few labels, the U-Net is better.**
All three U-Net seeds (0.825 to 0.833) are above all three Prithvi seeds (0.788 to 0.793). The paired bootstrap agrees: +0.037, interval +0.018 to +0.068 (Stage 5 report).

**2. On Bolivia, with few labels, the two models are about equal.**
The paired difference is +0.012, interval -0.036 to +0.063. The interval includes 0.

**3. The U-Net lost nothing detectable on test with 10% of the chips.**
Paired difference, all labels minus 10%: +0.001, interval -0.006 to +0.008. On Bolivia, it lost a small amount: +0.015, interval +0.003 to +0.043.
The test chips come from the same 10 events as the training chips, and the 26 chips include chips from each event.

**4. Prithvi lost on test, but not detectably on Bolivia.**
On test: +0.033, interval +0.016 to +0.062. On Bolivia: -0.009, interval -0.116 to +0.051. This interval is very wide, so the test cannot show a small effect. The Bolivia seeds of Prithvi are close to each other (range 0.005, against 0.041 for the U-Net), but this is seed variation only; it does not show that the score is certain.

**5. Prithvi's validation IoU stopped at about 0.76.**
Its training loss became very small (about 0.04), but its validation IoU stayed at about 0.75 to 0.76 from about epoch 30. Thus the small decoder learned the 26 chips well, but this did not transfer to new chips. With a frozen backbone, only the decoder can adapt.

**6. The U-Net overfitted, but the best-checkpoint rule protected it.**
The U-Net reached its best validation IoU early (epochs 17, 20 and 48). After that, its validation IoU fell slowly to about 0.77 to 0.79, while its training loss continued to go down. The evaluation used the best epoch.

### 10.7 Answer to half 2 of the main question

**In this setup, the frozen Prithvi did not need fewer labels than a U-Net trained from zero.** With 26 labelled chips, the U-Net was better on the known events (test) and about equal on the new event (Bolivia).

### 10.8 Limits of this test

1. **One fraction only (10%).** The test at 25% was dropped because of the free GPU limit.
2. **One subset of chips.** Another set of 26 chips could give different results. The bootstrap does not include this source of chance.
3. **The backbone was frozen.** A fully fine-tuned Prithvi was not tested with few labels.
4. **All 89 validation chips were used to select the best epoch.** In a real situation with few labels, a large labelled validation set would not exist. Thus both models had an advantage that a real user would not have. The U-Net probably gained more, because its validation curve jumped more.
5. **The same settings as the full runs.** We did not tune any setting for the small training set.
6. **Bolivia has 15 chips.** Its intervals are wide, so small differences (below about 0.05) are usually hidden there.

### 10.9 Incident: three runs were lost and trained again

**What happened.** After the last label-fraction run, three results files were not on Drive: `unet_s2_6_frac10_seed2`, `prithvi_frozen_frac10_seed1` and `prithvi_frozen_frac10_seed2`. The notebook had printed their scores, but the files had not reached Drive.

**Cause.** Colab saves a file to Drive in two steps. The file first goes to a temporary store on the Colab server. A background program uploads it to Drive later. During the Prithvi seed 1 run, the upload stopped. The training continued to "save" into the temporary store. Then the server was removed, and the files that had not been uploaded were lost.

**A wrong recovery.** For Prithvi seed 1, a checkpoint folder existed on Drive. We evaluated its `best.pt` on a CPU. The notebook printed `Best checkpoint: epoch 1`, not the expected epoch 56. Drive had received only the first version of `best.pt`. The CPU evaluation gave test 0.552 and Bolivia 0.434. These numbers are not valid. We deleted the file and the predictions.

**Solution.** We deleted the three incomplete checkpoint folders and trained the three runs again on the GPU. All numbers in section 10 come from the rerun files. The first-run numbers, which came from the notebook output only, are not used.

**Difference between the first run and the rerun with the same seed:**

| Run | First run (test / Bolivia) | Rerun (test / Bolivia) |
|---|---|---|
| U-Net, seed 2 | 0.825 / 0.791 | 0.825 / 0.791 |
| Prithvi frozen, seed 1 | 0.796 / 0.761 | 0.793 / 0.753 |
| Prithvi frozen, seed 2 | 0.790 / 0.748 | 0.792 / 0.755 |

The U-Net repeated exactly. The Prithvi runs differ by up to 0.008 on Bolivia, also with the same seed. A GPU does some calculations in a different order from one run to the next. Thus **differences between methods smaller than about 0.01 are within the run-to-run variation of Prithvi.**

**Rules for the future.**

1. At the end of every long session, run `drive.flush_and_unmount()` before you close the server.
2. After each important run, open the Drive folder **in the browser** and check that the new results file exists.
3. Before a re-evaluation, print the epoch of the checkpoint and compare it with the training log.

---

## 11. Files from this stage

| File | Location |
|---|---|
| `04_prithvi.ipynb` | `codes/notebooks/` |
| `03_unet.ipynb` (Cell 11: 6-band U-Net) | `codes/notebooks/` |
| `prithvi.py` | `codes/src/floodfm/` |
| `dataset.py`, `train.py` (input type `s2_6`) | `codes/src/floodfm/` |
| `prithvi_frozen_seed0.json` to `prithvi_frozen_seed2.json` | Drive, `content/results/` |
| `prithvi_full_seed0.json` | Drive, `content/results/` |
| `unet_s2_6_seed0.json` to `unet_s2_6_seed2.json` | Drive, `content/results/` |
| Checkpoint folders (`last.pt`, `best.pt`, `history.json`) | Drive, `content/checkpoints/` |
| `unet_s2_6_frac10_seed0.json` to `unet_s2_6_frac10_seed2.json` | Drive, `content/results/` |
| `prithvi_frozen_frac10_seed0.json` to `prithvi_frozen_frac10_seed2.json` | Drive, `content/results/` |
| `subset_frac10.json` (the 26 training chips) | Drive, `content/results/` |
| Bolivia predictions (`.npz`), all runs | Drive, `content/predictions/` |
