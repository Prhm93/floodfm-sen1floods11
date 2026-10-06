# Final report: Prithvi-EO-2.0 against U-Nets for flood mapping on Sen1Floods11

**Author:** Parham Imanzadeh Charandabi
**Contact:** pimanzadeh.ch@gmail.com
**Repository:** https://github.com/Prhm93/floodfm-sen1floods11
**Writing standard:** ASD-STE100 (Simplified Technical English)

---

## Summary

This project tests how well the geospatial foundation model **Prithvi-EO-2.0** (300 million weights, used through TerraTorch) maps floods on a flood event that it has never seen, compared with **U-Nets trained from zero** on the same labels. It also tests how many labelled chips each model needs.

The data is the hand-labelled part of **Sen1Floods11** (446 chips from Sentinel-1 radar and Sentinel-2 optical images, 11 flood events). The official splits are used. Bolivia (15 chips) is one event that is never used in training. Each trained model has 3 seeds. Scores are water IoU. Uncertainty is shown with a 95% chip bootstrap and paired differences.

**Main results**

1. With all 252 training chips, frozen Prithvi and a U-Net on the same 6 bands are **not clearly different**: test IoU 0.824 and 0.829 (paired difference +0.005, interval -0.002 to +0.012).
2. On the unseen Bolivia event, they are **not clearly different** either (+0.036, interval -0.014 to +0.155). The interval is wide because Bolivia has 15 chips. There is no evidence that Prithvi generalises better to a new event, and no evidence that it generalises worse.
3. With **10% of the training labels** (26 chips), the U-Net is better on the test events (0.828 against 0.791; +0.037, interval +0.018 to +0.068). On Bolivia, the difference is not clear.
4. With fewer labels, the U-Net lost nothing detectable on test (+0.001, interval -0.006 to +0.008). Frozen Prithvi lost 0.033 (interval +0.016 to +0.062).
5. In this setup, **frozen Prithvi did not need fewer labels than a U-Net trained from zero**.

These results are for one dataset, one set of training settings, one 10% subset of chips, and a frozen backbone. They do not say anything about other foundation models or other datasets.

![Design of the experiment](../figures/fig0_design.png)

---

## 1. Question

> How well does a satellite foundation model work on a flood event it has never seen, compared with a U-Net trained on the same labels, and how many labels does each need?

Foundation models are often described as models that adapt "with minimal supervision" and "generalise across regions". The two halves of the question test these two claims on a small scale:

| Half | Question | Test |
|---|---|---|
| 1 | Does the foundation model work better on a **new** flood event? | The Bolivia split |
| 2 | Does the foundation model need **fewer labels**? | The label-fraction test (10% of the training chips) |

---

## 2. Data

**Sen1Floods11, version 1.1** (Bonafilia et al., 2020). Only the 446 hand-labelled chips are used. Source: https://github.com/cloudtostreet/Sen1Floods11.

- One chip: 512 x 512 pixels, about 10 m per pixel, about 5 km x 5 km.
- Files for each chip: Sentinel-1 radar (VV and VH, in dB), Sentinel-2 optical (13 bands), and a water label (1 water, 0 not water, -1 no data).
- Official splits: train 252, valid 89, test 90, Bolivia 15. Train, valid and test contain the same 10 events. Bolivia is a separate event.
- Water share of the scored pixels: train 9.5%, valid 11.0%, test 12.5%, Bolivia 15.9%.

**Properties of the labels that affect the results**

- They show all water (rivers, lakes, sea and flood water), not only flood water.
- They were made with Sentinel-2 images. This favours models that use Sentinel-2.
- Many chips have large areas with no data. Scores and water shares use only pixels with the label 0 or 1.

---

## 3. Methods

| Method | Description |
|---|---|
| Otsu | Otsu threshold on the Sentinel-1 VH band, one threshold for each chip |
| Fixed VH threshold | One threshold (-23.6 dB) for all chips, chosen on the training chips |
| U-Net radar | ResNet-34 encoder, random start, radar input (2 bands) |
| U-Net optical 13 | The same model, optical input (13 bands) |
| U-Net optical 6 | The same model, the 6 bands of Prithvi (B2, B3, B4, B8A, B11, B12) |
| Prithvi frozen | Pretrained Prithvi-EO-2.0-300M-TL backbone fixed; necks, UNet decoder and head learn (20.3 M of 324 M weights); 6 bands |
| Prithvi full | The same model, all weights learn (324 M); 6 bands; one seed |

The **6-band U-Net** is the control for the main comparison. Prithvi sees 6 bands. A U-Net on 13 bands would see more information.

| Setting | U-Net | Prithvi |
|---|---|---|
| Library | segmentation-models-pytorch | TerraTorch 1.2.13 |
| Loss | Cross-entropy + Dice | Dice |
| Optimiser | AdamW, learning rate 0.001, weight decay 0.0001 | AdamW, learning rate 0.0001, weight decay 0.1 |
| Schedule | Cosine | Halved after 5 epochs without improvement of the validation loss |
| Augmentation | Random flips | Random flips and 90 degree turns |
| Epochs, batch | 100, 8 | 100, 8 (full fine-tuning: 2 x 4 with gradient accumulation) |

Each family uses the settings of its own source. No setting was tuned. The Prithvi settings follow the TerraTorch 1.x example configuration from the Prithvi team, because the NASA configuration file does not run on TerraTorch 1.2.13 (the UperNet setting `scale_modules` is not accepted).

**Rules that are the same for all methods:** the same official splits; seeds 0, 1 and 2; the best epoch chosen by validation IoU; evaluation on full 512 x 512 chips; the same scoring code; and the missing-input rule (a pixel without input data is predicted as "not water").

### The label-fraction test

Prithvi frozen and the 6-band U-Net were also trained with **26 of the 252 training chips**. The chips were chosen once, stratified by event (about 10% of each event, at least one chip), and the same chips were used for both models and all seeds. Each epoch still had 252 samples (the 26 chips are drawn again and again with new random flips or turns). Thus both settings have the same number of learning steps (3,100), and only the number of labelled chips changes.

---

## 4. Evaluation

- **Water IoU** = TP / (TP + FP + FN). Correct land pixels are not counted. Accuracy is not used, because a model that predicts "no water" would get a high accuracy.
- TP, FP and FN are **added over all chips** of a group first. Then one IoU is calculated.
- **95% chip bootstrap:** 10,000 random chip lists drawn with replacement, the same lists for all methods. For methods with 3 seeds, the seed-mean IoU of each sample. Interval: 2.5th to 97.5th percentile.
- **Paired differences:** A minus B on the same chip lists. A difference is "clear" if the whole interval is above or below 0.
- The interval shows chance from the chips only. The tables also give the seed range.

---

## 5. Results

![Water IoU of all methods](../figures/fig1_main_results.png)

### 5.1 Test split (90 chips, water share 0.125)

| Method | Seeds | Water IoU | 95% interval | Seed range |
|---|---|---|---|---|
| Otsu (VH) | 1 | 0.230 | 0.162 to 0.298 | - |
| Fixed VH threshold | 1 | 0.524 | 0.411 to 0.612 | - |
| U-Net radar, 2 bands | 3 | 0.667 | 0.545 to 0.760 | 0.663 to 0.673 |
| U-Net optical, 13 bands | 3 | 0.820 | 0.735 to 0.879 | 0.816 to 0.828 |
| U-Net optical, 6 bands | 3 | 0.829 | 0.752 to 0.882 | 0.827 to 0.832 |
| Prithvi frozen, 6 bands | 3 | 0.824 | 0.750 to 0.876 | 0.822 to 0.826 |
| Prithvi full, 6 bands | 1 | 0.812 | 0.726 to 0.871 | - |
| U-Net optical, 6 bands, 10% labels | 3 | 0.828 | 0.754 to 0.879 | 0.825 to 0.833 |
| Prithvi frozen, 6 bands, 10% labels | 3 | 0.791 | 0.708 to 0.850 | 0.788 to 0.793 |

### 5.2 Bolivia (15 chips, water share 0.159)

| Method | Seeds | Water IoU | 95% interval | Seed range |
|---|---|---|---|---|
| Otsu (VH) | 1 | 0.403 | 0.184 to 0.612 | - |
| Fixed VH threshold | 1 | 0.531 | 0.372 to 0.623 | - |
| U-Net radar, 2 bands | 3 | 0.624 | 0.418 to 0.738 | 0.507 to 0.688 |
| U-Net optical, 13 bands | 3 | 0.766 | 0.599 to 0.849 | 0.740 to 0.781 |
| U-Net optical, 6 bands | 3 | 0.782 | 0.652 to 0.859 | 0.760 to 0.795 |
| Prithvi frozen, 6 bands | 3 | 0.746 | 0.518 to 0.842 | 0.729 to 0.760 |
| Prithvi full, 6 bands | 1 | 0.759 | 0.549 to 0.852 | - |
| U-Net optical, 6 bands, 10% labels | 3 | 0.767 | 0.613 to 0.854 | 0.750 to 0.791 |
| Prithvi frozen, 6 bands, 10% labels | 3 | 0.755 | 0.578 to 0.844 | 0.753 to 0.758 |

### 5.3 Paired differences

| Comparison (A against B) | Split | A minus B | 95% interval | Result |
|---|---|---|---|---|
| Band-matched U-Net vs Prithvi frozen, all labels | test | +0.005 | -0.002 to +0.012 | not clear |
| Band-matched U-Net vs Prithvi frozen, all labels | bolivia | +0.036 | -0.014 to +0.155 | not clear |
| Band-matched U-Net vs Prithvi frozen, 10% labels | test | +0.037 | +0.018 to +0.068 | A better |
| Band-matched U-Net vs Prithvi frozen, 10% labels | bolivia | +0.012 | -0.036 to +0.063 | not clear |
| Prithvi full vs frozen, all labels (full: 1 seed) | test | -0.012 | -0.034 to +0.005 | not clear |
| Prithvi full vs frozen, all labels (full: 1 seed) | bolivia | +0.013 | -0.020 to +0.069 | not clear |
| U-Net 6 bands vs 13 bands | test | +0.009 | -0.003 to +0.030 | not clear |
| U-Net 6 bands vs 13 bands | bolivia | +0.016 | +0.002 to +0.061 | A better |
| U-Net optical vs radar | test | +0.153 | +0.092 to +0.226 | A better |
| U-Net optical vs radar | bolivia | +0.142 | +0.057 to +0.238 | A better |
| U-Net radar vs fixed threshold | test | +0.143 | +0.110 to +0.182 | A better |
| U-Net radar vs fixed threshold | bolivia | +0.092 | +0.028 to +0.142 | A better |
| Fixed threshold vs Otsu | test | +0.294 | +0.206 to +0.375 | A better |
| Fixed threshold vs Otsu | bolivia | +0.128 | -0.040 to +0.264 | not clear |
| U-Net 6 bands: all labels vs 10% | test | +0.001 | -0.006 to +0.008 | not clear |
| U-Net 6 bands: all labels vs 10% | bolivia | +0.015 | +0.003 to +0.043 | A better |
| Prithvi frozen: all labels vs 10% | test | +0.033 | +0.016 to +0.062 | A better |
| Prithvi frozen: all labels vs 10% | bolivia | -0.009 | -0.116 to +0.051 | not clear |

![Paired differences](../figures/fig3_paired_differences.png)

### 5.4 Fewer labels

![Fewer labels](../figures/fig2_label_fraction.png)

### 5.5 Per event and difficult chips

![Per-event test IoU](../figures/fig4_per_event_test.png)

![Difficult chips](../figures/fig5_difficult_chips.png)

---

## 6. Findings

1. **No clear difference between a band-matched U-Net and frozen Prithvi with all labels**, on the test events or on Bolivia. The test interval is narrow (-0.002 to +0.012). The Bolivia interval is wide (-0.014 to +0.155).
2. **With 10% of the labels, the U-Net is better on the test events** and not clearly different on Bolivia. The U-Net is better in 9 of 10 test events. The largest gap is Pakistan (0.830 against 0.644). The subset keeps only 2 of the 16 Pakistan training chips.
3. **The U-Net lost almost nothing with 10% of the labels on test.** The test chips come from the same events as the training chips, and the 26 chips include chips from each event. This does not show that the U-Net works in a new place.
4. **Frozen Prithvi lost 0.033 on test with 10% of the labels.** Its validation IoU stopped at about 0.76 while its training loss became very small. The small decoder learned the 26 chips, but this did not transfer to new chips.
5. **Full fine-tuning is not clearly different from a frozen backbone** (one seed; two times more time for each epoch).
6. **Optical is clearly better than radar** (+0.153 on test, +0.142 on Bolivia). The labels were made with the optical images, so this is partly an effect of the dataset.
7. **The radar U-Net is unstable on a new event.** Its three seeds gave 0.688, 0.676 and 0.507 on Bolivia, but almost the same validation IoU (0.648, 0.643, 0.648). The validation chips come from the training events and did not show the problem.
8. **Number of bands:** the 6-band U-Net is a little better than the 13-band U-Net on Bolivia (+0.016, interval +0.002 to +0.061; the lower limit is almost 0) and not clearly different on test. The idea that extra bands add noise is not proven.
9. **Difficult chips:** on `Ghana_866994` (90% water), the radar methods score better than the optical models; the reason is not known. On `Bolivia_312675` (6% water), the 6-band U-Net scores 0.567 and frozen Prithvi 0.237; this one chip explains much of the wide Bolivia interval.

---

## 7. Limits

1. **The labels favour optical models** and show all water, not only flood water.
2. **Bolivia has 15 chips from one event.** Small differences (below about 0.05) are usually hidden by the wide intervals. The test split uses the same events as training, so it does not test a new place.
3. **Different training settings** for the two model families. No setting was tuned.
4. **The label-fraction test** uses one fraction (10%), one subset, and a frozen backbone. A 25% test was dropped because of the free GPU limit.
5. **All 89 validation chips** were used to choose the best epoch, also in the 10% setting. A real user with few labels would not have such a large labelled validation set.
6. **Prithvi full fine-tuning has one seed** and needed gradient accumulation (2 x 4 chips) to fit in the 15.6 GB of the T4 GPU.
7. **The bootstrap does not include** the chance from the seeds, from the choice of the 10% subset, or from chips of the same event that belong together.
8. **Run-to-run variation of the GPU.** The same Prithvi seed gave results that differ by up to 0.008 on Bolivia between two runs. Differences between methods below about 0.01 are within this noise.
9. **A data problem.** On the test chip `Paraguay_34417` (50,026 scored pixels, all labelled water), every method scores 0, also the optical models. The chip probably has no Sentinel-1 and no Sentinel-2 data (to be confirmed). About 2% of the test water pixels are lost for all methods.
10. **The scores cannot be compared with the scores in the Sen1Floods11 paper.** That paper uses a different model and the mean of the IoU of each chip.
11. **One dataset only.**

---

## 8. A data-handling incident

After the label-fraction runs, three results files were missing on Google Drive (`unet_s2_6_frac10_seed2`, `prithvi_frozen_frac10_seed1`, `prithvi_frozen_frac10_seed2`), although the notebook had printed their scores. Colab saves files to Drive in the background, and the server was removed before all uploads ended.

A re-evaluation of the saved checkpoint of `prithvi_frozen_frac10_seed1` gave very low scores (test 0.552, Bolivia 0.434). The printed checkpoint epoch was 1, not the expected 56: Drive had received only the first version of `best.pt`. The result was deleted. The three runs were trained again, and **all numbers in this report come from the second runs.**

| Run | First run (test / Bolivia) | Second run (test / Bolivia) |
|---|---|---|
| U-Net, seed 2 | 0.825 / 0.791 | 0.825 / 0.791 |
| Prithvi frozen, seed 1 | 0.796 / 0.761 | 0.793 / 0.753 |
| Prithvi frozen, seed 2 | 0.790 / 0.748 | 0.792 / 0.755 |

The U-Net repeated exactly. The Prithvi runs differ by up to 0.008 with the same seed.

**Rules that followed:** run `drive.flush_and_unmount()` at the end of a long session; check new files in the Drive folder in a browser; print the epoch of a checkpoint before any re-evaluation.

---

## 9. Reproducibility

- **Environment:** Google Colab (free), Tesla T4 GPU (15.6 GB), Python 3.13, PyTorch 2.11.0 (CUDA 13.0 build), TerraTorch 1.2.13, numpy 2.5.3. The version of segmentation-models-pytorch was not recorded.
- **Code:** `src/floodfm/` (data, dataset, metrics, baselines, train, evaluate, prithvi) and the notebooks `00` to `05`. The README gives the order of the steps.
- **Results:** `results/` has the four CSV tables and the 21 results files (JSON, with the counts of each chip). Every number in the README and in this report comes from these files.
- **Figures and tables:** `python scripts/make_figures.py` and `python scripts/make_tables.py`.
- **Time on the T4 for one epoch:** U-Net radar 10 s, U-Net optical 13 bands 13 s, U-Net 6 bands 17 s, Prithvi frozen 31 s, Prithvi full 65 s.

---

## 10. Conclusion

In this small controlled test, a U-Net trained from zero on the same 6 bands matched frozen Prithvi-EO-2.0 on known and unseen flood events. With 10% of the labels, the U-Net was better on known events. The data gives no evidence that frozen Prithvi generalises better to a new event, or that it needs fewer labels. The Bolivia intervals are wide, so small differences stay hidden. A full test of foundation-model adaptation needs more events for the unseen test, more label fractions and subsets, and fully fine-tuned models.

---

## Appendix A: each run

| Method | Seed | Best epoch | Test IoU | Test F1 | Bolivia IoU | Bolivia F1 |
|---|---|---|---|---|---|---|
| Otsu | - | - | 0.230 | 0.374 | 0.403 | 0.575 |
| Fixed VH threshold | - | - | 0.524 | 0.688 | 0.531 | 0.694 |
| U-Net radar | 0 | 56 | 0.663 | 0.798 | 0.688 | 0.815 |
| U-Net radar | 1 | 64 | 0.673 | 0.805 | 0.676 | 0.807 |
| U-Net radar | 2 | 66 | 0.665 | 0.799 | 0.507 | 0.672 |
| U-Net optical 13 | 0 | 100 | 0.828 | 0.906 | 0.777 | 0.875 |
| U-Net optical 13 | 1 | 88 | 0.816 | 0.899 | 0.740 | 0.850 |
| U-Net optical 13 | 2 | 81 | 0.817 | 0.899 | 0.781 | 0.877 |
| U-Net optical 6 | 0 | 96 | 0.827 | 0.905 | 0.760 | 0.864 |
| U-Net optical 6 | 1 | 82 | 0.829 | 0.907 | 0.791 | 0.883 |
| U-Net optical 6 | 2 | 87 | 0.832 | 0.908 | 0.795 | 0.886 |
| Prithvi frozen | 0 | 78 | 0.826 | 0.905 | 0.729 | 0.843 |
| Prithvi frozen | 1 | 91 | 0.825 | 0.904 | 0.760 | 0.864 |
| Prithvi frozen | 2 | 97 | 0.822 | 0.902 | 0.749 | 0.857 |
| Prithvi full | 0 | 82 | 0.812 | 0.896 | 0.759 | 0.863 |
| U-Net optical 6, 10% | 0 | 48 | 0.833 | 0.909 | 0.750 | 0.857 |
| U-Net optical 6, 10% | 1 | 20 | 0.827 | 0.905 | 0.760 | 0.864 |
| U-Net optical 6, 10% | 2 | 17 | 0.825 | 0.904 | 0.791 | 0.883 |
| Prithvi frozen, 10% | 0 | 46 | 0.788 | 0.881 | 0.758 | 0.862 |
| Prithvi frozen, 10% | 1 | 58 | 0.793 | 0.884 | 0.753 | 0.859 |
| Prithvi frozen, 10% | 2 | 40 | 0.792 | 0.884 | 0.755 | 0.860 |

## Appendix B: the stage reports

Each stage has its own report in this folder: `Stage0-1_Report.md` (workspace, data, first look), `Stage2_Report.md` (data code and baselines), `Stage3_Report.md` (U-Net on radar and optical), `Stage4_Report.md` (Prithvi, the 6-band U-Net, the label-fraction test, with the facts from the official sources) and `Stage5_Report.md` (bootstrap and paired differences).

## Sources

- Sen1Floods11 dataset: https://github.com/cloudtostreet/Sen1Floods11
- Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020). Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms for Sentinel-1. CVPR Workshops. https://openaccess.thecvf.com/content_CVPRW_2020/papers/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.pdf
- Prithvi-EO-2.0-300M-TL fine-tuned on Sen1Floods11 (model card): https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11
- NASA configuration (does not run on TerraTorch 1.2.13): https://github.com/NASA-IMPACT/Prithvi-EO-2.0/blob/main/configs/sen1floods11.yaml
- TerraTorch 1.x example configuration used here: https://github.com/blumenstiel/TerraTorch-Examples/blob/main/configs/prithvi_v2_eo_300_tl_unet_sen1floods11.yaml
- TerraTorch documentation: https://ibm.github.io/terratorch
