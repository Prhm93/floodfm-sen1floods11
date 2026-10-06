# Stage 5 report: how certain are the results?

**Project:** FloodFM-Sen1Floods11
**Writing standard:** ASD-STE100 (Simplified Technical English)
**Status:** complete

**Source of the numbers:** all numbers in this report were calculated by the notebook `05_analysis.ipynb` from the 21 results files on Drive (`content/results/`). The notebook saved four tables on Drive: `final_results.csv`, `paired_differences.csv`, `per_event_test.csv` and `difficult_chips.csv`. The README reads its tables from these files.

---

## 1. Summary

1. We calculated 95% intervals with a chip bootstrap (10,000 samples), and compared pairs of methods on the same random chip lists.
2. **Frozen Prithvi and the band-matched U-Net are not clearly different** with all training labels: on test (+0.005, interval -0.002 to +0.012) and on Bolivia (+0.036, interval -0.014 to +0.155).
3. **We cannot say that Prithvi generalises better to the new event, and we cannot say that it generalises worse.** The Bolivia intervals are wide because Bolivia has only 15 chips.
4. **With 10% of the labels, the U-Net is clearly better on test** (+0.037, interval +0.018 to +0.068), and not clearly different on Bolivia.
5. **With fewer labels, the U-Net lost nothing detectable on test. Frozen Prithvi lost 0.033 on test.** In this setup, frozen Prithvi did not need fewer labels.
6. **Full fine-tuning is not clearly different from frozen** (one seed only).
7. **Optical is better than radar, and every deep model is better than the thresholds.** These differences are clear. The labels favour optical (Stage 3 report, section 7.1).
8. **All methods miss one test chip completely** (`Paraguay_34417`). This is a data problem, not a model problem (section 7).

---

## 2. Method

### 2.1 Chip bootstrap

| Item | Value |
|---|---|
| Samples | 10,000 |
| Resampled unit | One chip, drawn with replacement |
| Score of each sample | Water IoU from the TP, FP and FN summed over the drawn chips |
| For methods with 3 seeds | The IoU of each seed on the same chips, then the mean of the seeds |
| Interval | The middle 95% of the 10,000 values (2.5th to 97.5th percentile) |
| Random generator | Seed 0. All methods use the same 10,000 chip lists. |

### 2.2 Paired differences

For each of the 10,000 chip lists, we calculated the IoU of method A and of method B on the same chips. The interval is the middle 95% of the 10,000 differences A - B.

| Interval of A - B | Meaning |
|---|---|
| Fully above 0 | A is better. The difference is larger than chip chance. |
| Fully below 0 | B is better. |
| Includes 0 | We cannot say which method is better. |

### 2.3 What the interval does not include

1. **Chance from the seeds.** We show it with the seed range (min to max).
2. **Chance from the 10% subset.** We used one fixed subset of 26 chips.
3. **Chips that belong together.** The bootstrap treats the chips as independent. Chips from the same flood event can be similar. Then the real intervals can be wider than the intervals in this report.
4. **Run-to-run variation of the GPU.** The same Prithvi seed gave results that differ by up to 0.008 on Bolivia (Stage 4 report, section 10.9).

---

## 3. Final results table

IoU of the mean of the seeds, with the 95% bootstrap interval and the seed range.

### 3.1 Test split (90 chips, water share 0.125)

| Method | Seeds | IoU | 95% interval | Seed range |
|---|---|---|---|---|
| Otsu (VH) | – | 0.230 | 0.162 to 0.298 | – |
| Fixed VH threshold | – | 0.524 | 0.411 to 0.612 | – |
| U-Net radar, 2 bands | 3 | 0.667 | 0.545 to 0.760 | 0.663 to 0.673 |
| U-Net optical, 13 bands | 3 | 0.820 | 0.735 to 0.879 | 0.816 to 0.828 |
| **U-Net optical, 6 bands** | 3 | **0.829** | 0.752 to 0.882 | 0.827 to 0.832 |
| Prithvi frozen, 6 bands | 3 | 0.824 | 0.750 to 0.876 | 0.822 to 0.826 |
| Prithvi full, 6 bands | 1 | 0.812 | 0.726 to 0.871 | – |
| U-Net optical, 6 bands, 10% labels | 3 | 0.828 | 0.754 to 0.879 | 0.825 to 0.833 |
| Prithvi frozen, 6 bands, 10% labels | 3 | 0.791 | 0.708 to 0.850 | 0.788 to 0.793 |

### 3.2 Bolivia (15 chips, water share 0.159)

| Method | Seeds | IoU | 95% interval | Seed range |
|---|---|---|---|---|
| Otsu (VH) | – | 0.403 | 0.184 to 0.612 | – |
| Fixed VH threshold | – | 0.531 | 0.372 to 0.623 | – |
| U-Net radar, 2 bands | 3 | 0.624 | 0.418 to 0.738 | 0.507 to 0.688 |
| U-Net optical, 13 bands | 3 | 0.766 | 0.599 to 0.849 | 0.740 to 0.781 |
| **U-Net optical, 6 bands** | 3 | **0.782** | 0.652 to 0.859 | 0.760 to 0.795 |
| Prithvi frozen, 6 bands | 3 | 0.746 | 0.518 to 0.842 | 0.729 to 0.760 |
| Prithvi full, 6 bands | 1 | 0.759 | 0.549 to 0.852 | – |
| U-Net optical, 6 bands, 10% labels | 3 | 0.767 | 0.613 to 0.854 | 0.750 to 0.791 |
| Prithvi frozen, 6 bands, 10% labels | 3 | 0.755 | 0.578 to 0.844 | 0.753 to 0.758 |

### 3.3 How to read the tables

- **The Bolivia intervals are much wider than the test intervals.** Bolivia has 15 chips. Test has 90 chips.
- **Most deep models overlap each other on Bolivia.** The table alone cannot rank them. Use the paired differences (section 4).
- **Overlapping intervals do not mean "no difference".** On Bolivia, the radar U-Net (0.418 to 0.738) and the fixed threshold (0.372 to 0.623) overlap. But the paired difference is clear: +0.092, interval +0.028 to +0.142. Both methods go up and down together when the chips change, so the difference is more certain than each score.

---

## 4. Paired differences

A - B, on the same chip lists. "A better" means that the whole interval is above 0.

| Comparison (A against B) | Split | A - B | 95% interval | Result |
|---|---|---|---|---|
| U-Net 6 bands vs Prithvi frozen, all labels | test | +0.005 | -0.002 to +0.012 | Not clear |
| | Bolivia | +0.036 | -0.014 to +0.155 | Not clear |
| U-Net 6 bands vs Prithvi frozen, 10% labels | test | +0.037 | +0.018 to +0.068 | **U-Net better** |
| | Bolivia | +0.012 | -0.036 to +0.063 | Not clear |
| Prithvi full vs frozen (full: 1 seed) | test | -0.012 | -0.034 to +0.005 | Not clear |
| | Bolivia | +0.013 | -0.020 to +0.069 | Not clear |
| U-Net 6 bands vs 13 bands | test | +0.009 | -0.003 to +0.030 | Not clear |
| | Bolivia | +0.016 | +0.002 to +0.061 | 6 bands better (small) |
| U-Net optical vs radar | test | +0.153 | +0.092 to +0.226 | Optical better |
| | Bolivia | +0.142 | +0.057 to +0.238 | Optical better |
| U-Net radar vs fixed threshold | test | +0.143 | +0.110 to +0.182 | U-Net better |
| | Bolivia | +0.092 | +0.028 to +0.142 | U-Net better |
| Fixed threshold vs Otsu | test | +0.294 | +0.206 to +0.375 | Fixed better |
| | Bolivia | +0.128 | -0.040 to +0.264 | Not clear |
| U-Net 6 bands: all labels vs 10% | test | +0.001 | -0.006 to +0.008 | Not clear |
| | Bolivia | +0.015 | +0.003 to +0.043 | All labels better (small) |
| Prithvi frozen: all labels vs 10% | test | +0.033 | +0.016 to +0.062 | **All labels better** |
| | Bolivia | -0.009 | -0.116 to +0.051 | Not clear |

---

## 5. Findings

### 5.1 Main question, half 1: the new event (Bolivia)

**With all labels,** the band-matched U-Net is +0.036 above frozen Prithvi, but the interval is -0.014 to +0.155. **We cannot say which one is better.**

**With 10% of the labels,** the difference is +0.012, interval -0.036 to +0.063. **We cannot say which one is better.**

**Important:** "not clear" does not mean "equal". The intervals are wide. A real difference of 0.03 to 0.05 can exist and stay hidden. What the data does show: **there is no evidence that Prithvi generalises better to the new event.**

**Why the Bolivia upper limit (+0.155) is so large:** a few chips with large differences dominate. For example, `Bolivia_312675` has an IoU of 0.567 for the 6-band U-Net and 0.237 for frozen Prithvi (section 6.2).

### 5.2 Main question, half 2: how many labels

| Model | Test: all labels minus 10% | Bolivia: all labels minus 10% |
|---|---|---|
| U-Net, 6 bands | +0.001 (not clear) | +0.015 (small, clear) |
| Prithvi frozen | **+0.033 (clear)** | -0.009 (not clear) |

- **The U-Net lost nothing detectable on test** with 10% of the labels.
- **Frozen Prithvi lost 0.033 on test.**
- With 10% of the labels, the U-Net is better than Prithvi on test by +0.037 (clear).

**In this setup, frozen Prithvi did not need fewer labels than a U-Net trained from zero.**

**Why the U-Net does well with 26 chips on test:** the test chips come from the same 10 events as the training chips. The 26 chips include chips from each event. Thus the U-Net can learn the look of each event from a few chips. This is not the same as working in a new place.

### 5.3 Full fine-tuning

Full against frozen: -0.012 on test (interval -0.034 to +0.005) and +0.013 on Bolivia (interval -0.020 to +0.069). **No clear difference.** The full run used 2 times more time per epoch, gradient accumulation, and had one seed only.

### 5.4 Number of bands

The 6-band U-Net is not clearly different from the 13-band U-Net on test, and a little better on Bolivia (+0.016, interval +0.002 to +0.061). The lower limit is very close to 0. Treat this as "no sign that more bands help". The earlier hypothesis that extra bands add noise (Stage 4 report, section 7.2) is **not proven**.

### 5.5 Radar and optical, and the thresholds

- Optical is clearly better than radar on both splits (+0.153 and +0.142).
- The radar U-Net is clearly better than the fixed threshold on both splits.
- The fixed threshold is clearly better than Otsu on test. On Bolivia, the difference is not clear.
- **Caution:** the labels were made with the Sentinel-2 images. This favours the optical models.

---

## 6. Per-event and difficult-chip results

### 6.1 IoU of each test event (mean of the seeds)

| Event | Water share | Fixed | U-Net radar | U-Net 13 | U-Net 6 | Prithvi frozen | Prithvi full | U-Net 6, 10% | Prithvi frozen, 10% |
|---|---|---|---|---|---|---|---|---|---|
| Ghana | 0.080 | 0.453 | 0.521 | 0.493 | 0.523 | 0.566 | 0.482 | 0.554 | 0.560 |
| India | 0.213 | 0.593 | 0.704 | 0.902 | 0.898 | 0.889 | 0.890 | 0.886 | 0.859 |
| Mekong | 0.145 | 0.724 | 0.858 | 0.931 | 0.930 | 0.918 | 0.919 | 0.926 | 0.904 |
| Nigeria | 0.162 | 0.708 | 0.885 | 0.938 | 0.936 | 0.911 | 0.933 | 0.934 | 0.923 |
| Pakistan | 0.180 | 0.150 | 0.224 | 0.679 | 0.800 | 0.820 | 0.700 | 0.830 | 0.644 |
| Paraguay | 0.073 | 0.397 | 0.533 | 0.607 | 0.600 | 0.587 | 0.597 | 0.599 | 0.563 |
| Somalia | 0.065 | 0.256 | 0.516 | 0.807 | 0.814 | 0.792 | 0.782 | 0.810 | 0.756 |
| Spain | 0.238 | 0.640 | 0.724 | 0.845 | 0.841 | 0.836 | 0.837 | 0.832 | 0.823 |
| Sri-Lanka | 0.185 | 0.717 | 0.854 | 0.926 | 0.930 | 0.928 | 0.929 | 0.928 | 0.900 |
| USA | 0.031 | 0.379 | 0.550 | 0.804 | 0.804 | 0.787 | 0.799 | 0.785 | 0.765 |

**What the table shows:**

1. **With all labels, the U-Net (6 bands) is better than frozen Prithvi in 8 of 10 events.** Prithvi is better in Ghana and Pakistan. The picture is mixed, as the overall result says.
2. **With 10% of the labels, the U-Net is better than Prithvi in 9 of 10 events.** The largest gap is in **Pakistan** (0.830 against 0.644). The 10% subset keeps only 2 of the 16 Pakistan training chips.
3. **Ghana, Paraguay and USA are hard for all methods.** Ghana and Paraguay have the lowest scores for the deep models (about 0.5 to 0.6).
4. **Pakistan is very hard for radar methods** (fixed 0.150, U-Net radar 0.224), and much easier for optical models (0.68 to 0.83). This agrees with the Stage 2 hypothesis that water in Pakistan gives a stronger VH echo than the fixed threshold expects. **This hypothesis is still not tested.**

Each event has only 4 to 14 test chips. Treat single event values with care.

### 6.2 Difficult chips (mean of the seeds)

| Chip | Split | Water share | Fixed | U-Net radar | U-Net 13 | U-Net 6 | Prithvi frozen | Prithvi full | U-Net 6, 10% | Prithvi frozen, 10% |
|---|---|---|---|---|---|---|---|---|---|---|
| Ghana_866994 | test | 0.902 | 0.564 | 0.622 | 0.452 | 0.463 | 0.510 | 0.396 | 0.506 | 0.534 |
| Paraguay_34417 | test | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Bolivia_242570 | Bolivia | 0.156 | 0.309 | 0.075 | 0.584 | 0.614 | 0.620 | 0.598 | 0.588 | 0.497 |
| Bolivia_312675 | Bolivia | 0.061 | 0.189 | 0.090 | 0.429 | 0.567 | 0.237 | 0.190 | 0.424 | 0.467 |
| Bolivia_129334 | Bolivia | 0.659 | 0.700 | 0.806 | 0.918 | 0.924 | 0.923 | 0.927 | 0.927 | 0.925 |
| Bolivia_314919 | Bolivia | 0.443 | 0.553 | 0.631 | 0.750 | 0.758 | 0.831 | 0.836 | 0.729 | 0.814 |
| Bolivia_432776 | Bolivia | 0.460 | 0.465 | 0.476 | 0.764 | 0.769 | 0.747 | 0.766 | 0.765 | 0.773 |

**What the table shows:**

1. **`Ghana_866994`:** 90% of its scored pixels are water. The optical models score 0.40 to 0.53. The **radar** methods score better (0.564 and 0.622). The optical models miss much of this water. The reason is not known. Possible reasons (cloud, shadow, turbid water) are not tested.
2. **`Bolivia_242570`:** in Stage 3, the radar U-Net was worse than the fixed threshold on this chip. The optical models are much better (0.58 to 0.62). The earlier problem is a **radar problem**.
3. **`Bolivia_312675`** (6% water) is the chip with the largest difference between the two optical model families: 6-band U-Net 0.567, frozen Prithvi 0.237, full Prithvi 0.190. This one chip explains much of the wide Bolivia interval.
4. **The three wet Bolivia chips** (`129334`, `314919`, `432776`), where the radar U-Net seed 2 failed in Stage 3, are easy for all optical models (0.73 to 0.93).

---

## 7. A data problem: `Paraguay_34417`

All methods score 0.000 on this test chip, **also the optical models**. The chip has 50,026 scored pixels, and all of them are labelled water.

**What this shows (inference, not yet checked):** the Stage 2 report said that this chip has no **radar** data. The optical models also get 0. Our missing-input rule predicts "not water" where all 13 Sentinel-2 bands are 0. A model that could see a valid image of an all-water chip would predict water. Thus this chip very probably has **no Sentinel-2 data** either. A label without any input is a data fault.

**Effect:** every method misses these 50,026 water pixels. This is about 2% of the water pixels of the test split. It lowers all test scores a little, in the same way.

**Check:** Cell 6 of `05_analysis.ipynb` (optional) counts the pixels with input data on this chip. If the check confirms the inference, the README must mention it.

---

## 8. Limits

1. The bootstrap does not include the chance from seeds, from the 10% subset, from chips of the same event that belong together, or from GPU run-to-run variation.
2. Bolivia has 15 chips from one event. Differences smaller than about 0.05 cannot be detected there.
3. Prithvi full has one seed. Its interval shows chip chance only.
4. The 10% test used one fraction, one subset, a frozen backbone, and all 89 validation chips.
5. Different training settings for the two model families (Stage 4 report, section 8).
6. The labels were made with the Sentinel-2 images.
7. One test chip has no input data (section 7).

---

## 8b. Statements that the results support, and statements that they do not support

| Statement | Supported? |
|---|---|
| A band-matched U-Net trained from zero matched frozen Prithvi-EO-2.0 on the test events (difference +0.005, interval -0.002 to +0.012). | **Yes** |
| On the unseen Bolivia event, the two models were not clearly different, and the intervals are wide. | **Yes** |
| Prithvi generalises better to a new flood event than a U-Net. | **No.** No evidence. |
| Prithvi generalises worse to a new flood event than a U-Net. | **No.** No evidence. |
| With 10% of the labels, the U-Net was better than frozen Prithvi on the test events. | **Yes** (+0.037, interval +0.018 to +0.068) |
| Foundation models need fewer labels. | **No.** In this setup, frozen Prithvi did not. |
| Foundation models do not need fewer labels. | **Only for this setup:** one dataset, frozen backbone, 10%, 6 bands, one subset. |
| Optical models are better than radar models. | **Only for this dataset:** the labels were made with the optical images. |
| Full fine-tuning is better than frozen. | **No.** Not clear, one seed. |

---

## 9. Decisions for Stage 6

1. **README results table:** read it from `final_results.csv`. Show IoU, the 95% interval and the seed range. Show the water share next to the score.
2. **README text:** use the supported statements in section 8b. Do not write the unsupported statements.
3. **Show the paired differences** (`paired_differences.csv`) for the two main questions.
4. **Mention in the limits:** the Drive incident and the rerun, the data problem of `Paraguay_34417`, and the different training settings.
5. **Demo page:** Bolivia chips, with the true water and the predictions of the main methods (U-Net radar, U-Net 6 bands, Prithvi frozen). Include the difficult chips `Bolivia_312675` and `Bolivia_242570` as examples.

---

## 10. Files from this stage

| File | Location |
|---|---|
| `05_analysis.ipynb` | `codes/notebooks/` |
| `final_results.csv` | Drive, `content/results/` |
| `paired_differences.csv` | Drive, `content/results/` |
| `per_event_test.csv` | Drive, `content/results/` |
| `difficult_chips.csv` | Drive, `content/results/` |
