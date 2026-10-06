# Stage 2 report: data module, baselines and dataset

**Project:** FloodFM-Sen1Floods11
**Writing standard:** ASD-STE100 (Simplified Technical English)
**Status:** complete

**Source of the numbers:** all numbers in this report come from the notebook `02_otsu_baseline.ipynb`. The notebook calculated them and saved them in the results files on Drive (`otsu_vh.json`, `fixed_vh.json`, `norm_stats.json`). The README table will read the numbers from those files with code. Nobody types a score by hand in the README.

---

## 1. Summary

1. We wrote four code modules. All notebooks now read the data in the same way.
2. We tested two baselines that do not use deep learning.
3. The Otsu baseline is weak. It makes many false alarms on chips with little or no water.
4. A fixed VH threshold, tuned on the training set, is more than 2 times better on the test set.
5. **The fixed threshold is the bar for all models.** Test IoU = 0.524. Bolivia IoU = 0.531.
6. The PyTorch dataset is ready for the U-Net in Stage 3.

---

## 2. Code that this stage made

| File | Task |
|---|---|
| `src/floodfm/data.py` | Reads the official splits and the chips |
| `src/floodfm/metrics.py` | Counts TP, FP, FN and TN. Calculates IoU, F1 and water share. |
| `src/floodfm/baselines.py` | Otsu threshold and fixed threshold on the VH band |
| `src/floodfm/dataset.py` | Gives normalised chips as PyTorch tensors |
| `notebooks/02_otsu_baseline.ipynb` | Runs all the steps of Stage 2 |

---

## 3. Facts about the data

### 3.1 Scored pixels

We score a pixel only if its label is 0 (not water) or 1 (water).

| Split | Chips | Scored pixels | Share of all pixels |
|---|---|---|---|
| train | 252 | 57,303,407 | 86.7% |
| valid | 89 | 20,294,725 | 87.0% |
| test | 90 | 20,517,367 | 87.0% |
| bolivia | 15 | 2,867,815 | 72.9% |

### 3.2 Scored pixels without radar

Some pixels have a label but no radar data. A radar method cannot see these pixels. Our rule: the prediction at these pixels is "not water".

| Split | Scored pixels without radar | Share of scored pixels |
|---|---|---|
| train | 24,150 | 0.04% |
| valid | 20,571 | 0.10% |
| test | 73,408 | 0.36% |
| bolivia | 43,381 | 1.5% |

**One test chip has no radar data:** `Paraguay_34417`. It has 50,026 scored pixels, and all of them are water. Thus every radar method gets 50,026 missed-water pixels on this chip. This is 68% of the test pixels without radar. It is about 2% of the test water pixels.

### 3.3 Water share

| Split | Water share |
|---|---|
| train | 9.5% |
| valid | 11.0% |
| test | 12.5% |
| bolivia | 15.9% |

---

## 4. The two baselines

### 4.1 Otsu threshold

- Otsu finds one threshold for each chip, from the VH values of that chip only.
- Otsu does not use the labels.
- A pixel is water if its VH value is less than the threshold.

### 4.2 Fixed threshold

- We tried thresholds from -50 dB to +10 dB, in steps of 0.1 dB.
- For each threshold, we calculated the IoU on the **training set only**.
- The best training threshold is **-23.6 dB**. The training IoU at this threshold is 0.465.
- We applied this one threshold to all chips in all splits.

The figure `stage2_fixed_threshold_curve.png` shows the training IoU for each threshold. The curve has one clear peak.

---

## 5. Results

### 5.1 Results for each split

| Split | Otsu IoU | Otsu F1 | Fixed IoU | Fixed F1 |
|---|---|---|---|---|
| train | 0.187 | 0.315 | 0.465 | 0.635 |
| valid | 0.220 | 0.361 | 0.492 | 0.659 |
| **test** | **0.230** | **0.374** | **0.524** | **0.688** |
| **bolivia** | **0.403** | **0.575** | **0.531** | **0.694** |

### 5.2 Results for each test event

| Event | Water share | Otsu IoU | Fixed IoU | Otsu F1 | Fixed F1 |
|---|---|---|---|---|---|
| Ghana | 8.0% | 0.109 | 0.453 | 0.197 | 0.624 |
| India | 21.3% | 0.386 | 0.593 | 0.557 | 0.745 |
| Mekong | 14.5% | 0.311 | 0.724 | 0.475 | 0.840 |
| Nigeria | 16.2% | 0.299 | 0.708 | 0.460 | 0.829 |
| Pakistan | 18.0% | 0.206 | 0.150 | 0.341 | 0.261 |
| Paraguay | 7.3% | 0.167 | 0.397 | 0.286 | 0.569 |
| Somalia | 6.5% | 0.161 | 0.256 | 0.277 | 0.408 |
| Spain | 23.8% | 0.413 | 0.640 | 0.585 | 0.780 |
| Sri-Lanka | 18.5% | 0.226 | 0.717 | 0.369 | 0.835 |
| USA | 3.1% | 0.078 | 0.379 | 0.145 | 0.550 |

---

## 6. Findings

### 6.1 Otsu makes too many false alarms

On the test set, Otsu has:

- TP = 1,832,911
- FP = 5,407,057
- FN = 733,190

Thus Otsu makes about **7 false alarms for each missed water pixel** (FP / FN = 7.4). Bolivia is similar (FP / FN = 7.9).

In simple words, on the test set:

- Otsu finds 71% of the real water.
- But only 25% of the pixels that Otsu calls water are real water.

### 6.2 Dry chips cause about half of the false alarms

- 28 of the 90 test chips have less than 1% water.
- These 28 chips make **48%** of all Otsu false alarms on the test set.
- On these chips, Otsu calls 39.7% of the pixels "water".

**The reason:** Otsu always divides a chip into two groups. On a dry chip, it still makes a "dark group" and calls that group water.

### 6.3 The Otsu thresholds change a lot between chips

On the 89 test chips with radar data, the Otsu threshold goes from -27.45 dB to -13.26 dB. The median is -17.80 dB. Thus Otsu uses a very different rule on each chip.

### 6.4 One fixed threshold is better on 9 of 10 test events

The fixed threshold does not make a "water group" on dry chips. Thus it makes far fewer false alarms.

### 6.5 Pakistan is the exception

On Pakistan, the fixed IoU (0.150) is less than the Otsu IoU (0.206).

**Hypothesis:** the water in the Pakistan chips gives a stronger VH echo than -23.6 dB. Wind, shallow water or plants in the water can cause this. **This hypothesis is not tested yet.** Stage 5 can test it with the mean VH value of the water pixels for each event.

### 6.6 Events with little water have low scores

USA (3.1% water) and Somalia (6.5% water) have low IoU for both baselines. With little water, a small number of false alarms has a large effect on IoU.

### 6.7 Test is higher than train for the fixed threshold

The fixed test IoU (0.524) is higher than the training IoU (0.465). This is not an error. The test set has a higher water share (12.5% against 9.5%). With more water, a high IoU is easier.

---

## 7. Normalisation statistics

We calculated the mean and the standard deviation of each band on the **training set only**. Missing values were not used (NaN in S1, 0 in S2).

| Band | Mean | Std |
|---|---|---|
| S1 VV (dB) | -10.393 | 4.039 |
| S1 VH (dB) | -17.241 | 4.754 |
| S2 B1 | 1661.123 | 666.127 |
| S2 B2 | 1424.481 | 718.937 |
| S2 B3 | 1392.123 | 715.993 |
| S2 B4 | 1243.430 | 855.719 |
| S2 B5 | 1496.563 | 755.291 |
| S2 B6 | 2436.733 | 863.260 |
| S2 B7 | 2905.197 | 1013.599 |
| S2 B8 | 2676.658 | 960.965 |
| S2 B8A | 3142.044 | 1121.895 |
| S2 B9 | 497.131 | 332.558 |
| S2 B10 | 65.140 | 145.223 |
| S2 B11 | 2074.410 | 944.490 |
| S2 B12 | 1204.740 | 752.659 |

---

## 8. Dataset check

| Item | Radar (s1) | Optical (s2) |
|---|---|---|
| Training chips | 252 | 252 |
| Image shape | (2, 512, 512) | (13, 512, 512) |
| Data type | float32 | float32 |
| NaN values after normalisation | 0 | 0 |
| Label values | -1, 0, 1 | -1, 0, 1 |
| Batch of 4 | (4, 2, 512, 512) | (4, 13, 512, 512) |

All checks are correct.

---

## 9. Limits of this stage

1. The labels show **all** visible water, not only flood water. Rivers and lakes are also "water".
2. The test split contains the **same 10 events** as the training split. Only Bolivia is a new event.
3. Bolivia has only **15 chips**. Its scores are less certain than the test scores.
4. Our Otsu baseline is simpler than the method of the dataset authors. They smoothed the VH band and used local thresholds. We used one threshold for each chip and no smoothing.
5. The fixed threshold uses the training labels to select one number. Thus it is a "tuned" baseline, not a "zero-learning" baseline. The report must state this.

---

## 10. Decisions for Stage 3

1. **The bar:** a model is useful only if it is better than the fixed threshold on test (0.524) and on Bolivia (0.531).
2. **Both baselines go in the final table.** Otsu shows the simplest method. The fixed threshold shows the strong simple method.
3. **The U-Net uses the same data module, the same scores and the same rules** as the baselines. Thus the comparison is fair.
4. **The U-Net loss skips pixels with the label -1,** in the same way as the scores.
