# Stage 0 and Stage 1 report: setup, data download and first look at the data

**Project:** FloodFM-Sen1Floods11
**Writing standard:** ASD-STE100 (Simplified Technical English)
**Status:** complete

**Source of the numbers:** all numbers in this report come from the notebooks `00_setup.ipynb` and `01_explore_chip.ipynb`. The main numbers are also saved on Drive in `stage0_data_report.json` and `stage1_label_stats.csv` (folder `content/results/`).

---

## 1. Summary

1. The workspace is ready: PC folders, a public GitHub repository, Colab with a GPU, and project folders on Google Drive.
2. The dataset was not on Drive. We downloaded the hand-labelled part from the official source.
3. The data is complete: 446 chips, with radar, optical and label files for each chip.
4. The four official splits contain 446 different chips. No chip is in two splits.
5. The test split contains the **same 10 flood events** as the training split. Only **Bolivia** is a new event.
6. Many chips have large areas with no data. Water share must be calculated on the scored pixels only.
7. The water share and the no-data share are very different between events.

---

## 2. Stage 0: the workspace

### 2.1 The four places

| Place | Task | Location |
|---|---|---|
| PC | Write the code | the local repository folder `codes` |
| GitHub | Keep the official copy of the code | github.com/Prhm93/floodfm-sen1floods11 (public) |
| Colab | Run the code on a GPU | Opened from VS Code with the Colab extension |
| Google Drive | Keep the data and the results | `MyDrive/2026Research/FloodFM-Sen1Floods11/content/` |

**Rule:** code goes to GitHub. Data and results go to Drive. Colab uses both while it works.

### 2.2 GPU check

| Item | Value |
|---|---|
| GPU available | True |
| GPU name | Tesla T4 |
| GPU memory | 15.6 GB |

### 2.3 Folders on Drive

`data`, `checkpoints`, `results` and `figures`, inside `MyDrive/2026Research/FloodFM-Sen1Floods11/content/`.

---

## 3. Stage 0: the data download

### 3.1 Source

- **Dataset:** Sen1Floods11, version 1.1
- **Official page:** https://github.com/cloudtostreet/Sen1Floods11
- **Official storage:** the public bucket `gs://sen1floods11/v1.1`
- **Tool:** `gsutil rsync` in Colab. No sign-in was necessary.

### 3.2 What we downloaded

| Folder | Content | Size |
|---|---|---|
| `S1Hand` | Radar images (Sentinel-1) | 695.7 MiB |
| `S2Hand` | Optical images (Sentinel-2) | 971.42 MiB |
| `LabelHand` | Hand-drawn water labels | 2.44 MiB |
| `splits/flood_handlabeled` | The four official split files | small |

**Not downloaded:** `JRCWaterHand`, `S1OtsuLabelHand` and the weakly labelled set. The project does not use them.

**Location on Drive:** `content/data/sen1floods11/`

### 3.3 Checks

| Check | Result |
|---|---|
| Files in `S1Hand` | 446 `.tif` |
| Files in `S2Hand` | 446 `.tif` |
| Files in `LabelHand` | 446 `.tif` |
| Split files | 4 (train, valid, test, bolivia) |
| Header line in the split files | None. Each line is one chip. |
| Chips in all splits together | 446 |
| Different chips | 446 |
| Chips in more than one split | 0 |
| Missing files | 0 |

### 3.4 Chips in each split and each event

| Event | train | valid | test | bolivia |
|---|---|---|---|---|
| Bolivia | 0 | 0 | 0 | 15 |
| Ghana | 31 | 11 | 11 | 0 |
| India | 40 | 14 | 14 | 0 |
| Mekong | 18 | 6 | 6 | 0 |
| Nigeria | 10 | 4 | 4 | 0 |
| Pakistan | 16 | 6 | 6 | 0 |
| Paraguay | 39 | 14 | 14 | 0 |
| Somalia | 15 | 5 | 6 | 0 |
| Spain | 18 | 6 | 6 | 0 |
| Sri-Lanka | 24 | 9 | 9 | 0 |
| USA | 41 | 14 | 14 | 0 |
| **Total** | **252** | **89** | **90** | **15** |

### 3.5 Finding: the two test sets answer two different questions

| Test set | Question |
|---|---|
| test (90 chips) | Can the model map water in **new chips from events it saw** during training? |
| bolivia (15 chips) | Can the model map water in **an event it never saw**? |

A central question of the project is whether models work in a new place. Thus the Bolivia result is the most important result of the project. But Bolivia has only 15 chips. Thus its scores are less certain.

---

## 4. Stage 1: facts about the files

These facts come from the official dataset page and from our checks.

### 4.1 Radar (S1Hand)

- 2 bands: **VV** (index 0) and **VH** (index 1).
- Unit: decibels (dB). Every -10 dB means 10 times less echo.
- Data type: float32.
- Pixels outside the satellite path are **NaN** (no value).

### 4.2 Optical (S2Hand)

- 13 bands, in the order B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12.
- Red is B4 (index 3). Green is B3 (index 2). Blue is B2 (index 1). Python counts from 0.
- Unit: reflectance multiplied by 10,000.
- Level-1C: the haze of the air is not removed.
- Pixels with no image have the value **0**.
- **Difference from the documentation:** the official page says UInt16. Our files are int16. Our values (0 to 5,662 in the first chip) fit in both types. Thus this difference has no effect.

### 4.3 Label (LabelHand)

- 1 band. Data type: int16.
- **1** = water. **0** = not water. **-1** = no data.
- Pixels with -1 are not used for training or for scores.

### 4.4 Position

- Coordinate system: EPSG:4326 (latitude and longitude).
- Pixel size: 0.0000898 degrees. This is about **10 m** on the ground.
- One chip: 512 × 512 pixels. This is about 5 km × 5 km.

---

## 5. Stage 1: the first chip (Bolivia_103757)

| Item | Value |
|---|---|
| Radar shape | (2, 512, 512) |
| Radar range | -48.25 dB to -0.69 dB |
| Radar NaN values | 321,042 of 524,288 (about 61%) |
| Optical shape | (13, 512, 512) |
| Optical range | 0 to 5,662 |
| Label: no data (-1) | 174,067 pixels (66.4%) |
| Label: not water (0) | 52,715 pixels (20.1%) |
| Label: water (1) | 35,362 pixels (13.5%) |
| **Water among scored pixels** | **40.1%** |

**Finding:** the water share of the whole chip (13.5%) is very different from the water share of the scored pixels (40.1%). **We must always calculate the water share on scored pixels only.**

**What the figure showed:**

- The dark areas of the radar agree with the water in the label.
- The thin river in the middle is dark in the radar and is water in the label.
- The small no-data patches in the label are at the same places as the clouds in the optical image.

---

## 6. Stage 1: the data across all 446 chips

### 6.1 Mean shares for each event

Each value is the mean of the chip values. Each chip has the same weight.

| Event | Chips | Mean no-data share | Mean water share (scored pixels) |
|---|---|---|---|
| Bolivia | 15 | 27.1% | 17.0% |
| Ghana | 53 | 33.8% | 7.1% |
| India | 68 | 16.0% | 12.8% |
| Mekong | 30 | 9.3% | 23.6% |
| Nigeria | 18 | 14.3% | 22.3% |
| Pakistan | 28 | 26.7% | 7.6% |
| Paraguay | 67 | 5.0% | 9.4% |
| Somalia | 26 | 21.1% | 6.2% |
| Spain | 30 | 1.7% | 14.1% |
| Sri-Lanka | 42 | 9.5% | 11.2% |
| USA | 69 | 2.4% | 4.6% |

**Findings:**

- The water share is between 4.6% (USA) and 23.6% (Mekong).
- Ghana has the most no data (33.8%). Spain (1.7%) and USA (2.4%) have almost none.

### 6.2 Three very different chips

| Chip | No data | Water (scored pixels) | Finding |
|---|---|---|---|
| Sri-Lanka_534068 | 0.0% | 98.1% | Almost all sea. A method that always makes two groups (Otsu) will call part of the sea "land". |
| Ghana_161233 | 99.3% | 0.0% | The optical image is all cloud. The radar sees through the cloud, but the label is almost all no data. This chip adds almost nothing to the scores. |
| Ghana_124834 | 0.0% | 0.0% | All land. Dark lines in the radar and optical images can cause false alarms. |

---

## 7. Decisions from Stage 0 and Stage 1

1. **Scored pixels only.** All scores and water shares use pixels with the label 0 or 1.
2. **Add the pixels, then calculate.** We add the counts of all chips in a group first. Then we calculate one IoU. Reason: on a chip with no water, the IoU of the chip can be 0 ÷ 0.
3. **No accuracy score.** With little water, a method that says "no water" everywhere gets a high accuracy. Thus we use water IoU and water F1.
4. **Water share next to each score.** A score has a different meaning when the water share is different.
5. **Report test and Bolivia separately.** They answer different questions.

---

## 8. Display rules (for figures only)

These rules change only the pictures. They do not change the data.

- **Percentile stretch:** each band is stretched between its 2nd and 98th percentile.
- **No data is grey** in all panels. Thus white in the radar always means a strong echo.
- **Optical value 0 is no data.** It is not used in the stretch.
- **Each chip is stretched alone.** Thus a chip that is almost all water can look grey and noisy, not dark.

---

## 9. Limits

1. The labels show **all** visible water: rivers, lakes, sea and flood water. Thus the task is "water mapping during floods", not "flood-only mapping".
2. Each chip is **one picture at one time**. The data does not show how the flood changes.
3. The people who drew the labels could not see through cloud. Thus cloudy areas have no labels, also where the radar has data.

---

## 10. Files from Stage 0 and Stage 1

| File | Location |
|---|---|
| `00_setup.ipynb` | `codes/notebooks/` |
| `01_explore_chip.ipynb` | `codes/notebooks/` |
| `stage0_data_report.json` | Drive, `content/results/` |
| `stage1_label_stats.csv` | Drive, `content/results/` |
| `stage1_<chip>.png` (4 figures) | Drive, `content/figures/` |
