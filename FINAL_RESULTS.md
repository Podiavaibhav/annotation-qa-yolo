# Annotation QA + YOLO Retraining Benchmark

## African Wildlife Dataset --- Final Experimental Results

> **Headline result:** Cleaning recovered **53.6% of the mAP50
> performance loss** introduced by synthetic annotation noise.

## 1. Experiment overview

The benchmark compared three dataset conditions:

-   **Original:** prepared dataset with its original annotations.
-   **Noisy:** a copy of the training dataset with synthetic annotation
    corruption injected.
-   **Cleaned:** the noisy dataset after the annotation QA and cleaning
    pipeline processed it.

Final YOLO comparisons used three random seeds: **42, 43, and 44**.
Metrics below are mean ± standard deviation across those three runs.

Dataset: - **1,504 images** total: 1,204 training and 300 validation. -
**2,685 bounding boxes**. - Four classes: buffalo, elephant, rhino, and
zebra.

The split was made within each source folder using a fixed random seed.
Original class labels were preserved; source folders were not assumed to
contain only their corresponding class.

## 2. Model performance

  -----------------------------------------------------------------------
  Dataset              mAP50       mAP50-95      Precision         Recall
  condition                                                
  ----------- -------------- -------------- -------------- --------------
  Original        **0.9673 ±     **0.8217 ±     **0.9514 ±     **0.9251 ±
                    0.0028**       0.0062**       0.0048**       0.0070**

  Noisy             0.9526 ±       0.7921 ±       0.9278 ±       0.9049 ±
                      0.0031         0.0071         0.0125         0.0141

  Cleaned           0.9605 ±       0.8068 ±       0.9509 ±       0.9108 ±
                      0.0040         0.0058         0.0137         0.0157
  -----------------------------------------------------------------------

Values are mean ± standard deviation across three seeds.

### mAP50 comparison

``` mermaid
xychart-beta
    title "mAP50 by dataset condition"
    x-axis ["Original", "Noisy", "Cleaned"]
    y-axis "mAP50 (%)" 90 --> 100
    bar [96.73, 95.26, 96.05]
```

### mAP50-95 comparison

``` mermaid
xychart-beta
    title "mAP50-95 by dataset condition"
    x-axis ["Original", "Noisy", "Cleaned"]
    y-axis "mAP50-95 (%)" 70 --> 90
    bar [82.17, 79.21, 80.68]
```

### Precision and recall comparison

``` mermaid
xychart-beta
    title "Precision and recall by dataset condition"
    x-axis ["Original", "Noisy", "Cleaned"]
    y-axis "Score (%)" 80 --> 100
    bar [95.14, 92.78, 95.09]
    bar [92.51, 90.49, 91.08]
```

The first bar series is precision; the second is recall.

## 3. Effect of synthetic noise and cleaning

-   mAP50 loss from Original to Noisy: **0.0146** (approximately 1.46
    percentage points).
-   mAP50 gain from Noisy to Cleaned: **0.0078** (approximately 0.78
    percentage points).
-   Share of mAP50 loss recovered by cleaning: **53.6%**.

``` mermaid
flowchart TD
    A["Original mAP50: 96.73%"] -->|"Synthetic noise: −1.46 pp"| B["Noisy mAP50: 95.26%"]
    B -->|"Cleaning: +0.78 pp"| C["Cleaned mAP50: 96.05%"]
```

  Metric         Noisy   Cleaned         Change
  ----------- -------- --------- --------------
  mAP50         95.26%    96.05%   **+0.79 pp**
  mAP50-95      79.21%    80.68%   **+1.47 pp**
  Precision     92.78%    95.09%   **+2.31 pp**
  Recall        90.49%    91.08%   **+0.59 pp**

### mAP50 by seed

    Seed   Original    Noisy   Cleaned
  ------ ---------- -------- ---------
      42     96.54%   94.98%    95.98%
      43     97.04%   95.21%    96.48%
      44     96.60%   95.60%    95.69%

The Cleaned condition improved over Noisy for all three seeds, but did
not exceed Original for any seed.

## 4. Annotation error detector evaluation

The detector was evaluated against the synthetic corruption manifest.
Recall means the share of injected errors that the detector flagged.

  ------------------------------------------------------------------------------
  Injected error           Injected         Flagged          Recall Additional
  type                                                              result
  ----------------- --------------- --------------- --------------- ------------
  Deleted                       163             133       **81.6%** 127 restored
  annotations                                                       with correct
                                                                    class

  Class flips                   170             126       **74.1%** 125
                                                                    diagnosed
                                                                    with correct
                                                                    class

  Shifted/resized               291             209       **71.8%** 205
  boxes                                                             predicted
                                                                    boxes
                                                                    matched
                                                                    original; 15
                                                                    mild cases
                                                                    excluded

  Duplicates                     70              58       **82.9%** ---
  ------------------------------------------------------------------------------

### Detector recall by error type

``` mermaid
xychart-beta
    title "Injected annotation error detection recall"
    x-axis ["Deleted", "Class flip", "Localization", "Duplicate"]
    y-axis "Recall (%)" 0 --> 100
    bar [81.6, 74.1, 71.8, 82.9]
```

### Flag precision

Precision is the share of detector flags matched to an injected error.

  Flag type              Flags   Matched an injected error   Precision
  -------------------- ------- --------------------------- -----------
  Missing                  190                         133   **70.0%**
  Class disagreement       152                         126   **82.9%**
  Localization             297                         210   **70.7%**

The experiment report notes that some unmatched flags may correspond to
annotation errors already present in the original dataset. Under that
assumption, these precision values are lower bounds, not definitive
estimates of false-positive rates.

## 5. Interpretation

-   Synthetic annotation corruption reduced detection performance across
    all four reported metrics.
-   The QA and cleaning pipeline improved performance over Noisy across
    all four mean metrics.
-   Cleaning recovered **53.6% of the mAP50 loss** caused by injected
    noise.
-   Detector recall was highest for duplicates (82.9%) and lowest for
    localization errors (71.8%).
-   Class-disagreement flags had the highest reported precision (82.9%).
-   Cleaned improved over Noisy for mAP50 on all three seeds, but did
    not surpass Original.

## 6. Limitations and responsible claims

1.  **Three seeds:** This is a small sample. Consistency across these
    runs is encouraging, but does not by itself establish statistical
    significance.
2.  **Synthetic corruption:** The corruption manifest provides ground
    truth for deliberately injected errors. It does not establish that
    every original annotation is correct.
3.  **Original validation labels:** Evaluation uses original validation
    annotations as the reference. If they contain errors, reported
    detection metrics may be affected.
4.  **Detection versus repair:** Flagging an error is not the same as
    correcting it. The reported detector metrics and downstream model
    metrics provide different evidence. These results are not a complete
    annotation-level audit of every repair.
5.  **Partial recovery:** Cleaning improved on Noisy but did not fully
    recover Original mAP50 or mAP50-95.

**Defensible conclusion:** The pipeline detected a substantial share of
injected annotation errors, and cleaning improved downstream YOLO
performance relative to training on the synthetically corrupted dataset.
The experiment does not show that cleaning makes the dataset error-free
or outperforms the original annotations.

## 7. Reproducibility details

  Setting                   Value
  ------------------------- ---------------------------------
  Dataset                   African Wildlife
  Total images              1,504
  Training / validation     1,204 / 300
  Classes                   Buffalo, elephant, rhino, zebra
  Original bounding boxes   2,685
  Model                     YOLOv8n
  Image size                640
  Final training epochs     30
  Training seeds            42, 43, 44
  OOF folds                 2
  OOF epochs per fold       30
  Compute                   Kaggle Tesla T4

The final comparison comprises three dataset conditions × three seeds =
nine final training runs. The OOF detector stage adds two fold-training
runs.

## Final takeaway

This experiment demonstrates an end-to-end workflow for **synthetic
annotation-noise injection, model-assisted annotation auditing, dataset
cleaning, and downstream model evaluation**.

The headline result is **53.6% recovery of the mAP50 performance loss
introduced by synthetic annotation noise**, with cleaned-model mAP50
improving over the noisy condition for all three training seeds.
