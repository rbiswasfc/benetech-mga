--------------
First of all, I would like to thank the Kaggle community for sharing great ideas and engaging discussions. Special shout outs to @nbroad  and @brendanartley. Congratulations to the winning teams - looking forward to your write ups.  

## Links

- Inference notebook: https://www.kaggle.com/code/conjuring92/a05-mga-split-pipe
- Training code, config & datasets: to be released soon


The following is a detailed summary of my solution:

## 1 Overview

My solution is entirely based on image-to-text models finetuned from the `google/matcha-base` backbone. The training pipeline, as depicted below, is a sequence of two phases. In the first phase, I leveraged a large number of synthetic graphs to adapt the backbone for our current task.  In the second phase, I used oversampled extracted / non-generated plots to specialize the pipeline for real world graphs. In this phase, I created separate models for scatter and non-scatter plots, primarily to mitigate difficulty in predicting scatter data points.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2125251%2F232cced3276ad6561fe6a56d2e7509ad%2Fmga_flow.png?generation=1687269107458839&alt=media)

## 2 Model

All models share the same architecture (image-to-text transformer) and input-output pattern. Model input is simply the plot image itself without any prompts. The output text has the following template:

```<|bos|> <|chart_type_start|> {chart_type} <|chart_type_end|> <|num_point_start|> {n_x} | {n_y} <|num_point_end|> <|x_span_start|> {x0} | {x1} | {x2} | … | {xn} <|x_span_end|> <|y_span_start|> {y0} | {y1} | {y2} | … | {ym}  <|y_span_end|> <|eos|>```

Some minor details:
- Numeric values are cast into scientific notation using `val = "{:.2e}".format(float(val))`.
- Added histogram as additional chart type, which later converted to vertical_bar during post processing

## 3 Data
Apart from the competition data, I used the following sources.

### Synthetic Dataset

I spent majority of my competition time creating the synthetic dataset. For the underlying data in synthetic plots, I used

- Wikitables data i.e. tables from wikipedia (25%)
    - http://websail-fe.cs.northwestern.edu/TabEL/
    - https://github.com/sunlab-osu/TURL
- Synthetic XY data (75%)
    - Categorical series: I created a list of categories using wikipedia glossary pages in STEM domain (https://www.kaggle.com/code/conjuring92/w03-stem-glossary/notebook)
    - Numerical series: random function generators ensuring all combinations of narrow to wide min-max range, small (1e-6) - large values (1e6), inclusion of outliers etc

I generated the plots using matplotlib ensuring they capture all aspects of the graph conventions. For example, in the case of line plots, the generated graphs included shared origin plots, having tick labels that are not included in the data-series, additional point markers in between two tick labels, unequal spacing between x tick labels etc. I tried to maximize the diversity in the graphs by -

- customizing tick marks, tick labels, tick direction, formatting of numerical tick labels (e.g. scientific notation, different rounding, European style formatting, adding % as suffix, currency symbol as prefix etc), major / minor grids, titles, axis limits, spines (e.g. setting bottom spine at y=0),  legends, markers etc
- grayscale, background textures, aspect ratio (very wide to very narrow)
- number of data points (4-24 points for non-scatter, 4-64 for scatter)
- random special effects: text boxes with stats on y values, horizontal / vertical bands, insets, random texts, random equations, annotating min/max points, error bands, adding random shapes, varying line width, color, data point marker size, hatches, error bars, slopes etc
- log-log, semi-log plots, reverse y axis (although these weren’t part of test set, my hypothesis (?) was they would help in model generalization)

The synthetic dataset consisted of

- 100k horizontal bars
- 100k vertical bars + histograms
- 100k dot plots
- 200k line plots
- 200k scatter plots

### Synthetic Dataset - Bartley
- Random selection of 25k data points from the synthetic dataset shared by @brendanartley: https://www.kaggle.com/datasets/brendanartley/benetech-extra-generated-data

### Pseudo Labelling
I took screenshot of around 700 images from wikimedia commons (e.g. https://commons.wikimedia.org/w/index.php?search=line+plots&title=Special:MediaSearch&go=Go&type=image). I used pseudo labelling, followed by manual correction, to generate the annotations.

### ICDAR dataset
I used around 1100 images from ICDAR, only those having 1 XY series (250 horizontal bar + 450 vertical bar + 250 lines + 150 scatter). I also did post-processing to ensure annotations match competition graph conventions (e.g. handling percentages, interpolation of line plot data to match tick labels etc).

## 4 Datamix
- Datamix 1: used for domain adaptation

| Dataset                         | Size | Multiplier | Effective Size |
| ------------------------------- | ---- | ---------- | -------------- |
| Competition Dataset - Synthetic | 60k  | 3          | 180k           |
| Competition Dataset - Extracted | 1.1k | 16         | 17k            |
| Synthetic Dataset - Self        | 700k | 1          | 700k           |
| Synthetic Dataset - Bartley     | 25k  | 1          | 25k            |

- Datamix 2: scatter specialization

| Dataset                         | Size | Multiplier | Effective Size |
| ------------------------------- | ---- | ---------- | -------------- |
| Competition Dataset - Synthetic | 11k  | 1          | 11k            |
| Competition Dataset - Extracted | 0.2k | 16         | 3.2k           |
| Synthetic Dataset - Self        | 30k  | 1          | 30k            |
| Pseudo Labelled Dataset         | 0.1k | 16         | 1.6k           |
| ICDAR Dataset                   | 0.2k | 16         | 3.2k           |

- Datamix 3: non-scatter specialization

| Dataset                         | Size | Multiplier | Effective Size |
| ------------------------------- | ---- | ---------- | -------------- |
| Competition Dataset - Synthetic | 48k  | 1          | 48k            |
| Competition Dataset - Extracted | 0.9k | 8          | 7.2k           |
| Synthetic Dataset - Self        | 20k  | 1          | 20k            |
| Pseudo Labelled Dataset         | 0.5k | 8          | 4k             |
| ICDAR Dataset                   | 1k   | 8          | 8k             |

## 5 Training

The main hyper-parameters for training were `max_patches` and `max_length`. I used the following settings at various phases of training:

##### Phase 1 Training

- patch size: 2048
- max length: 1024
- lr: 5e-5
- batch size: 2
- gradient accumulation: 16

##### Phase 2 training - non-scatter
- patch size: 4096
- max length: 512
- lr: 2e-5
- batch size: 4
- gradient accumulation: 2

##### Phase 2 training - scatter
- patch size: 3072
- max length: 1024
- lr: 2e-5
- batch size: 8
- gradient accumulation: 1
- AWP

As minor detail, I used Exponential Moving Average (EMA) of model weights, gradient clipping, cosine scheduler with liner warmup during training.

## 6 Augmentation

Since I was repeating the extracted images many times, I decided to include the following augmentation

```
transforms = A.Compose(
        [
            A.OneOf(
                [
                    A.RandomToneCurve(scale=0.3),
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.1, 0.2),
                        contrast_limit=(-0.4, 0.5),
                        brightness_by_max=True,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=(-20, 20),
                        sat_shift_limit=(-30, 30),
                        val_shift_limit=(-20, 20)
                    )
                ],
                p=0.5,
            ),

            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3),
                    A.MedianBlur(blur_limit=3),
                    A.GaussianBlur(blur_limit=3),
                    A.GaussNoise(var_limit=(3.0, 9.0)),
                ],
                p=0.5,
            ),

            A.Downscale(always_apply=False, p=0.1, scale_min=0.90, scale_max=0.99),
        ],

        p=0.5,
    )
```

## 7 References
* Matcha Paper: https://arxiv.org/pdf/2212.09662v2.pdf
* https://www.kaggle.com/code/nbroad/donut-train-benetech
* https://www.kaggle.com/code/nbroad/donut-infer-lb-0-44-benetech
* https://www.kaggle.com/datasets/brendanartley/benetech-extra-generated-data
* AWP: https://www.kaggle.com/code/wht1996/feedback-nn-train/notebook

PS: Sorry for the long write up. Please let me know if you have any queries / suggestions. I plan to release all scripts, configs and datasets by next week.