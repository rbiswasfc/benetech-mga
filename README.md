# Benetech - Making Graphs Accessible

This repo contains my solution code for the [Benetech - Making Graphs Accessible](https://www.kaggle.com/competitions/benetech-making-graphs-accessible/overview) competition. A detailed summary of the solution is posted [here](https://www.kaggle.com/competitions/benetech-making-graphs-accessible/discussion/418430). Please refer to the following sections for details on training, inference and dependencies. If you run into any issues with the setup/code or have any questions/suggestions, please feel free to contact me at saun.walker.150892@gmail.com. Thanks!


## Section 1: Setup

### 1.1 Hardware
I used the computing resources from [Jarvislabs.ai](https://cloud.jarvislabs.ai/). Specifically, models were trained on the following instance:

* Ubuntu 20.04.5 LTS (128 GB boot disk)
* Intel(R) Xeon(R) Silver 4216 CPU @ 2.10GHz (7 vCPUs)
* 1 x NVIDIA A100 40GB GPU OR 1 x NVIDIA A6000 48GB GPU


### 1.2 Software
I used PyTorch-2.0 image from [Jarvislabs.ai](https://cloud.jarvislabs.ai/), which comes with:

* Python 3.10.11
* CUDA 11.8
* Python packages installation: `pip install -r requirements.txt`

### 1.3 Datasets
Please make sure [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed. Then run the following script to download the required datasets:

```
chmod +x ./setup.sh
./setup.sh
```

Please note that the above script will create a `datasets` folder in the directory located one level above the current directory. The external datasets will be downloaded in the `datasets` folder. After executing the above script, the `datasets` folder should have the following structure:

```
./processed
./processed/fold_split
./processed/deps
./processed/deps/mga_textures_cc
./processed/deps/mga_textures_cc/mga_textures_cc
./processed/synthetic
./processed/synthetic/annotations
./processed/synthetic/images
./processed/mga_pl
./processed/mga_pl/annotations
./processed/mga_pl/images
./processed/mga_icdar
./processed/mga_icdar/annotations
./processed/mga_icdar/images
./benetech-making-graphs-accessible
./benetech-making-graphs-accessible/test
./benetech-making-graphs-accessible/test/images
./benetech-making-graphs-accessible/train
./benetech-making-graphs-accessible/train/annotations
./benetech-making-graphs-accessible/train/images
```

## Section 2: Training
The training scripts and configurations are located in the `code` and `conf` folder respectively. The training pipeline is implemented using HuggingFace's transformers, datasets and accelerate libraries. The training comprises of two phases: domain adaptation and specialization.

### 2.1 Domain Adaptation
In the first phase, I leveraged a large number of synthetic plots to adapt the `google/matcha-base` backbone for the current task.

```
HYDRA_FULL_ERROR=1 python ./code/train_r_final.py \
--config-name conf_r_final \
fold=0 \
use_wandb=false \
debug=false
```

Expected training time: 50 hours (A6000 GPU) OR 36 hours (A100 GPU)

### 2.2 Specialization
In the second phase, I used over-sampled extracted / non-generated plots to specialize the pipeline for real world graphs. In this phase, I created separate models for scatter and non-scatter plots, primarily to boost performance of scatter plot subset. The checkpoint from the domain adaptation phase is used as the starting point for the specialization phase.
#### 2.2.1 Scatter Specialization

Training for scatter plot subset is implemented using the following script:

```
HYDRA_FULL_ERROR=1 python ./code/train_r_scatter.py \
--config-name conf_r_scatter \
fold=0 \
use_wandb=false \
debug=false
```

#### 2.2.2 Non-Scatter Specialization
Training for non-scatter plot subset is implemented using the following script:

```
HYDRA_FULL_ERROR=1 python ./code/train_r_rest.py \
--config-name conf_r_rest \
fold=0 \
use_wandb=false \
debug=false
```


## Section 3: Inference
The inference notebook is published here: https://www.kaggle.com/code/conjuring92/a05-mga-split-pipe

The notebook uses three trained model checkpoints which are hosted as kaggle datasets:
* https://www.kaggle.com/datasets/conjuring92/mga-r-final-matcha-v2-ft-i2
* https://www.kaggle.com/datasets/conjuring92/mga-r-rest-v2
* https://www.kaggle.com/datasets/conjuring92/mga-r-scatter-matcha


## Section 4: Synthetic Dataset Generation (Optional)
My solution relies on generating a large synthetic dataset for training of models. The dataset is hosted [here](https://www.kaggle.com/datasets/conjuring92/mga-synthetic). The following sections describe the steps to generate the synthetic dataset. You can customize parameters such as number of generated images, output locations and other details using the config files located in the `gen/conf` directory. Note that, generation code assumes fonts such as Helvetica, Courier New, Times New Roman, Lucida Grande, Arial and Verdana are pre-installed.

### 4.1 Vertical Bar Plots

Basic vertical bar plots are generated using the following script:

``` 
python ./gen/run_gen_vbar.py \
--wiki_path ../datasets/processed/deps/sanitized_wiki.json \
--stem_path ../datasets/processed/deps/mga_stem_kws.pickle \
--conf_path ./gen/conf/conf_vbar.yaml \
--texture_dir ../datasets/processed/deps/mga_textures_cc/mga_textures_cc
```

Vertical bar plots with more customization are generated using the following script:

```
python ./gen/run_gen_vbar_a0.py \
--wiki_path ../datasets/processed/deps/sanitized_wiki.json \
--stem_path ../datasets/processed/deps/mga_stem_kws.pickle \
--conf_path ./gen/conf/conf_vbar_a0.yaml \
--texture_dir ../datasets/processed/deps/mga_textures_cc/mga_textures_cc
```

### 4.2 Horizontal Bar Plots
Basic horizontal bar plots are generated using the following script:

```
python ./gen/run_gen_hbar.py \
--wiki_path ../datasets/processed/deps/sanitized_wiki.json \
--stem_path ../datasets/processed/deps/mga_stem_kws.pickle \
--conf_path ./gen/conf/conf_hbar.yaml \
--texture_dir ../datasets/processed/deps/mga_textures_cc/mga_textures_cc
```

Horizontal bar plots with more customization are generated using the following script:

```
python ./gen/run_gen_hbar_a0.py \
--wiki_path ../datasets/processed/deps/sanitized_wiki.json \
--stem_path ../datasets/processed/deps/mga_stem_kws.pickle \
--conf_path ./gen/conf/conf_hbar_a0.yaml \
--texture_dir ../datasets/processed/deps/mga_textures_cc/mga_textures_cc
```

### 4.3 Dot Plots

The dot plots are generated using the following script:

```
python ./gen/run_gen_dot.py \
--stem_path ../datasets/processed/deps/mga_stem_kws.pickle \
--conf_path ./gen/conf/conf_dot.yaml \
--texture_dir ../datasets/processed/deps/mga_textures_cc/mga_textures_cc
```
### 4.4 Line Plots

The basic line plots are generated using the following script:

```
python ./gen/run_gen_line.py \
--wiki_path ../datasets/processed/deps/sanitized_wiki.json \
--stem_path ../datasets/processed/deps/mga_stem_kws.pickle \
--conf_path ./gen/conf/conf_line.yaml \
--texture_dir ../datasets/processed/deps/mga_textures_cc/mga_textures_cc
```

Line plots with more customization are generated using the following script:

```
python ./gen/run_gen_line_a0.py \
--wiki_path ../datasets/processed/deps/sanitized_wiki.json \
--stem_path ../datasets/processed/deps/mga_stem_kws.pickle \
--conf_path ./gen/conf/conf_line_a0.yaml \
--texture_dir ../datasets/processed/deps/mga_textures_cc/mga_textures_cc
```

Line plots with shared origin edge case are generated using the following script:

```
python ./gen/run_gen_line_shared.py \
--conf_path ./gen/conf/conf_line_shared.yaml \
--texture_dir ../datasets/processed/deps/mga_textures_cc/mga_textures_cc
```


Line plots with even more diverse styles are generated using the following script:

```
python ./gen/run_gen_line_a3.py \
--wiki_path ../datasets/processed/deps/sanitized_wiki.json \
--stem_path ../datasets/processed/deps/mga_stem_kws.pickle \
--conf_path ./gen/conf/conf_line_a3.yaml \
--texture_dir ../datasets/processed/deps/mga_textures_cc/mga_textures_cc
```

### 4.5 Scatter Plots

The basic scatter plots are generated using the following script:
```
python ./gen/run_gen_scatter.py \
--conf_path ./gen/conf/conf_scatter.yaml \
--texture_dir ../datasets/processed/deps/mga_textures_cc/mga_textures_cc
```

Scatter plots with more customization are generated using the following script:

```
python ./gen/run_gen_scatter_a0.py \
--conf_path ./gen/conf/conf_scatter_a0.yaml \
--texture_dir ../datasets/processed/deps/mga_textures_cc/mga_textures_cc
```

