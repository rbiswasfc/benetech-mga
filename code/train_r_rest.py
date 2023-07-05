import gc
import glob
import json
import os
import random
import time
from copy import deepcopy
from textwrap import wrap

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import torch
import wandb
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import GenerationConfig, get_cosine_schedule_with_warmup

try:
    from r_rest.mga_dataloader import MGACollator
    from r_rest.mga_dataset import (TOKEN_MAP, MGADataset,
                                    create_train_transforms)
    from r_rest.mga_model import AWP, MGAModel
    from utils.constants import EXCLUDE_IDS
    from utils.data_utils import process_annotations
    from utils.metric_utils import compute_metrics
    from utils.train_utils import (EMA, AverageMeter, as_minutes, get_lr,
                                   init_wandb, print_gpu_utilization,
                                   print_line, save_checkpoint,
                                   seed_everything)

except Exception as e:
    print(e)
    raise ImportError

pd.options.display.max_colwidth = 1000
BOS_TOKEN = TOKEN_MAP["bos_token"]


# --- show batch ------------------------------------------------------------------#


def run_sanity_check(cfg, batch, tokenizer, prefix="mga", num_examples=8):
    print("generating sanity check results for a training batch...")
    os.makedirs(os.path.join(cfg.outputs.model_dir, "examples"), exist_ok=True)

    num_examples = min(num_examples, len(batch['images']))
    print(f"num_examples={num_examples}")

    for i in range(num_examples):
        image = batch['images'][i]
        text = tokenizer.decode(
            batch['decoder_input_ids'][i], skip_special_tokens=True)

        text = "\n".join(wrap(text, width=128))

        # display image and its corresponding text label ---
        plt.figure(figsize=(15, 15))
        plt.imshow(image)
        plt.xlabel(text)

        image_path = os.path.join(cfg.outputs.model_dir, "examples", f"example_{prefix}_{i}.jpg")
        plt.savefig(image_path)
    print("done!")

# -------- Evaluation -------------------------------------------------------------#


def post_process(pred_string, token_map, delimiter="|"):
    # get chart type ---
    chart_options = [
        "horizontal_bar",
        "dot",
        "scatter",
        "vertical_bar",
        "line",
        "histogram",
    ]

    chart_type = "line"  # default type

    for ct in chart_options:
        if token_map[ct] in pred_string:
            chart_type = ct
            break

    if chart_type == "histogram":
        chart_type = "vertical_bar"

    # get x series ---
    x_start_tok = token_map["x_start"]
    x_end_tok = token_map["x_end"]

    try:
        x = pred_string.split(x_start_tok)[1].split(x_end_tok)[0].split(delimiter)
        x = [elem.strip() for elem in x if len(elem.strip()) > 0]
    except IndexError:
        x = []

    # get y series ---
    y_start_tok = token_map["y_start"]
    y_end_tok = token_map["y_end"]

    try:
        y = pred_string.split(y_start_tok)[1].split(y_end_tok)[0].split(delimiter)
        y = [elem.strip() for elem in y if len(elem.strip()) > 0]
    except IndexError:
        y = []

    # min_length = min(len(x), len(y))

    # if len(x) > min_length + 1:
    #     x = x[:min_length]
    # if len(y) > min_length + 1:
    #     y = y[:min_length]

    return chart_type, x, y


def run_evaluation(cfg, model, valid_dl, label_df, tokenizer, token_map):

    # # config for text generation ---
    conf_g = {
        "max_new_tokens": cfg.model.max_length_generation,  # 256,
        "do_sample": False,
        "top_k": 1,
        "use_cache": True,
    }

    generation_config = GenerationConfig(**conf_g)

    # put model in eval mode ---
    model.eval()

    all_ids = []
    all_texts = []

    progress_bar = tqdm(range(len(valid_dl)))
    for batch in valid_dl:
        with torch.no_grad():
            batch_ids = batch["id"]
            generated_ids = model.backbone.generate(
                flattened_patches=batch['flattened_patches'],
                attention_mask=batch['attention_mask'],
                generation_config=generation_config,
            )
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            all_ids.extend(batch_ids)
            all_texts.extend(generated_texts)

        progress_bar.update(1)
    progress_bar.close()

    # prepare output dataframe ---
    preds = []
    extended_preds = []
    for this_id, this_text in zip(all_ids, all_texts):
        id_x = f"{this_id}_x"
        id_y = f"{this_id}_y"
        pred_chart, pred_x, pred_y = post_process(this_text, token_map)

        preds.append([id_x, pred_x, pred_chart])
        preds.append([id_y, pred_y, pred_chart])

        extended_preds.append([id_x, pred_x, pred_chart, this_text])
        extended_preds.append([id_y, pred_y, pred_chart, this_text])

    pred_df = pd.DataFrame(preds)
    pred_df.columns = ["id", "data_series", "chart_type"]

    eval_dict = compute_metrics(label_df, pred_df)

    result_df = pd.DataFrame(extended_preds)
    result_df.columns = ["id", "pred_data_series", "pred_chart_type", "pred_text"]
    result_df = pd.merge(label_df, result_df, on="id", how="left")
    result_df['score'] = eval_dict['scores']  # individual scores

    results = {
        "oof_df": pred_df,
        "result_df": result_df,
    }

    for k, v in eval_dict.items():
        if k != 'scores':
            results[k] = v

    print_line()
    print("Evaluation Results:")
    print(results)
    print_line()

    return results


# -------- Main Function ---------------------------------------------------------#

def execution_setup(cfg):
    print_line()
    if cfg.use_random_seed:
        seed = random.randint(401, 999)
        cfg.seed = seed

    print(f"setting seed: {cfg.seed}")
    seed_everything(cfg.seed)

    if cfg.all_data:
        print("running training with all data...")
        fold = 0
        cfg.train_folds = [i for i in range(cfg.fold_metadata.n_folds)]
        cfg.valid_folds = [fold]
        cfg.outputs.model_dir = os.path.join(
            cfg.outputs.model_dir, f"all_data_training/seed_{cfg.seed}"
        )
    else:
        fold = cfg.fold
        cfg.train_folds = [i for i in range(cfg.fold_metadata.n_folds) if i != fold]
        cfg.valid_folds = [fold]

    cfg.train_folds.append(99)  # to include synthetic data
    print(f"train folds: {cfg.train_folds}")
    print(f"valid folds: {cfg.valid_folds}")

    # folder ---
    os.makedirs(cfg.outputs.model_dir, exist_ok=True)

    return cfg


@hydra.main(version_base=None, config_path="../conf/r_rest", config_name="conf_r_rest")
def run_training(cfg):
    cfg = execution_setup(cfg)
    fold = cfg.fold

    # labels ---
    label_df = process_annotations(cfg)
    # only scatter ---
    label_df = label_df[label_df["chart_type"] != "scatter"].copy()
    label_df["original_id"] = label_df["id"].apply(lambda x: x.split("_")[0])
    rest_ids = label_df["original_id"].unique().tolist()
    label_df = label_df.reset_index(drop=True)

    # show labels ---
    print("labels:")
    print(label_df.head())
    print_line()

    # ------- load data -----------------------------------------------------------------#
    print_line()
    fold_dir = cfg.fold_metadata.fold_dir
    fold_df = pd.read_parquet(os.path.join(fold_dir, cfg.fold_metadata.fold_path))

    # extracted multiplier ---
    extracted_ids = fold_df[fold_df["kfold"] != 99]["id"].unique().tolist()
    extracted_ids = [gid for gid in extracted_ids if gid not in EXCLUDE_IDS]

    train_ids = fold_df[fold_df["kfold"].isin(cfg.train_folds)]["id"].unique().tolist()
    print(f'# images in original train: {len(train_ids)}')
    train_ids = list(set(train_ids).intersection(set(rest_ids)))
    print(f'# images in original train (non scatter): {len(train_ids)}')

    extracted_ids = list(set(extracted_ids).intersection(set(rest_ids)))
    print(f'# images in extracted (non scatter): {len(extracted_ids)}')

    # ---- repeat original train ids ----#
    train_ids = train_ids * cfg.original_multiplier
    print(f'# images in original train after multiplier: {len(train_ids)}')

    # ----------------------------------#
    extracted_train_ids = deepcopy(extracted_ids)
    # extracted_train_ids = list(set(train_ids).intersection(set(extracted_ids)))

    print(f"# extracted train ids: {len(extracted_train_ids)}")
    extracted_train_ids = extracted_train_ids * max(cfg.extracted_multiplier - 1, 1)
    if len(extracted_train_ids) > 0:
        train_ids.extend(extracted_train_ids)

    print(f'# images in original train after extracted multiplier: {len(train_ids)}')

    # valid ids
    valid_ids = fold_df[fold_df["kfold"].isin(cfg.valid_folds)]["id"].unique().tolist()
    valid_ids = list(set(valid_ids).intersection(set(rest_ids)))

    # ----
    label_df = label_df[label_df["original_id"].isin(valid_ids)].copy()
    label_df = label_df.drop(columns=["original_id"])
    label_df = label_df.sort_values(by="source")
    label_df = label_df.reset_index(drop=True)
    # ----

    print(f"# of graphs in train: {len(train_ids)}")
    print(f"# of graphs in valid: {len(valid_ids)}")
    print_line()

    if cfg.add_syn:
        syn_image_dir = f"{cfg.competition_dataset.syn_dir}/images"
        syn_anno_dir = f"{cfg.competition_dataset.syn_dir}/annotations"

        print(f"adding  data from {syn_image_dir}")
        syn_ids = glob.glob(f"{syn_image_dir}/*.jpg")
        syn_ids_anno = glob.glob(f"{syn_anno_dir}/*.json")

        syn_ids = [fid.split("/")[-1].split(".jpg")[0] for fid in syn_ids]
        syn_ids = [gid.split("_v0")[0] for gid in syn_ids]
        syn_ids = [gid for gid in syn_ids if 'scatter' not in gid]

        syn_ids_anno = [fid.split("/")[-1].split(".json")[0] for fid in syn_ids_anno]

        print(f"# Synthetic before filter: {len(syn_ids)}")
        syn_ids = list(set(syn_ids).intersection(set(syn_ids_anno)))
        print(f"# Synthetic after filter: {len(syn_ids)}")

        syn_ids = [gid for gid in syn_ids if gid not in EXCLUDE_IDS]
        print(f"# SYN ids after exclusion: {len(syn_ids)}")

        print('sampling 10k syn ids')
        syn_ids = random.sample(syn_ids, k=10000)

        train_ids.extend(syn_ids)
        print(f"# of graphs in train: {len(train_ids)}")
        print_line()

    if cfg.add_pl:
        pl_image_dir = f"{cfg.competition_dataset.pl_dir}/images"
        pl_anno_dir = f"{cfg.competition_dataset.pl_dir}/annotations"

        print(f"adding  data from {pl_image_dir}")
        pl_ids = glob.glob(f"{pl_image_dir}/*.jpg")
        pl_ids = [fid.split("/")[-1].split(".jpg")[0] for fid in pl_ids]

        pl_ids_anno = glob.glob(f"{pl_anno_dir}/*.json")
        pl_ids_anno = [fid.split("/")[-1].split(".json")[0] for fid in pl_ids_anno]

        pl_ids = list(set(pl_ids).intersection(set(pl_ids_anno)))

        print(f"# PL ids before exclusion: {len(pl_ids)}")
        pl_ids = [gid for gid in pl_ids if gid not in EXCLUDE_IDS]
        pl_ids = [gid for gid in pl_ids if 'scatter' not in gid]

        print(f"# PL ids after exclusion: {len(pl_ids)}")

        print(f"# PL before multiplier: {len(pl_ids)}")
        pl_ids = pl_ids*cfg.pl_multiplier
        print(f"# PL after multiplier: {len(pl_ids)}")

        train_ids.extend(pl_ids)
        print(f"# of graphs in train: {len(pl_ids)}")
        print_line()

    if cfg.add_icdar:
        icdar_image_dir = f"{cfg.competition_dataset.icdar_dir}/images"
        print(f"adding  data from {icdar_image_dir}")
        icdar_ids = glob.glob(f"{icdar_image_dir}/*.jpg")
        icdar_ids = [fid.split("/")[-1].split(".jpg")[0] for fid in icdar_ids]
        print(f"# ICDAR ids before exclusion: {len(icdar_ids)}")
        icdar_ids = [gid for gid in icdar_ids if 'scatter' not in gid]
        icdar_ids = [gid for gid in icdar_ids if gid not in EXCLUDE_IDS]
        print(f"# ICDAR ids after exclusion: {len(icdar_ids)}")

        print(f"# ICDAR before multiplier: {len(icdar_ids)}")
        icdar_ids = icdar_ids*cfg.icdar_multiplier
        print(f"# ICDAR after multiplier: {len(icdar_ids)}")

        train_ids.extend(icdar_ids)
        print(f"# of graphs in train: {len(train_ids)}")
        print_line()

    # ------- Datasets ------------------------------------------------------------------#
    # The datasets for LECR Dual Encoder
    # -----------------------------------------------------------------------------------#

    train_transforms = None
    if cfg.use_augmentations:
        print_line()
        print("using augmentations...")
        train_transforms = create_train_transforms()
        print_line()

    mga_train_ds = MGADataset(cfg, train_ids, transform=train_transforms)
    mga_valid_ds = MGADataset(cfg, valid_ids)

    tokenizer = mga_train_ds.processor.tokenizer
    cfg.model.len_tokenizer = len(tokenizer)

    cfg.model.pad_token_id = tokenizer.pad_token_id
    cfg.model.decoder_start_token_id = tokenizer.convert_tokens_to_ids([BOS_TOKEN])[0]
    cfg.model.bos_token_id = tokenizer.convert_tokens_to_ids([BOS_TOKEN])[0]

    # ------- data collators --------------------------------------------------------------#
    collate_fn = MGACollator(tokenizer=tokenizer)

    train_dl = DataLoader(
        mga_train_ds,
        batch_size=cfg.train_params.train_bs,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=2,
    )

    valid_dl = DataLoader(
        mga_valid_ds,
        batch_size=cfg.train_params.valid_bs,
        collate_fn=collate_fn,
        shuffle=False,
    )

    # ------- Wandb --------------------------------------------------------------------#
    print_line()
    if cfg.use_wandb:
        print("initializing wandb run...")
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        init_wandb(cfg_dict)
    print_line()

    # --- show batch--------------------------------------------------------------------#
    print_line()
    for idx, b in enumerate(train_dl):
        if idx == 16:
            break
        run_sanity_check(cfg, b, tokenizer, prefix=f"train_{idx}")

    for idx, b in enumerate(valid_dl):
        if idx == 4:
            break
        run_sanity_check(cfg, b, tokenizer, prefix=f"valid_{idx}")

    # ------- Config -------------------------------------------------------------------#
    print("config for the current run")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    print(json.dumps(cfg_dict, indent=4))

    # ------- Model --------------------------------------------------------------------#
    print_line()
    print("creating the MGA model...")
    model = MGAModel(cfg)  # get_model(cfg)
    print_line()

    print("loading model from previously trained checkpoint...")
    checkpoint = cfg.model.ckpt_path
    ckpt = torch.load(checkpoint)
    model.load_state_dict(ckpt['state_dict'])
    del ckpt
    gc.collect()

    # # # torch 2.0
    # model = model.to("cuda:0")
    # model = torch.compile(model)  # pytorch 2.0

    # ------- Optimizer ----------------------------------------------------------------#
    print_line()
    print("creating the optimizer...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )

    # ------- Scheduler -----------------------------------------------------------------#
    print_line()
    print("creating the scheduler...")

    num_epochs = cfg.train_params.num_epochs
    grad_accumulation_steps = cfg.train_params.grad_accumulation
    warmup_pct = cfg.train_params.warmup_pct

    num_update_steps_per_epoch = len(train_dl)//grad_accumulation_steps
    num_training_steps = num_epochs * num_update_steps_per_epoch

    num_warmup_steps = int(warmup_pct*num_training_steps)

    print(f"# training updates per epoch: {num_update_steps_per_epoch}")
    print(f"# training steps: {num_training_steps}")
    print(f"# warmup steps: {num_warmup_steps}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    # #------- AWP --------------------------------------------------------------------------#
    AWP_FLAG = False

    # AWP
    if cfg.awp.use_awp:
        awp = AWP(model, optimizer, adv_lr=cfg.awp.adv_lr, adv_eps=cfg.awp.adv_eps)

    # ------- Accelerator --------------------------------------------------------------#
    print_line()
    print("accelerator setup...")

    accelerator = Accelerator(
        mixed_precision='bf16',  # changed 'fp16' to 'bf16'
    )  # cpu = True

    model, optimizer, train_dl, valid_dl = accelerator.prepare(
        model, optimizer, train_dl, valid_dl)

    print("model preparation done...")
    print(f"current GPU utilization...")
    print_gpu_utilization()
    print_line()

    # ------- training setup --------------------------------------------------------------#
    best_lb = -1.
    save_trigger = cfg.train_params.save_trigger

    patience_tracker = 0
    current_iteration = 0

    # ------- EMA -----------------------------------------------------------------------#
    if cfg.train_params.use_ema:
        print_line()
        decay_rate = cfg.train_params.decay_rate
        ema = EMA(model, decay=decay_rate)
        ema.register()

        print(f"EMA will be used during evaluation with decay {round(decay_rate, 4)}...")
        print_line()

    # ------- training  --------------------------------------------------------------------#
    start_time = time.time()
    num_vbar = 0
    num_hbar = 0
    num_histogram = 0
    num_dot = 0
    num_line = 0
    num_scatter = 0

    for epoch in range(num_epochs):
        # AWP Flag check
        if (cfg.awp.use_awp) & (epoch >= cfg.awp.awp_trigger_epoch):
            print("AWP is triggered...")
        epoch_progress = 0
        # close and reset progress bar
        if epoch != 0:
            progress_bar.close()

        progress_bar = tqdm(range(num_update_steps_per_epoch))
        loss_meter = AverageMeter()
        loss_meter_main = AverageMeter()
        loss_meter_cls = AverageMeter()

        model.train()
        for step, batch in enumerate(train_dl):
            num_vbar += len([ct for ct in batch['chart_type'] if ct == 'vertical_bar'])
            num_hbar += len([ct for ct in batch['chart_type'] if ct == 'horizontal_bar'])
            num_histogram += len([ct for ct in batch['chart_type'] if ct == 'histogram'])
            num_dot += len([ct for ct in batch['chart_type'] if ct == 'dot'])
            num_line += len([ct for ct in batch['chart_type'] if ct == 'line'])
            num_scatter += len([ct for ct in batch['chart_type'] if ct == 'scatter'])

            loss, loss_dict = model(
                flattened_patches=batch["flattened_patches"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            accelerator.backward(loss)
            epoch_progress += 1
            if AWP_FLAG:
                awp.attack_backward(batch, accelerator)

            if (step + 1) % grad_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip_value)  # added gradient clip

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                loss_meter.update(loss.item())
                loss_meter_main.update(loss_dict["loss_main"].item())
                loss_meter_cls.update(loss_dict["loss_cls"].item())

                # ema ---
                if cfg.train_params.use_ema:
                    ema.update()

                progress_bar.set_description(
                    f"STEP: {epoch_progress+1:5}/{len(train_dl):5}. "
                    f"T-STEP: {current_iteration+1:5}/{num_training_steps:5}. "
                    f"LR: {get_lr(optimizer):.4f}. "
                    f"Loss: {loss_meter.avg:.4f}. "
                )
                progress_bar.update(1)
                current_iteration += 1

                if cfg.use_wandb:
                    wandb.log({"train_loss": round(loss_meter.avg, 5)}, step=current_iteration)
                    wandb.log({"main_loss": round(loss_meter_main.avg, 5)}, step=current_iteration)
                    wandb.log({"cls_loss": round(loss_meter_cls.avg, 5)}, step=current_iteration)

                    wandb.log({"num_vbar": num_vbar}, step=current_iteration)
                    wandb.log({"num_hbar": num_hbar}, step=current_iteration)
                    wandb.log({"num_histogram": num_histogram}, step=current_iteration)
                    wandb.log({"num_dot": num_dot}, step=current_iteration)
                    wandb.log({"num_line": num_line}, step=current_iteration)
                    wandb.log({"num_scatter": num_scatter}, step=current_iteration)

                    wandb.log({"lr": get_lr(optimizer)}, step=current_iteration)

            # >--------------------------------------------------|
            # >-- evaluation ------------------------------------|
            # >--------------------------------------------------|

            if (epoch_progress + 1) % cfg.train_params.eval_frequency == 0:
                print("\n")
                print("GPU Utilization before evaluation...")
                print_gpu_utilization()

                # set model in eval mode
                model.eval()

                # apply ema if it is used ---
                if cfg.train_params.use_ema:
                    ema.apply_shadow()

                result_dict = run_evaluation(
                    cfg,
                    model=model,
                    valid_dl=valid_dl,
                    label_df=label_df,
                    tokenizer=tokenizer,
                    token_map=TOKEN_MAP,
                )

                lb = result_dict["lb"]
                oof_df = result_dict["oof_df"]
                result_df = result_dict["result_df"]

                print_line()
                et = as_minutes(time.time()-start_time)
                print(f">>> Epoch {epoch+1} | Step {step} | Total Step {current_iteration} | Time: {et}")

                is_best = False
                if lb >= best_lb:
                    best_lb = lb
                    is_best = True
                    patience_tracker = 0

                    # ---
                    best_dict = dict()
                    for k, v in result_dict.items():
                        if "df" not in k:
                            best_dict[f"{k}_at_best"] = v

                else:
                    patience_tracker += 1

                print_line()
                print(f">>> Current LB = {round(lb, 4)}")
                for k, v in result_dict.items():
                    if ("df" not in k) & (k != "lb"):
                        print(f">>> Current {k}={round(v, 4)}")
                print_line()

                if is_best:
                    oof_df.to_csv(os.path.join(cfg.outputs.model_dir, f"oof_df_fold_{fold}_best.csv"), index=False)
                    result_df.to_csv(os.path.join(cfg.outputs.model_dir, f"result_df_fold_{fold}_best.csv"), index=False)

                else:
                    print(f">>> patience reached {patience_tracker}/{cfg_dict['train_params']['patience']}")
                    print(f">>> current best score: {round(best_lb, 4)}")

                oof_df.to_csv(os.path.join(cfg_dict["outputs"]["model_dir"], f"oof_df_fold_{fold}.csv"), index=False)
                result_df.to_csv(os.path.join(cfg.outputs.model_dir, f"result_df_fold_{fold}.csv"), index=False)

                # save pickle for analysis
                result_df.to_pickle(os.path.join(cfg.outputs.model_dir, f"result_df_fold_{fold}.pkl"))

                # saving -----
                accelerator.wait_for_everyone()
                model = accelerator.unwrap_model(model)
                model_state = {
                    'step': current_iteration,
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'lb': lb,
                }

                if best_lb > save_trigger:
                    save_checkpoint(cfg_dict, model_state, is_best=is_best)

                # logging ----
                if cfg.use_wandb:
                    wandb.log({"lb": lb}, step=current_iteration)
                    wandb.log({"best_lb": best_lb}, step=current_iteration)

                    # ----
                    for k, v in result_dict.items():
                        if "df" not in k:
                            wandb.log({k: round(v, 4)}, step=current_iteration)

                    # --- log best scores dict
                    for k, v in best_dict.items():
                        if "df" not in k:
                            wandb.log({k: round(v, 4)}, step=current_iteration)

                # awp
                if (cfg.awp.use_awp) & (best_lb >= cfg.awp.awp_trigger):
                    print("AWP is triggered...")
                    AWP_FLAG = True

                # -- post eval
                model.train()
                torch.cuda.empty_cache()

                # ema ---
                if cfg.train_params.use_ema:
                    ema.restore()

                print_line()

                # early stopping ----
                if patience_tracker >= cfg_dict['train_params']['patience']:
                    print("stopping early")
                    model.eval()
                    return


if __name__ == "__main__":
    run_training()
