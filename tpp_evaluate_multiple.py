#!/usr/bin/env python
import time
import torch

from tqdm import tqdm
from pathlib import Path

from tpp_train import args as default_args
from tpp_train import SetSeed
from datasets.tpp_loader import load_dataset
from models.tpp_warper import TPPWarper
from models.prob_decoders import get_decoder
from models.embeddings import get_embedding
from models.hist_encoders import get_encoder
from models.libs.logger import remove_file_handlers
from trainers.trainer import Trainer

default_args["dataset_dir"] = Path("./data")
default_args["log_dir"] = "experiments"
default_args["device"] = torch.device("cuda:6")

def evaluate_multiple():
    exps_path = [p.stem for p in Path("./experiments/").glob("*") if p.is_dir()]
    with tqdm(exps_path) as t:
        for exp in t:
            hist_enc, prob_dec, dataset_dir, seed = str(exp).split("_")

            dataset_dir = default_args["dataset_dir"] / dataset_dir
            seed = int(seed)

            args = default_args.copy()
            args.update(
                {
                    "hist_enc": hist_enc,
                    "prob_dec": prob_dec,
                    "dataset_dir": f"{dataset_dir}/",
                    "seed": seed,
                }
            )

            SetSeed(args["seed"])

            t.clear()
            (
                data,
                event_type_num,
                seq_lengths,
                max_length,
                max_t,
                mean_log_dt,
                std_log_dt,
                max_dt,
            ) = load_dataset(**args)
            t.refresh()

            args["event_type_num"] = int(event_type_num)
            args["max_length"] = int(max_length)
            args["max_t"] = max_t
            args["mean_log_dt"] = mean_log_dt
            args["std_log_dt"] = std_log_dt
            args["max_dt"] = max_dt
            args["experiment_name"] = "{}_{}_{}_{}".format(
                args["hist_enc"],
                args["prob_dec"],
                args["dataset_dir"].split("/")[-2],
                args["seed"],
            )

            time_embedding, type_embedding, position_embedding = get_embedding(**args)
            hist_encoder = get_encoder(**args)
            prob_decoder = get_decoder(**args)

            model = TPPWarper(
                time_embedding=time_embedding,
                type_embedding=type_embedding,
                position_embedding=position_embedding,
                encoder=hist_encoder,
                decoder=prob_decoder,
                **args,
            )

            trainer = Trainer(data=data, model=model, seq_length=seq_lengths, **args)

            trainer._logger.info("Updated QQDEV v2.")
            if "Determ" in exp:
                metrics = ["LOG_LOSS", "CE", "MAPE", "TOP1_ACC", "TOP3_ACC", "QQDEV"]
            else:
                metrics = ["QQDEV"]
            trainer.final_test(n=1, metrics=metrics)

            trainer._logger.handlers.clear()
            time.sleep(1)


if __name__ == "__main__":
    evaluate_multiple()
