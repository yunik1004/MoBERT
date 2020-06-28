from __future__ import absolute_import, division
import argparse
import csv
import datetime
import os
from pathlib import Path
import time
import torch
from transformers import BertForPreTraining, BertConfig
from dataset import WikiDataset, PretrainDataset, pretrain_collate_fn
from model import MoBert


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Pre-train the BERT or MoBERT")
    parser.add_argument(
        "--target",
        "-t",
        default="mobert",
        choices=["mobert", "bert"],
        help="Choose the model to be pre-trained",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.join(".", "checkpoint", "pretrain"),
        help="Output directory of the models",
    )
    parser.add_argument(
        "--device",
        "-d",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device",
    )
    args = parser.parse_args()

    # Torch settings
    torch.set_num_threads(10)
    device = torch.device(args.device)

    checkpoint_folder = os.path.join(
        args.output, datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    )

    # Set the directories where the models will be saved
    mobert_dir = os.path.join(checkpoint_folder, "mobert")
    mobert_bert_dir = os.path.join(checkpoint_folder, "mobert_bert")
    origin_dir = os.path.join(checkpoint_folder, "origin")
    origin_bert_dir = os.path.join(checkpoint_folder, "origin_bert")

    log_path = os.path.join(checkpoint_folder, "log.csv")

    # If not exist, then create directories
    Path(mobert_dir).mkdir(parents=True, exist_ok=True)
    Path(mobert_bert_dir).mkdir(parents=True, exist_ok=True)
    Path(origin_dir).mkdir(parents=True, exist_ok=True)
    Path(origin_bert_dir).mkdir(parents=True, exist_ok=True)

    # Open the log file
    log_f = open(log_path, "w")
    log_writer = csv.writer(log_f)

    # Hyperparameters
    PRINT_STEP = 1000
    MODEL_SAVE_STEP = 50000
    BATCH_SIZE = 4
    LEARNING_RATE = 5e-5

    # Generate the data loader
    wikidata = WikiDataset()
    pretraindata = PretrainDataset(wikidata)
    dataloader = torch.utils.data.DataLoader(
        pretraindata, BATCH_SIZE, collate_fn=pretrain_collate_fn
    )

    # Set the config of the bert
    config = BertConfig(
        num_hidden_layers=4,
        hidden_size=312,
        intermediate_size=1200,
        max_position_embeddings=1024,
    )

    if args.target == "mobert":
        config.num_labels = pretraindata.token_num + 1
        model = MoBert(config)
    elif args.target == "bert":
        model = BertForPreTraining(config)
    model = model.to(device)

    # Pre-train the MoBERT model
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()

    step = 1
    total_loss = 0
    total_loss_pre = 0
    total_loss_cl = 0
    start = time.time()
    for src, mlm, mask, nsp, mt, token_type_ids in dataloader:
        src = src.to(device)
        mlm = mlm.to(device)
        mask = mask.to(device)
        nsp = nsp.to(device)
        mt = mt.to(device)
        token_type_ids = token_type_ids.to(device)
        optimizer.zero_grad()

        if args.target == "mobert":
            out = model(
                src,
                attention_mask=mask,
                token_type_ids=token_type_ids,
                masked_lm_labels=mlm,
                next_sentence_label=nsp,
                labels=mt,
            )
            loss = out[0]
            loss_pre = out[-2]
            loss_cl = out[-1]

        elif args.target == "bert":
            loss = model(
                src,
                attention_mask=mask,
                token_type_ids=token_type_ids,
                masked_lm_labels=mlm,
                next_sentence_label=nsp,
            )[0]

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if args.target == "mobert":
            total_loss_pre += loss_pre.item()
            total_loss_cl += loss_cl.item()

        if step % PRINT_STEP == 0:
            avg_loss = total_loss / PRINT_STEP
            elapsed_time = time.time() - start

            if args.target == "mobert":
                avg_loss_pre = total_loss_pre / PRINT_STEP
                avg_loss_cl = total_loss_cl / PRINT_STEP
                print(
                    f"[Step {step}] Elapsed time: {elapsed_time:.3f}\tAverage loss: {avg_loss:.3f}\tAverage loss_pre: {avg_loss_pre:.3f}\tAverage loss_cl: {avg_loss_cl:.3f}"
                )
                log_writer.writerow(
                    [step, elapsed_time, avg_loss, avg_loss_pre, avg_loss_cl]
                )
            elif args.target == "bert":
                print(
                    f"[Step {step}] Elapsed time: {elapsed_time:.3f}\tAverage loss: {avg_loss:.3f}"
                )
                log_writer.writerow([step, elapsed_time, avg_loss])

            log_f.flush()

            total_loss = 0
            total_loss_pre = 0
            total_loss_cl = 0

        if step % MODEL_SAVE_STEP == 0:
            # Save the model in the checkpoint folder
            mobert_dir_step = os.path.join(mobert_dir, str(step))
            mobert_bert_dir_step = os.path.join(mobert_bert_dir, str(step))
            Path(mobert_dir_step).mkdir(parents=True, exist_ok=True)
            Path(mobert_bert_dir_step).mkdir(parents=True, exist_ok=True)

            model.save_pretrained(mobert_dir_step)
            model.bert.save_pretrained(mobert_bert_dir_step)

        step += 1

    log_f.close()
