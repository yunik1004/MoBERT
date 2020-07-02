from __future__ import absolute_import, division
import argparse
import csv
import datetime
import os
from pathlib import Path
import time
import torch
from tqdm.auto import tqdm
from transformers import BertForSequenceClassification
from dataset_nsmc import NaverSentimentDataset


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train the classification model using pre-trained BERT or MoBERT"
    )
    parser.add_argument(
        "--pretrain",
        "-p",
        default=os.path.join(".", "bestmodel", "mobert"),
        help="Directory of the pretrained bert model",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.join(".", "checkpoint", "cl"),
        help="Output directory of the trained classification model",
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

    # Set the directories where the models will be saved
    bert_dir = args.pretrain

    cl_model_dir = os.path.join(
        args.output, datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    )

    log_path = os.path.join(cl_model_dir, "log.csv")

    # If not exist, then create directories
    Path(cl_model_dir).mkdir(parents=True, exist_ok=True)

    # Open the log file
    log_f = open(log_path, "w")
    log_writer = csv.writer(log_f)

    # Hyperparameters
    EPOCHS = 100
    LEARNING_RATE = 5e-5
    BATCH_SIZE = 32

    # Generate the data loader
    dataset = NaverSentimentDataset()
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)]
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=train_dataset.dataset.collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=val_dataset.dataset.collate_fn,
    )  # shuffle is not needed for validation

    # Generate question answering model using pretrained bert
    model = BertForSequenceClassification.from_pretrained(bert_dir, num_labels=1)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        total_train_loss = 0
        num_train_data = 0

        # Training
        model.train()
        for _, (input_ids, token_type_ids, attention_mask, rating) in tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):
            optimizer.zero_grad()

            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            rating = rating.float().to(device)

            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=rating,
            )[0]
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            num_train_data += 1

        avg_train_loss = total_train_loss / num_train_data

        # Validation
        total_val_loss = 0
        num_val_data = 0

        model.eval()
        with torch.no_grad():
            for input_ids, token_type_ids, attention_mask, rating in val_loader:
                input_ids = input_ids.to(device)
                token_type_ids = token_type_ids.to(device)
                attention_mask = attention_mask.to(device)
                rating = rating.float().to(device)

                loss = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=rating,
                )[0]

                total_val_loss += loss.item()
                num_val_data += 1

        avg_val_loss = total_val_loss / num_val_data

        # Show the output
        elapsed_time = time.time() - start
        print(
            f"[Epoch {epoch}] Elapsed time: {elapsed_time:.3f}\tAverage train loss: {avg_train_loss:.3f}\tAverage validation loss: {avg_val_loss:.3f}"
        )
        log_writer.writerow([epoch, elapsed_time, avg_train_loss, avg_val_loss])

        bert_dir_epoch = os.path.join(cl_model_dir, str(epoch))
        Path(bert_dir_epoch).mkdir(parents=True, exist_ok=True)

        model.save_pretrained(bert_dir_epoch)

    log_f.close()
