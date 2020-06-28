from __future__ import absolute_import
import argparse
import os
from pathlib import Path
import sys
import torch
from transformers import BertForPreTraining, BertConfig
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
    args = parser.parse_args()

    # Set the directories where the models will be saved
    mobert_dir = os.path.join(args.output, "mobert")
    mobert_bert_dir = os.path.join(args.output, "mobert_bert")
    origin_dir = os.path.join(args.output, "origin")
    origin_bert_dir = os.path.join(args.output, "origin_bert")

    # If not exist, then create directories
    Path(mobert_dir).mkdir(parents=True, exist_ok=True)
    Path(mobert_bert_dir).mkdir(parents=True, exist_ok=True)
    Path(origin_dir).mkdir(parents=True, exist_ok=True)
    Path(origin_bert_dir).mkdir(parents=True, exist_ok=True)

    # Hyperparameters
    EPOCHS = 10
    LEARNING_RATE = 5e-5

    # Select the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # TODO: Set the config of the bert
    config = BertConfig()

    if args.target == "mobert":
        """
        Pre-train the MoBERT model
        """
        config.num_labels = 10
        model = MoBert(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        model.train()

        for epoch in range(1, EPOCHS + 1):
            """ TODO: implement dataloader
            for data in dataloader:
                optimizer.zero_grad()
                loss = model(data)[0]
                loss.backward()
                optimizer.step()
            """

            # Save the model in the checkpoint folder
            model.save_pretrained(mobert_dir)
            model.bert.save_pretrained(mobert_bert_dir)

    elif args.target == "bert":
        """
        Pre-train the original BERT model
        """
        model = BertForPreTraining(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        model.train()

        for epoch in range(1, EPOCHS + 1):
            """ TODO: implement dataloader
            for data in dataloader:
                optimizer.zero_grad()
                loss = model(data)[0]
                loss.backward()
                optimizer.step()
            """

            # Save the model in the checkpoint folder
            model.save_pretrained(origin_dir)
            model.bert.save_pretrained(origin_bert_dir)

    else:
        sys.exit(-1)
