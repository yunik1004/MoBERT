from __future__ import absolute_import
import argparse
import os
from pathlib import Path
import torch
from transformers import BertForQuestionAnswering


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train the question answering model using pre-trained BERT or MoBERT"
    )
    parser.add_argument(
        "--pretrain",
        "-p",
        default=os.path.join(".", "checkpoint", "pretrain", "mobert_bert"),
        help="Directory of the pretrained bert model",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.join(".", "checkpoint", "qa"),
        help="Output directory of the trained qa model",
    )
    args = parser.parse_args()

    # Set the directories where the models will be saved
    bert_dir = args.pretrain
    qa_model_dir = args.output

    # If not exist, then create directories
    Path(qa_model_dir).mkdir(parents=True, exist_ok=True)

    # Hyperparameters
    EPOCHS = 10
    LEARNING_RATE = 5e-5

    # Select the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Generate question answering model using pretrained bert
    model = BertForQuestionAnswering.from_pretrained(bert_dir, num_labels=2)
    model = model.to(device)

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

        model.save_pretrained(qa_model_dir)
