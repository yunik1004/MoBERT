from __future__ import absolute_import
import argparse
import torch
from transformers import BertForQuestionAnswering


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Pre-train the BERT or MoBERT")
    parser.add_argument(
        "--pretrain",
        "-p",
        default="./checkpoint",
        help="Directory of the pretrained model",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./checkpoint/qa",
        help="Directory of the trained qa model",
    )
    args = parser.parse_args()

    # Hyperparameters
    EPOCHS = 10
    LEARNING_RATE = 5e-5

    # Select the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Generate question answering model using pretrained bert
    model = BertForQuestionAnswering.from_pretrained(args.pretrain, num_labels=2)
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

        model.save_pretrained(args.output)
