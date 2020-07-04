from __future__ import absolute_import, division
import argparse
import os
import torch
from tqdm import tqdm
from transformers import BertForSequenceClassification
from dataset_nsmc import NaverSentimentDataset


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train the question answering model using pre-trained BERT or MoBERT"
    )
    parser.add_argument(
        "--pretrain",
        "-p",
        default=os.path.join(".", "bestmodel", "mobert_cl"),
        help="Directory of the trained bert model",
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

    # Generate the data loader
    test_dataset = NaverSentimentDataset(train=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
        num_workers=2,
    )

    # Generate question answering model using pretrained bert
    model = BertForSequenceClassification.from_pretrained(bert_dir)
    model = model.to(device)

    # Evaluation
    total_test_loss = 0.0
    total_test_accuracy = 0

    model.eval()
    with torch.no_grad():
        for input_ids, token_type_ids, attention_mask, rating in tqdm(test_loader):
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            rating_bool = rating.bool().to(device)
            rating = rating.float().to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=rating,
            )
            loss = out[0]
            logits = (out[1] >= 0.5).squeeze()
            accurate = ~(rating_bool ^ logits)
            accuracy = accurate.sum()

            total_test_loss += loss.item() * rating.size(0)
            total_test_accuracy += accuracy.item()

    avg_test_loss = total_test_loss / len(test_dataset)
    avg_test_accuracy = total_test_accuracy / len(test_dataset)

    print(
        f"Average test loss: {avg_test_loss:.3f}\tAverage test accuracy: {avg_test_accuracy:.3f}"
    )
