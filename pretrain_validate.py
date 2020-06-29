from __future__ import absolute_import, division
import argparse
import os
import torch
from transformers import BertForPreTraining
from dataset import WikiDataset, PretrainDataset, pretrain_collate_fn
from model import MoBert


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Validate the pre-trained BERT or MoBERT"
    )
    parser.add_argument(
        "--target",
        "-t",
        default="mobert",
        choices=["mobert", "bert"],
        help="Choose the model to be pre-trained",
    )
    parser.add_argument(
        "--pretrain",
        "-p",
        default=os.path.join(".", "checkpoint", "pretrain", "mobert_bert"),
        help="Directory of the pretrained bert model",
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

    # Load the model
    if args.target == "mobert":
        model = MoBert.from_pretrained(args.pretrain)
    elif args.target == "bert":
        model = BertForPreTraining.from_pretrained(args.pretrain)
    model = model.to(device)

    # Hyperparameters
    TOTAL_STEP = 1000

    # Generate the data loader
    wikidata = WikiDataset(train=False)
    pretraindata = PretrainDataset(wikidata)
    dataloader = torch.utils.data.DataLoader(
        pretraindata, 1, collate_fn=pretrain_collate_fn
    )

    # Evaluation
    model.eval()

    step = 0
    total_loss = 0
    total_loss_pre = 0
    total_loss_cl = 0
    with torch.no_grad():
        for src, mlm, mask, nsp, mt, token_type_ids in dataloader:
            if step >= TOTAL_STEP:
                break

            src = src.to(device)
            mlm = mlm.to(device)
            mask = mask.to(device)
            nsp = nsp.to(device)
            mt = mt.to(device)
            token_type_ids = token_type_ids.to(device)

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

            total_loss += loss.item()
            if args.target == "mobert":
                total_loss_pre += loss_pre.item()
                total_loss_cl += loss_cl.item()

            step += 1

    avg_loss = total_loss / step
    if args.target == "mobert":
        avg_loss_pre = total_loss_pre / step
        avg_loss_cl = total_loss_cl / step
        print(
            f"Average loss: {avg_loss:.3f}\tAverage loss_pre: {avg_loss_pre:.3f}\tAverage loss_cl: {avg_loss_cl:.3f}"
        )
    elif args.target == "bert":
        print(f"Average loss: {avg_loss:.3f}")
