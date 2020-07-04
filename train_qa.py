from __future__ import absolute_import, division
import argparse
import csv
import datetime
import os
from pathlib import Path
import time
import torch
from tqdm.auto import tqdm
from transformers import BertForQuestionAnswering
from dataset_korquad import KorquadDataset
from evaluate_korquad import exact_match_score, f1_score


def inference_start_end(
    start_probs: torch.Tensor,
    end_probs: torch.Tensor,
    context_start_pos: int,
    context_end_pos: int,
):
    """ Inference fucntion for the start and end token position.
    Find the start and end positions of the answer which maximize
    p(start, end | context_start_pos <= start <= end <= context_end_pos)
    Note: assume that p(start) and p(end) are independent.
    Hint: torch.tril or torch.triu function would be helpful.
    Arguments:
    start_probs -- Probability tensor for the start position
                    in shape (sequence_length, )
    end_probs -- Probatility tensor for the end position
                    in shape (sequence_length, )
    context_start_pos -- Start index of the context
    context_end_pos -- End index of the context
    """
    assert start_probs.sum().allclose(torch.scalar_tensor(1.0))
    assert end_probs.sum().allclose(torch.scalar_tensor(1.0))

    ### YOUR CODE HERE (~6 lines)
    start_pos: int = context_start_pos
    end_pos: int = context_start_pos

    prob = torch.triu(
        torch.ger(
            start_probs[context_start_pos : context_end_pos + 1],
            end_probs[context_start_pos : context_end_pos + 1],
        )
    )

    values, indices1 = torch.max(prob, 0)
    _, indices2 = torch.max(values, 0)

    index_col = indices2.item()
    index_row = indices1.data[index_col].item()

    start_pos += index_row
    end_pos += index_col

    ### END YOUR CODE

    return start_pos, end_pos


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train the question answering model using pre-trained BERT or MoBERT"
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
        default=os.path.join(".", "checkpoint", "qa"),
        help="Output directory of the trained qa model",
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
    qa_model_dir = os.path.join(
        args.output, datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    )

    log_path = os.path.join(qa_model_dir, "log.csv")

    # If not exist, then create directories
    Path(qa_model_dir).mkdir(parents=True, exist_ok=True)

    # Open the log file
    log_f = open(log_path, "w")
    log_writer = csv.writer(log_f)

    # Hyperparameters
    EPOCHS = 100
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 32

    # Generate the data loader
    dataset = KorquadDataset()
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)]
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=train_dataset.dataset.collate_fn,
        num_workers=2,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=val_dataset.dataset.collate_fn,
        num_workers=2,
    )  # shuffle is not needed for validation

    # Generate question answering model using pretrained bert
    model = BertForQuestionAnswering.from_pretrained(bert_dir, num_labels=2)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        total_train_loss = 0.0
        total_train_em = 0.0
        total_train_f1 = 0.0

        # Training
        model.train()
        for _, (input_ids, attention_mask, token_type_ids, start_pos, end_pos) in tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):
            optimizer.zero_grad()

            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            start_pos = start_pos.to(device)
            end_pos = end_pos.to(device)

            out = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                start_positions=start_pos,
                end_positions=end_pos,
            )
            loss = out[0]
            start_logits = out[1]
            end_logits = out[2]

            """
            start_ids = torch.argmax(start_logits, dim=1)
            end_ids = torch.argmax(end_logits, dim=1)
            """
            start_probs = torch.nn.functional.softmax(start_logits, dim=1)
            end_probs = torch.nn.functional.softmax(end_logits, dim=1)

            loss.backward()
            optimizer.step()

            num_batch = input_ids.size(0)
            total_train_loss += loss.item() * num_batch

            for b in range(num_batch):
                start_pos_pred, end_pos_pred = inference_start_end(
                    start_probs[b], end_probs[b], 0, input_ids.size(1)
                )

                token_ids_ans = input_ids[b, start_pos[b] : end_pos[b] + 1].tolist()
                token_ids_pred = input_ids[
                    b, start_pos_pred : end_pos_pred + 1
                ].tolist()

                decode_ans = dataset.decode(token_ids_ans)
                decode_pred = dataset.decode(token_ids_pred)

                em = exact_match_score(decode_pred, decode_ans)
                f1 = f1_score(decode_pred, decode_ans)

                total_train_em += em
                total_train_f1 += f1

        avg_train_loss = total_train_loss / len(train_dataset)
        avg_train_em = total_train_em / len(train_dataset)
        avg_train_f1 = total_train_f1 / len(train_dataset)

        # Validation
        total_val_loss = 0.0
        total_val_em = 0.0
        total_val_f1 = 0.0

        model.eval()
        with torch.no_grad():
            for (
                input_ids,
                attention_mask,
                token_type_ids,
                start_pos,
                end_pos,
            ) in val_loader:
                input_ids = input_ids.to(device)
                token_type_ids = token_type_ids.to(device)
                attention_mask = attention_mask.to(device)
                start_pos = start_pos.to(device)
                end_pos = end_pos.to(device)

                out = model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    start_positions=start_pos,
                    end_positions=end_pos,
                )
                loss = out[0]
                start_logits = out[1]
                end_logits = out[2]

                """
                start_ids = torch.argmax(start_logits, dim=1)
                end_ids = torch.argmax(end_logits, dim=1)
                """
                start_probs = torch.nn.functional.softmax(start_logits, dim=1)
                end_probs = torch.nn.functional.softmax(end_logits, dim=1)

                num_batch = input_ids.size(0)
                total_val_loss += loss.item() * num_batch

                for b in range(num_batch):
                    start_pos_pred, end_pos_pred = inference_start_end(
                        start_probs[b], end_probs[b], 0, input_ids.size(1)
                    )

                    token_ids_ans = input_ids[b, start_pos[b] : end_pos[b] + 1].tolist()
                    """
                    token_ids_pred = input_ids[
                        b, start_ids[b] : end_ids[b] + 1
                    ].tolist()
                    """
                    token_ids_pred = input_ids[
                        b, start_pos_pred : end_pos_pred + 1
                    ].tolist()

                    decode_ans = dataset.decode(token_ids_ans)
                    decode_pred = dataset.decode(token_ids_pred)

                    em = exact_match_score(decode_pred, decode_ans)
                    f1 = f1_score(decode_pred, decode_ans)

                    total_val_em += em
                    total_val_f1 += f1

        avg_val_loss = total_val_loss / len(val_dataset)
        avg_val_em = total_val_em / len(val_dataset)
        avg_val_f1 = total_val_f1 / len(val_dataset)

        # Show the output
        elapsed_time = time.time() - start
        print(
            f"[Epoch {epoch}] Elapsed time: {elapsed_time:.3f}\tAverage train loss: {avg_train_loss:.3f}\tAverage train EM: {avg_train_em:.3f}\tAverage train F1-score: {avg_train_f1:.3f}\tAverage validation loss: {avg_val_loss:.3f}\tAverage validation EM: {avg_val_em:.3f}\tAverage validation F1-score: {avg_val_f1:.3f}"
        )

        log_writer.writerow(
            [
                epoch,
                elapsed_time,
                avg_train_loss,
                avg_train_em,
                avg_train_f1,
                avg_val_loss,
                avg_val_em,
                avg_val_f1,
            ]
        )
        log_f.flush()

        bert_dir_epoch = os.path.join(qa_model_dir, str(epoch))
        Path(bert_dir_epoch).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(bert_dir_epoch)

    log_f.close()
