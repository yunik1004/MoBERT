from __future__ import absolute_import, division
import argparse
import os
import torch
from tqdm import tqdm
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
        default=os.path.join(".", "bestmodel", "mobert_qa"),
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
    test_dataset = KorquadDataset(train=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=test_dataset.collate_fn,
    )

    """
    dataset = KorquadDataset()
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)]
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn,
    )
    """

    # Generate question answering model using pretrained bert
    model = BertForQuestionAnswering.from_pretrained(bert_dir, num_labels=2)
    model = model.to(device)

    # Validation
    total_test_loss = 0.0
    total_test_em = 0.0
    total_test_f1 = 0.0

    model.eval()
    with torch.no_grad():
        for (input_ids, attention_mask, token_type_ids, start_pos, end_pos,) in tqdm(
            test_loader
        ):
            input_ids = input_ids[:, :512].to(device)
            token_type_ids = token_type_ids[:, :512].to(device)
            attention_mask = attention_mask[:, :512].to(device)
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

            # start_ids = torch.argmax(start_logits, dim=1)
            # end_ids = torch.argmax(end_logits, dim=1)
            start_probs = torch.nn.functional.softmax(start_logits, dim=1)
            end_probs = torch.nn.functional.softmax(end_logits, dim=1)

            num_batch = input_ids.size(0)
            total_test_loss += loss.item() * num_batch

            for b in range(num_batch):
                start_pos_pred, end_pos_pred = inference_start_end(
                    start_probs[b], end_probs[b], 0, input_ids.size(1)
                )

                token_ids_ans = input_ids[b, start_pos[b] : end_pos[b] + 1].tolist()
                token_ids_pred = input_ids[
                    b, start_pos_pred : end_pos_pred + 1
                ].tolist()

                decode_ans = test_dataset.decode(token_ids_ans)
                decode_pred = test_dataset.decode(token_ids_pred)

                """
                decode_ans = dataset.decode(token_ids_ans)
                decode_pred = dataset.decode(token_ids_pred)
                """

                """
                print(token_ids_ans)
                print("\n\n\n")
                print(token_ids_pred)
                print("\n\n\n")
                """

                em = exact_match_score(decode_pred, decode_ans)
                f1 = f1_score(decode_pred, decode_ans)

                total_test_em += em
                total_test_f1 += f1

        avg_test_loss = total_test_loss / len(test_dataset)
        avg_test_em = total_test_em / len(test_dataset)
        avg_test_f1 = total_test_f1 / len(test_dataset)

        print(
            f"Average test loss: {avg_test_loss:.3f}\tAverage test EM: {avg_test_em:.3f}\tAverage test F1-score: {avg_test_f1:.3f}"
        )
