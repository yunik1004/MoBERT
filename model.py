from __future__ import absolute_import
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig
from transformers.modeling_bert import BertPreTrainingHeads


class MoBert(BertPreTrainedModel):
    """
    Korean BERT model for pre-training with morphological information
    """

    def __init__(self, config: BertConfig):
        super().__init__(config)

        self.num_labels = config.num_labels

        # BERT model that we want to train
        self.bert = BertModel(config)

        # For PreTraining
        self.cls = BertPreTrainingHeads(config)

        # For Morphological guessing
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        # Initialize weights
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        next_sentence_label=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs[:2]

        # Scores for PreTraining
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output
        )

        # Morphological classification score
        sequence_output2 = self.dropout(sequence_output)
        logits = self.classifier(sequence_output2)

        # Add hidden_states and attentions if they exist
        outputs = (prediction_scores, seq_relationship_score, logits,) + outputs[2:]

        if (
            masked_lm_labels is not None
            and next_sentence_label is not None
            and labels is not None
        ):
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)

            # Loss for pretraining
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1),
            )
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
            )

            loss_pre = masked_lm_loss + next_sentence_loss

            # Loss for classification
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss_cl = loss_fct(active_logits, active_labels)
            else:
                loss_cl = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            # Total loss = sum of two losses
            total_loss = loss_pre + loss_cl

            outputs = (total_loss,) + outputs + (loss_pre, loss_cl)

        return outputs  # (loss), prediction_scores, seq_relationship_scores, logits, (hidden_states), (attentions), (loss_pre), (loss_cl)
