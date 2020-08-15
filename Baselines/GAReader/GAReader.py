# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch

from Baselines.Models.UnOrderedLSTM import LSTM
from Baselines.Models.Linear import Linear
from Baselines.Models.MLPAttention import MLPAttention

def gated_attention(article, question):
    """
    Args:
        article: [batch_size, article_len , dim]
        question: [batch_size, question_len, dim]
    Returns:
        question_to_article: [batch_size, article_len, dim]
    """
    question_att = question.permute(0, 2, 1)
    # question : [batch_size * dim * question_len]

    att_matrix = torch.bmm(article, question_att)
    # att_matrix: [batch_size * article_len * question_len]

    att_weights = F.softmax(att_matrix.view(-1, att_matrix.size(-1)), dim=1).view_as(att_matrix)
    # att_weights: [batch_size, article_len, question_len]

    question_rep = torch.bmm(att_weights, question)
    # question_rep : [batch_size, article_len, dim]

    question_to_article = torch.mul(article, question_rep)
    # question_to_article: [batch_size, article_len, dim]

    return question_to_article



class GAReader(nn.Module):
    """
    Some difference between our GAReader and the original GAReader
    1. The query GRU is shared across hops.
    2. Dropout is applied to all hops (including the initial hop).
    3. Gated-attention is applied at the final layer as well.
    4. No character-level embeddings are used.
    """

    def __init__(self, embedding_dim, output_dim, hidden_size, rnn_num_layers, ga_layers, bidirectional, dropout, word_emb):
        super(GAReader, self).__init__()

        self.word_embedding = nn.Embedding.from_pretrained(word_emb, freeze=False)

        self.rnn = LSTM(embedding_dim, hidden_size, True,
                        rnn_num_layers, bidirectional, dropout)

        self.ga_rnn = LSTM(hidden_size * 2, hidden_size, True,
                           rnn_num_layers, bidirectional, dropout)
        
        self.ga_layers = ga_layers

        self.mlp_att = MLPAttention(hidden_size * 2, dropout)

        self.dot_layer = MLPAttention(hidden_size * 2, dropout)

        self.final_liear = Linear(hidden_size * 10, output_dim)


        self.dropout = nn.Dropout(dropout)
    
    def forward(self, batch):

        article, article_lengths = batch.article
        # article: [article_len, batch_size], article_lengths: [batch_size]

        question, question_lengths = batch.question
        # question: [question_len, batch_size], question_lengths: [batch_size]

        option0, option0_lengths = batch.option_0
        option1, option1_lengths = batch.option_1
        option2, option2_lengths = batch.option_2
        option3, option3_lengths = batch.option_3
        option4, option4_lengths = batch.option_4
        # option: [option_len, batch_size]

        article_emb = self.dropout(self.word_embedding(article))
        # article_emb: [article_len, batch_size, emd_dim]

        question_emb = self.dropout(self.word_embedding(question))
        # question_emb: [question_len, batch_size, emd_dim]

        option0_emb = self.dropout(self.word_embedding(option0))
        option1_emb = self.dropout(self.word_embedding(option1))
        option2_emb = self.dropout(self.word_embedding(option2))
        option3_emb = self.dropout(self.word_embedding(option3))
        option4_emb = self.dropout(self.word_embedding(option4))
        # option: [option_len, batch_size, emd_dim]

        article_emb = article_emb.permute(1, 0, 2)  # [batch_size, seq_len, dim]
        question_emb = question_emb.permute(1, 0, 2)
        option0_emb = option0_emb.permute(1, 0, 2)
        option1_emb = option1_emb.permute(1, 0, 2)
        option2_emb = option2_emb.permute(1, 0, 2)
        option3_emb = option3_emb.permute(1, 0, 2)
        option4_emb = option4_emb.permute(1, 0, 2)

        question_hidden, question_out = self.rnn(question_emb, question_lengths)
        # question_out: [batch_size, question_len, hidden_size * 2]
        # question_hidden: [batch_size, hidden_size * 2]

        option0_hidden, option0_out = self.rnn(option0_emb, option0_lengths)
        option1_hidden, option1_out = self.rnn(option1_emb, option1_lengths)
        option2_hidden, option2_out = self.rnn(option2_emb, option2_lengths)
        option3_hidden, option3_out = self.rnn(option3_emb, option3_lengths)
        option4_hidden, option4_out = self.rnn(option4_emb, option4_lengths)
        # option_out: [batch_size, option_len,  hidden_size * 2]

        _, article_out = self.rnn(article_emb, article_lengths)
        # article_out: [article_len, batch_size, hidden_size * 2]


        for layer in range(self.ga_layers):
                        
            article_emb = self.dropout(gated_attention(article_out, question_out))
            # article_emb: [batch_size, article_len, hidden_size * 2]

            _, article_out = self.ga_rnn(article_emb, article_lengths)
            # article_out: [batch_size, article_len, hidden_size * 2]
        
        ATT_article_question = self.dropout(self.mlp_att(question_hidden, article_out, article_out))
        # ATT_article_question: [batch_size, hidden_size * 2]
        
        # 融合 option 信息 [batch_size, hidden_size * 2]
        ATT_option0 = self.dropout(self.dot_layer(
            ATT_article_question, option0_out, option0_out))
        ATT_option1 = self.dropout(self.dot_layer(
            ATT_article_question, option1_out, option1_out))
        ATT_option2 = self.dropout(self.dot_layer(
            ATT_article_question, option2_out, option2_out))
        ATT_option3 = self.dropout(self.dot_layer(
            ATT_article_question, option3_out, option3_out))
        ATT_option4 = self.dropout(self.dot_layer(
            ATT_article_question, option4_out, option4_out))
        
        all_infomation = torch.cat((ATT_option0, ATT_option1, ATT_option2, ATT_option3, ATT_option4), dim=1)

        logit = self.dropout(self.final_liear(all_infomation))

        return logit



        
        








            









        












