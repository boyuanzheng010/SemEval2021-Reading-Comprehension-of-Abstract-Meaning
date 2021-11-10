import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence


def hotfix_pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True):
    lengths = torch.as_tensor(lengths, dtype=torch.int64)
    lengths = lengths.cpu()
    if enforce_sorted:
        sorted_indices = None
    else:
        lengths, sorted_indices = torch.sort(lengths, descending=True)
        sorted_indices = sorted_indices.to(input.device)
        batch_dim = 0 if batch_first else 1
        input = input.index_select(batch_dim, sorted_indices)

    data, batch_sizes = \
        torch._C._VariableFunctions._pack_padded_sequence(input, lengths, batch_first)
    return PackedSequence(data, batch_sizes, sorted_indices)


class LSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        batch_first=False,
        num_layers=1,
        bidirectional=False,
        dropout=0.2,
    ):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )
        self.reset_params()
        self.dropout = nn.Dropout(p=dropout)

    def reset_params(self):
        for i in range(self.rnn.num_layers):
            nn.init.orthogonal_(getattr(self.rnn, f"weight_hh_l{i}"))
            nn.init.kaiming_normal_(getattr(self.rnn, f"weight_ih_l{i}"))
            nn.init.constant_(getattr(self.rnn, f"bias_hh_l{i}"), val=0)
            nn.init.constant_(getattr(self.rnn, f"bias_ih_l{i}"), val=0)
            bias = getattr(self.rnn, f"bias_hh_l{i}").detach()
            bias.chunk(4)[1].fill_(1)
            with torch.no_grad():
                setattr(self.rnn, f"bias_hh_l{i}",nn.Parameter(bias))

            if self.rnn.bidirectional:
                nn.init.orthogonal_(getattr(self.rnn, f"weight_hh_l{i}_reverse"))
                nn.init.kaiming_normal_(getattr(self.rnn, f"weight_ih_l{i}_reverse"))
                nn.init.constant_(getattr(self.rnn, f"bias_hh_l{i}_reverse"), val=0)
                nn.init.constant_(getattr(self.rnn, f"bias_ih_l{i}_reverse"), val=0)
                bias = getattr(self.rnn, f"bias_hh_l{i}_reverse").detach()
                bias.chunk(4)[1].fill_(1)
                with torch.no_grad():
                    setattr(self.rnn, f"bias_hh_l{i}_reverse",nn.Parameter(bias))

    def forward(self, x, x_len):
        # x: [batch_size, seq_len, dim], x_len:[batch_size]
        x_len_sorted, x_idx = torch.sort(x_len, descending=True)
        x_sorted = torch.index_select(x, dim=0, index=x_idx)
        sorted_x, x_ori_idx = torch.sort(x_idx)

        # x_packed = nn.utils.rnn.pack_padded_sequence(
        #     x_sorted, x_len_sorted, batch_first=True
        # )
        x_packed = hotfix_pack_padded_sequence(
            x_sorted, x_len_sorted, batch_first=True
        )
        x_packed, (hidden, c) = self.rnn(x_packed)

        x = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)[0]
        x = x.index_select(dim=0, index=x_ori_idx)

        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # hidden = hidden.permute(1, 0, 2).contiguous().view(-1,
        #                                          hidden.size(0) * hidden.size(2)).squeeze()
        hidden = hidden.index_select(dim=0, index=x_ori_idx)

        return hidden, x
