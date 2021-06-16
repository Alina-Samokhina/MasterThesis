import numpy as np
import torch
import torch.nn as nn
from nengolib.signal import Identity, cont2discrete
from nengolib.synapses import LegendreDelay


class LMUCell(nn.Module):
    """ A simple layer for LMU-LSTM"""

    def __init__(
        self, input_size, hidden_size, order, theta, method="euler",
    ):
        super().__init__()
        self.method = method
        self.get_solver()

        self.hidden_size = hidden_size
        self.order = order
        self.dt = 1.0
        self.theta = theta

        realizer = Identity()
        self._realizer_result = realizer(LegendreDelay(theta=theta, order=order))

        self._ss = cont2discrete(
            self._realizer_result.realization, dt=1.0, method=method
        )

        self._A = self._ss.A
        self.AT = self._A.T
        self._B = self._ss.B
        self.B = self._B
        self._C = self._ss.C
        assert np.allclose(self._ss.D, 0)  # proper LTI

        self.input_encoder = nn.Linear(input_size, hidden_size)
        nn.init.kaiming_uniform_(self.input_encoder.weight, mode="fan_in")
        self.hidden_encoder = nn.Linear(hidden_size, hidden_size)
        nn.init.kaiming_uniform_(self.hidden_encoder.weight, mode="fan_in")
        self.memory_encoder = nn.Linear(order, hidden_size)
        torch.nn.init.zeros_(self.memory_encoder.weight)

        self.input_linear = nn.Linear(input_size, hidden_size)
        nn.init.xavier_normal_(self.input_linear.weight)
        self.hidden_linear = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_normal_(self.hidden_linear.weight)
        self.memory_linear = nn.Linear(order, hidden_size)
        nn.init.xavier_normal_(self.memory_linear.weight)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def get_solver(self):
        if self.method == "euler":
            self.solver = self._euler

    def _euler(self):
        self.ID = torch.eye(self.order)
        AT = self.ID + self.dt * self.AT  # * self.theta
        B = self.dt * self.B  # * self.theta
        return (AT.to(dtype=torch.float32), torch.tensor(B, dtype=torch.float32))

    def forward(self, batch, hidden=None, memory=None):
        if hidden is None:
            hidden = torch.zeros((batch.size(0), self.hidden_size, self.order))
        if memory is None:
            memory = torch.zeros((batch.size(0), self.order))

        ex = self.input_encoder(batch)
        u = ex
        h = hidden.view(-1, self.hidden_size, self.order)
        AT, B = self.solver()

        h = torch.einsum("ijk,kk->ikj", h, AT) + B * u[:, None]

        h = self.relu(h.reshape(batch.size(0), self.hidden_size * self.order))

        return h, None


class LmuLstm(nn.Module):
    """LMU LSTM Neural Network.

    Based on the paper
    https://papers.nips.cc/paper/2019/file/952285b9b7e7a1be5aa7849f32ffff05-Paper.pdf
    """

    def __init__(
        self,
        in_features,
        hidden_size,
        order,
        out_feature,
        theta,  # relative to dt=1
        method="euler",
        return_sequences=False,
    ):
        """
        Args:
            in_features: number of channels
            hidden_size: hidden dimension
            order: order of Legendre polynomials to use in the approximation
            out_feature: number of instances to predict (num of classes)
            theta: window size
            method: method for solving ODE
            return_sequences: whether to get predictions for each input (if True),
                or only one predictions per sequens(if False)
        """
        super().__init__()

        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_feature = out_feature
        self.return_sequences = return_sequences

        self.rnn_cell = LMUCell(in_features, hidden_size, order, theta, method=method,)
        self.fc = nn.Linear(hidden_size * order, self.out_feature)

    def forward(self, x, hidden=None, memory=None):
        batch_size = x.size(0)
        seq_len = x.size(1)

        outputs = []
        last_output = torch.zeros((batch_size, self.out_feature))
        for t in range(seq_len):
            inputs = x[:, t]
            hidden, memory = self.rnn_cell.forward(inputs, hidden, memory)

            current_output = self.fc(hidden)
            outputs.append(current_output)
            last_output = current_output
        if self.return_sequences:
            outputs = torch.stack(outputs, dim=1)
        else:
            outputs = last_output.squeeze()
        return outputs
