import numpy as np
import torch
import torch.nn as nn
from nengolib.signal import Identity, cont2discrete
from nengolib.synapses import LegendreDelay


class LMUCell(nn.Module):
    """ A layer for LMU-LSTM"""

    def __init__(
        self,
        input_size,
        hidden_size,
        order,
        theta,  # relative to dt=1
        method="euler",
        memory: bool = False,
        gated: bool = False,
    ):
        super().__init__()
        self.method = method
        self.get_solver()

        self.hidden_size = hidden_size
        self.order = order
        self.dt = 1.0
        self.theta = theta

        self.memory = memory
        self.gated = gated

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

        if self.gated:
            self.forget_input = nn.Linear(input_size, order)
            torch.nn.init.ones_(self.forget_input.weight)
            self.forget_hidden = nn.Linear(hidden_size, hidden_size)
            torch.nn.init.ones_(self.forget_hidden.weight)

        # self.linearmem = nn.Linear(order, order)
        # nn.init.xavier_normal_(self.linearmem.weight)

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
            hidden = torch.zeros((batch.size(0), self.hidden_size))
        if memory is None:
            memory = torch.zeros((batch.size(0), self.order))

        ex = self.input_encoder(batch)
        eh = self.hidden_encoder(hidden)
        em = self.memory_encoder(memory)

        ex = self.relu(ex)
        eh = self.relu(eh)
        em = self.relu(em)

        u = ex + eh + em

        self.AT, self.B = self.solver()

        if self.gated:
            fx = self.forget_input(batch)
            fh = self.forget_hidden(hidden)
            f = self.relu(fx) + self.relu(fh)

        if self.memory:
            if self.gated:
                m = torch.mm(memory, self.AT) + torch.mm(f, torch.mm(u, self.BT))
            else:
                m = torch.mm(memory, self.AT) + torch.mm(u, self.B)
            h = (
                self.input_linear(batch)
                + self.hidden_linear(hidden)
                + self.memory_linear(m)
            )

        else:
            h = torch.mm(eh, self.AT) + torch.mm(u, self.B)
            m = None

        h = self.tanh(h)

        return h, m


class LMULSTM(nn.Module):
    """ A layer for LMU-LSTM"""

    def __init__(
        self,
        in_features,
        hidden_size,
        order,
        out_feature,
        theta,  # relative to dt=1
        method="euler",
        memory: bool = False,
        gated: bool = False,
        return_sequences=False,
    ):
        super().__init__()

        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_feature = out_feature
        self.return_sequences = return_sequences

        self.rnn_cell = LMUCell(
            in_features,
            hidden_size,
            order,
            theta,
            method=method,
            memory=memory,
            gated=gated,
        )
        self.fc = nn.Linear(self.hidden_size, self.out_feature)

    def forward(self, x, hidden=None, memory=None):
        batch_size = x.size(0)
        seq_len = x.size(1)

        outputs = []
        last_output = torch.zeros((batch_size, self.out_feature))
        for t in range(seq_len):
            inputs = x[:, t]
            # ts = timespans[:, t].squeeze()
            hidden, memory = self.rnn_cell.forward(inputs, hidden, memory)

            current_output = self.fc(hidden)
            outputs.append(current_output)
            last_output = current_output
        if self.return_sequences:
            outputs = torch.stack(outputs, dim=1)  # return entire sequence
        else:
            outputs = last_output.squeeze()  # only last item
        return outputs
