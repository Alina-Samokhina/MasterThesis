# Code based on the Neural CDE authors repo
# https://github.com/patrick-kidger/NeuralCDE

import torch
import torchcde


class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(CDEFunc, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)

    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)

        z = z.tanh()

        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z


class NeuralCde(torch.nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        output_channels,
        return_sequences=False,
        # solver_type="dopri5",
        interpolation="cubic",
    ):
        super(NeuralCde, self).__init__()

        self.func = CDEFunc(input_channels, hidden_channels)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.interpolation = interpolation
        self.return_sequences = return_sequences

    def forward(self, coeffs):
        if self.interpolation == "cubic":
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == "linear":
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError(
                "Only 'linear' and 'cubic' interpolation methods are implemented."
            )

        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)

        if self.return_sequences:
            pred_y = []
            times = torch.arange(X.interval[0], X.interval[1] + 1)
            z_t = torchcde.cdeint(X=X, z0=z0, func=self.func, t=times)
            for ti in times:
                z_ti = z_t[:, int(ti) - 1]
                pred_y.append(self.readout(z_ti))
            pred_y = torch.stack(pred_y)
        else:
            z_T = torchcde.cdeint(X=X, z0=z0, func=self.func, t=X.interval)
            z_T = z_T[:, 1]
            pred_y = self.readout(z_T)

        return pred_y
