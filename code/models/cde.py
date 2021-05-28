import torch
import torchcde


class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(CDEFunc, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)

    # t argument can be ignored unless you want your CDE to behave
    # differently at different times, which would be unusual
    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)

        z = z.tanh()

        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z


class NeuralCDE(torch.nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        output_channels,
        # return_sequences=True,
        # solver_type="dopri5",
        interpolation="cubic",
    ):
        super(NeuralCDE, self).__init__()

        self.func = CDEFunc(input_channels, hidden_channels)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.interpolation = interpolation

    def forward(self, coeffs):
        if self.interpolation == "cubic":
            X = torchcde.NaturalCubicSpline(coeffs)
        elif self.interpolation == "linear":
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError(
                "Only 'linear' and 'cubic' interpolation methods are implemented."
            )

        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)

        z_T = torchcde.cdeint(X=X, z0=z0, func=self.func, t=X.interval)

        z_T = z_T[:, 1]
        pred_y = self.readout(z_T)
        return pred_y
