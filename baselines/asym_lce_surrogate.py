import torch

from helpers.asymmetric_softmax import expert_log_odds
from .basemethod import BaseSurrogateMethod


class AsymLceSurrogate(BaseSurrogateMethod):
    name = "AsymLceSurrogate"

    def surrogate_loss_function(self, outputs, hum_preds, data_y):
        return Asym_CE(output=outputs, label=data_y, expert=hum_preds)


def Asym_CE(output, label, expert):
    label = label.to(output.device).reshape(-1)
    expert_correct = expert.to(output.device).reshape(-1) == label
    class_loss = torch.nn.functional.cross_entropy(output[:, :-1], label)
    log_odds = expert_log_odds(output)
    # Bernoulli negative log likelihood in logit space, including saturated scores.
    expert_loss = torch.nn.functional.softplus(torch.where(expert_correct, -log_odds, log_odds))
    return class_loss + expert_loss.mean()
