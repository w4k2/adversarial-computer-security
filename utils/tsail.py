import eagerpy as ep
import foolbox.attacks
from foolbox.attacks.base import get_criterion
from foolbox.attacks.gradient_descent_base import normalize_lp_norms
from foolbox.criteria import Misclassification


class TsAIL(foolbox.attacks.LinfBasicIterativeAttack):
    def __init__(self, *args, mu=0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu

    def run(
        self,
        model,
        inputs,
        criterion,
        *,
        epsilon: float,
        **kwargs,
    ):
        x0, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

        # perform a gradient ascent (targeted attack) or descent (untargeted attack)
        if isinstance(criterion_, Misclassification):
            gradient_step_sign = 1.0
            classes = criterion_.labels
        elif hasattr(criterion_, "target_classes"):
            gradient_step_sign = -1.0
            classes = criterion_.target_classes  # type: ignore
        else:
            raise ValueError("unsupported criterion")

        loss_fn = self.get_loss_fn(model, classes)

        if self.abs_stepsize is None:
            stepsize = self.rel_stepsize * epsilon
        else:
            stepsize = self.abs_stepsize

        if self.random_start:
            x = self.get_random_start(x0, epsilon)
            x = ep.clip(x, *model.bounds)
        else:
            x = x0

        g = ep.zeros_like(x)

        for _ in range(self.steps):
            _, gradients = self.value_and_grad(loss_fn, x)
            gradients = self.normalize(gradients, x=x, bounds=model.bounds)
            # gradients = normalize_lp_norms(gradients, p=1)
            g = self.mu * g + gradients
            x = x + gradient_step_sign * stepsize * g
            x = self.project(x, x0, epsilon)
            x = ep.clip(x, *model.bounds)

        return restore_type(x)
