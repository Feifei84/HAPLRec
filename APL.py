import torch
from torch.optim.optimizer import Optimizer


class APL(Optimizer):
    def __init__(self, params):
        defaults = dict()
        super(APL, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, loss_array, ranks):
        self.balance_GradMagnitudes(loss_array, ranks)

    @staticmethod
    def _flattening(grad):
        return torch.cat(tuple(g.reshape(-1, ) for i, g in enumerate(grad)), axis=0)

    def balance_GradMagnitudes(self, loss_array, ranks):

        for loss_index, loss in enumerate(loss_array):
            if loss_index + 1 == len(loss_array):
                loss.backward()
            else:
                loss.backward(retain_graph=True)
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        print("breaking")
                        break

                    state = self.state[p]
                    if len(state) == 0:
                        for j, _ in enumerate(loss_array):
                            if j == 0: p.norms = [torch.zeros(1).cuda()]
                            else: p.norms.append(torch.zeros(1).cuda())

                    p.norms[loss_index] = torch.norm(p.grad)

                    if loss_index == 0:
                        state['grad_0'] = torch.zeros_like(p.data)
                        state['grad_0'] += p.grad
                    else:
                        if p.norms[loss_index] > 1e-10:
                            cosine = torch.sum(torch.mul(p.grad, state['grad_0']))
                            cosine = cosine/torch.norm(p.grad)/torch.norm(state['grad_0'])
                            if torch.clamp(cosine, -1, 1) <= 0:
                                p.grad = torch.zeros_like(p.data)
                            else:
                                p.grad = ranks[loss_index] * p.norms[0] * p.grad / p.norms[loss_index]

                    if loss_index == 0:
                        state['sum_gradient'] = torch.zeros_like(p.data)
                        state['sum_gradient'] += p.grad
                    else:
                        state['sum_gradient'] += p.grad

                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

                    if loss_index == len(loss_array) - 1:
                        p.grad = state['sum_gradient']
                else:
                    continue