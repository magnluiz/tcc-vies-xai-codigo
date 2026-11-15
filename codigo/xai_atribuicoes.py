"""Métodos de atribuição usados nos experimentos: Grad-CAM e Integrated Gradients."""

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM na saída da layer4 de uma ResNet (Selvaraju et al., 2017)."""

    def __init__(self, model: torch.nn.Module):
        self.model = model.eval()
        self.activations = None
        self.gradients = None
        self.handle = model.layer4.register_forward_hook(self._save_activation)

    def _save_activation(self, _module, _inputs, output):
        self.activations = output
        output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient):
        self.gradients = gradient

    def remove(self) -> None:
        self.handle.remove()

    def __call__(self, inputs: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        """Mapas normalizados em [0, 1], no tamanho da entrada."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(inputs)
        logits.gather(1, classes.view(-1, 1)).sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
        peak = cam.flatten(1).amax(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
        return (cam / peak).squeeze(1).detach()


def integrated_gradients(model: torch.nn.Module, inputs: torch.Tensor, classes: torch.Tensor,
                         baseline: torch.Tensor, steps: int = 32) -> torch.Tensor:
    """Integrated Gradients (Sundararajan et al., 2017), regra do ponto médio.

    Retorna, por imagem, a soma dos valores absolutos das atribuições nos três canais.
    `inputs` e `baseline` já devem estar normalizados como a entrada do modelo.
    """
    model.eval()
    alphas = (torch.arange(steps, device=inputs.device, dtype=inputs.dtype) + 0.5) / steps
    maps = []
    for image, target in zip(inputs, classes):
        path = baseline + alphas.view(-1, 1, 1, 1) * (image - baseline)
        path.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = model(path)
        gradients, = torch.autograd.grad(logits[:, target].sum(), path)
        attribution = (image - baseline) * gradients.mean(dim=0)
        maps.append(attribution.abs().sum(dim=0).detach())
    return torch.stack(maps)


def mass_fraction(maps: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Fração da atribuição total que cai dentro da máscara, por imagem."""
    total = maps.reshape(len(maps), -1).sum(axis=1)
    inside = (maps * masks).reshape(len(maps), -1).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, inside / total, np.nan)
