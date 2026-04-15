"""
Combined knowledge distillation loss used by both distillation trainers:

  L = α · CE(student_logits, hard_labels)
    + (1-α) · T² · KL(softmax(student/T) ‖ softmax(teacher/T))

α balances hard-label CE against soft-label KL. T² rescaling keeps gradient
magnitudes comparable between the two terms (Hinton et al. 2015). This is
particularly valuable on confusion-attack benchmarks where near-miss tools
carry more information than in standard classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


DEFAULT_ALPHA: float = 0.5
DEFAULT_TEMPERATURE: float = 2.0


class KDLoss(nn.Module):
    """
    α · CE(student, hard_labels) + (1-α) · T² · KL(student_soft ‖ teacher_soft)

    KL is computed over the full vocabulary at each token position, restricted
    to positions where labels != -100 (the assistant turn only). This matches
    the masking applied during SFT.

    Parameters
    ----------
    alpha : float
        Weight on hard-label CrossEntropy (0–1).
        alpha=1.0 → standard SFT. alpha=0.0 → pure soft-target KD.
    temperature : float
        Softmax temperature applied to both student and teacher logits.
        Higher T → softer distributions → more signal from near-miss tokens.
    """

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        super().__init__()

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}.")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")

        self.alpha = alpha
        self.temperature = temperature

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        student_logits : Tensor  (batch, seq_len, vocab_size)
        teacher_logits : Tensor  (batch, seq_len, vocab_size)  — detached, no grad
        labels : Tensor          (batch, seq_len)  — -100 at prompt positions

        Returns
        -------
        Tensor
            Scalar loss.
        """
        # Shift by 1 for next-token prediction — same as HuggingFace CausalLM
        # internals, replicated here so both CE and KL use the same mask.
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="mean",
        )

        if self.alpha < 1.0:
            T = self.temperature
            active_mask = shift_labels.view(-1) != -100

            if active_mask.sum() == 0:
                return ce_loss

            flat_student = shift_logits.view(-1, shift_logits.size(-1))
            flat_teacher = teacher_logits[..., :-1, :].contiguous().view(
                -1, teacher_logits.size(-1)
            )

            active_student = flat_student[active_mask]
            active_teacher = flat_teacher[active_mask]

            student_log_soft = F.log_softmax(active_student / T, dim=-1)
            teacher_soft     = F.softmax(active_teacher / T, dim=-1)

            kl_loss = F.kl_div(
                student_log_soft,
                teacher_soft,
                reduction="batchmean",
                log_target=False,
            )
            kl_loss = kl_loss * (T ** 2)  # T² rescaling (Hinton et al. 2015)
        else:
            kl_loss = torch.tensor(0.0, device=student_logits.device)

        return self.alpha * ce_loss + (1.0 - self.alpha) * kl_loss

    def __repr__(self) -> str:
        return (
            f"KDLoss(alpha={self.alpha}, temperature={self.temperature})\n"
            f"  L = {self.alpha}·CE + {1.0 - self.alpha}·T²·KL  "
            f"(T={self.temperature})"
        )
