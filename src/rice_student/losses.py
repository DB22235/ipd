"""
src/rice_student/losses.py
==========================
Knowledge Distillation Loss implementations for the Rice Student Model.
Guarantees mathematical correctness:
  - Strict T^2 temperature scaling to preserve distillation gradients
  - Proper temperature-softened probability distribution generation
  - Linear logits ingestion (from_logits=True)
  - Configurable alpha weighting between supervised CE and KD KL-divergence
"""

from typing import Optional, Dict
import keras
from keras import ops


def compute_distillation_loss(
    y_true: keras.KerasTensor,
    student_logits: keras.KerasTensor,
    teacher_logits: keras.KerasTensor,
    temperature: float = 3.0,
    alpha: float = 0.5,
    sample_weight: Optional[keras.KerasTensor] = None,
) -> Dict[str, keras.KerasTensor]:
    """
    Computes mathematically rigorous Knowledge Distillation loss.
    
    Total Loss = alpha * CE(y_true, student_logits) + (1 - alpha) * T^2 * KL(P_teacher^T, P_student^T)
    
    Returns a dict with:
      - 'loss': Total combined scalar loss
      - 'loss_ce': Supervised cross-entropy loss
      - 'loss_kd': Temperature-scaled KL divergence loss
    """
    # 1. Supervised Cross-Entropy on Hard Labels (T=1)
    loss_ce = keras.losses.sparse_categorical_crossentropy(
        y_true, student_logits, from_logits=True
    )
    if sample_weight is not None:
        if hasattr(sample_weight, "to") and hasattr(loss_ce, "device"):
            sample_weight = sample_weight.to(loss_ce.device)
        loss_ce = loss_ce * sample_weight
    loss_ce = ops.mean(loss_ce)

    # 2. Temperature-Softened Distributions
    soft_teacher = ops.softmax(teacher_logits / temperature, axis=-1)
    soft_student = ops.softmax(student_logits / temperature, axis=-1)

    # 3. Kullback-Leibler Divergence scaled by T^2
    # KL(teacher || student) = sum(teacher * log(teacher / student))
    # Epsilon clip for numerical stability
    eps = 1e-7
    soft_student = ops.clip(soft_student, eps, 1.0 - eps)
    soft_teacher = ops.clip(soft_teacher, eps, 1.0 - eps)
    
    kl_div = ops.sum(
        soft_teacher * (ops.log(soft_teacher) - ops.log(soft_student)),
        axis=-1
    )
    loss_kd = ops.mean(kl_div) * (temperature ** 2)

    # 4. Balanced Total Loss
    total_loss = (alpha * loss_ce) + ((1.0 - alpha) * loss_kd)

    return {
        "loss": total_loss,
        "loss_ce": loss_ce,
        "loss_kd": loss_kd,
    }


class DistillationLoss(keras.losses.Loss):
    """
    Keras Loss wrapper for Distillation loss monitoring.
    """
    def __init__(
        self,
        temperature: float = 3.0,
        alpha: float = 0.5,
        name: str = "distillation_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.temperature = float(temperature)
        self.alpha = float(alpha)

    def call(self, y_true, student_logits, teacher_logits, sample_weight=None):
        res = compute_distillation_loss(
            y_true=y_true,
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            temperature=self.temperature,
            alpha=self.alpha,
            sample_weight=sample_weight,
        )
        return res["loss"]

    def get_config(self):
        config = super().get_config()
        config.update({
            "temperature": self.temperature,
            "alpha": self.alpha,
        })
        return config
