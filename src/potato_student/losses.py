"""
src/potato_student/losses.py
===========================
Loss functions for the Potato Student Model.
Implements:
  - Supervised Sparse Categorical Cross-Entropy (from_logits=True)
  - Temperature-scaled Knowledge Distillation Loss combining ground-truth
    cross-entropy and teacher-student Kullback-Leibler (KL) divergence.
"""

from typing import Tuple
import tensorflow as tf
import keras
from keras import losses, ops


def compute_distillation_loss(
    y_true: tf.Tensor,
    student_logits: tf.Tensor,
    teacher_logits: tf.Tensor,
    alpha: float = 0.5,
    temperature: float = 3.0,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Computes combined knowledge distillation loss:
      Loss = alpha * CE(y_true, student_logits) + (1 - alpha) * (T^2) * KL(soft_teacher, soft_student)

    Returns:
      total_loss, student_ce_loss, kl_distillation_loss
    """
    # 1. Supervised Student Hard Cross-Entropy
    student_ce_loss = losses.sparse_categorical_crossentropy(
        y_true, student_logits, from_logits=True
    )
    student_ce_loss = tf.reduce_mean(student_ce_loss)

    # 2. Temperature-scaled Soft Probabilities
    soft_teacher = tf.nn.softmax(teacher_logits / temperature, axis=-1)
    soft_student = tf.nn.softmax(student_logits / temperature, axis=-1)

    # 3. KL Divergence: KL(teacher || student) = teacher * (log(teacher) - log(student))
    eps = 1e-7
    soft_teacher = tf.clip_by_value(soft_teacher, eps, 1.0)
    soft_student = tf.clip_by_value(soft_student, eps, 1.0)

    kl_loss = tf.reduce_sum(
        soft_teacher * (tf.math.log(soft_teacher) - tf.math.log(soft_student)),
        axis=-1,
    )
    kl_loss = tf.reduce_mean(kl_loss) * (temperature ** 2)

    # 4. Combined Loss
    total_loss = alpha * student_ce_loss + (1.0 - alpha) * kl_loss

    return total_loss, student_ce_loss, kl_loss
