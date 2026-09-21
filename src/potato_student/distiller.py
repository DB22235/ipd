"""
src/potato_student/distiller.py
==============================
Knowledge Distillation trainer for the Potato Student Model.
Features:
  - Frozen EfficientNetB3 potato teacher (trainable=False, training=False)
  - Automatic input resizing if teacher and student dimensions differ
  - Separate logging for student hard-cross-entropy and teacher KL-divergence
"""

from typing import Dict, Any, Tuple
import tensorflow as tf
import keras
from keras import metrics

from .losses import compute_distillation_loss
from .contracts import INPUT_SHAPE_TEACHER


class PotatoDistiller(keras.Model):
    """
    Keras Model wrapper coordinating knowledge distillation from a frozen
    EfficientNetB3 teacher to a MobileNetV3 student.
    """
    def __init__(
        self,
        student: keras.Model,
        teacher: keras.Model,
        alpha: float = 0.5,
        temperature: float = 3.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.student = student
        self.teacher = teacher
        self.alpha = alpha
        self.temperature = temperature

        # Permanently freeze teacher
        self.teacher.trainable = False

        self.loss_tracker = metrics.Mean(name="loss")
        self.student_loss_tracker = metrics.Mean(name="student_loss")
        self.distillation_loss_tracker = metrics.Mean(name="distillation_loss")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.student_loss_tracker,
            self.distillation_loss_tracker,
        ]

    def call(self, inputs, training=False):
        return self.student(inputs, training=training)

    def _prepare_teacher_input(self, x: tf.Tensor) -> tf.Tensor:
        """Resizes input to teacher dimensions (300, 300) if required."""
        target_h, target_w = INPUT_SHAPE_TEACHER[:2]
        h, w = x.shape[1], x.shape[2]
        if h != target_h or w != target_w:
            return tf.image.resize(x, (target_h, target_w))
        return x

    def train_step(self, data):
        if isinstance(data, (tuple, list)):
            x, y = data[0], data[1]
        else:
            x, y = data, None

        # Compute teacher soft targets with gradient tracking disabled
        x_teacher = self._prepare_teacher_input(x)
        teacher_logits = self.teacher(x_teacher, training=False)

        with tf.GradientTape() as tape:
            student_logits = self.student(x, training=True)
            total_loss, student_ce, kl_loss = compute_distillation_loss(
                y_true=y,
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                alpha=self.alpha,
                temperature=self.temperature,
            )

        # Optimize student weights only
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.loss_tracker.update_state(total_loss)
        self.student_loss_tracker.update_state(student_ce)
        self.distillation_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        if isinstance(data, (tuple, list)):
            x, y = data[0], data[1]
        else:
            x, y = data, None

        x_teacher = self._prepare_teacher_input(x)
        teacher_logits = self.teacher(x_teacher, training=False)
        student_logits = self.student(x, training=False)

        total_loss, student_ce, kl_loss = compute_distillation_loss(
            y_true=y,
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            alpha=self.alpha,
            temperature=self.temperature,
        )

        self.loss_tracker.update_state(total_loss)
        self.student_loss_tracker.update_state(student_ce)
        self.distillation_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}
