"""
src/rice_student/distiller.py
=============================
Knowledge Distillation Engine for Rice Disease Classification.
Orchestrates:
  - Completely frozen EfficientNetB3 teacher (trainable=False, training=False)
  - Zero-gradient teacher inference with stop_gradient
  - Dynamic bilinear resolution alignment (224 -> 300) for synchronized augmentation views
  - Isolated gradient backpropagation strictly through the student
  - Comprehensive metric tracking (total loss, supervised CE, KD loss, accuracy)
"""

from typing import Dict, Any, Tuple
import keras
from keras import ops
import torch

from .losses import compute_distillation_loss


class RiceDistiller(keras.Model):
    """
    Keras 3 Distiller Model coordinating teacher and student training.
    """
    def __init__(
        self,
        student: keras.Model,
        teacher: keras.Model,
        temperature: float = 3.0,
        alpha: float = 0.5,
        teacher_input_size: Tuple[int, int] = (300, 300),
        name: str = "rice_distiller",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.student = student
        self.teacher = teacher
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.teacher_input_size = teacher_input_size

        # CRITICAL SAFEGUARD: Enforce completely frozen teacher
        self.teacher.trainable = False
        assert len(self.teacher.trainable_weights) == 0, (
            "Teacher model has trainable weights! Distillation requires a completely frozen teacher."
        )

        # Metric Trackers
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.ce_loss_tracker = keras.metrics.Mean(name="ce_loss")
        self.kd_loss_tracker = keras.metrics.Mean(name="kd_loss")
        self.acc_metric = keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.ce_loss_tracker,
            self.kd_loss_tracker,
            self.acc_metric,
        ]

    def build(self, input_shape):
        if hasattr(self.student, "built") and not self.student.built:
            self.student.build(input_shape)
        self.built = True

    def call(self, inputs, training=False):
        """Standard call routes directly to the student model."""
        return self.student(inputs, training=training)

    def train_step(self, data):
        """Custom training step computing both hard CE and teacher soft targets."""
        # Unpack input data
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        if sample_weight is not None and hasattr(sample_weight, "to") and hasattr(x, "device"):
            sample_weight = sample_weight.to(x.device)

        # 0. Zero leftover gradients
        self.zero_grad()

        # 1. Forward pass through Frozen Teacher with Gradient Stop & torch.no_grad()
        # Dynamically match teacher input resolution (224 -> 300) so both see the exact same augmented image
        x_teacher = ops.image.resize(x, self.teacher_input_size, interpolation="bilinear")
        with torch.no_grad():
            teacher_logits = self.teacher(x_teacher, training=False)

        # 2. Forward pass through Student with gradient tracking
        student_logits = self.student(x, training=True)
        loss_dict = compute_distillation_loss(
            y_true=y,
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            temperature=self.temperature,
            alpha=self.alpha,
            sample_weight=sample_weight,
        )
        total_loss = loss_dict["loss"]

        # 3. Scale loss if optimizer scales, then backward
        if self.optimizer is not None:
            scaled_loss = self.optimizer.scale_loss(total_loss)
        else:
            scaled_loss = total_loss

        scaled_loss.backward()

        # 4. Apply Gradients to Student Weights ONLY
        trainable_weights = self.student.trainable_weights[:]
        gradients = [v.value.grad for v in trainable_weights]
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        # 5. Update Metric Trackers
        self.loss_tracker.update_state(total_loss)
        self.ce_loss_tracker.update_state(loss_dict["loss_ce"])
        self.kd_loss_tracker.update_state(loss_dict["loss_kd"])
        self.acc_metric.update_state(y, student_logits)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """Evaluation step on validation/test data."""
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        if sample_weight is not None and hasattr(sample_weight, "to") and hasattr(x, "device"):
            sample_weight = sample_weight.to(x.device)

        with torch.no_grad():
            # Forward pass on teacher
            x_teacher = ops.image.resize(x, self.teacher_input_size, interpolation="bilinear")
            teacher_logits = self.teacher(x_teacher, training=False)

            # Forward pass on student
            student_logits = self.student(x, training=False)

            loss_dict = compute_distillation_loss(
                y_true=y,
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                temperature=self.temperature,
                alpha=self.alpha,
                sample_weight=sample_weight,
            )

        self.loss_tracker.update_state(loss_dict["loss"])
        self.ce_loss_tracker.update_state(loss_dict["loss_ce"])
        self.kd_loss_tracker.update_state(loss_dict["loss_kd"])
        self.acc_metric.update_state(y, student_logits)

        return {m.name: m.result() for m in self.metrics}
