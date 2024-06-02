from transformers import Trainer
from typing import Any, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from experiments.models.sparse_silu.utils import (
    get_mlp_class,
    get_decoder_class,
)


class SparseTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.regularization_coefficient = kwargs.pop(
            "regularization_coefficient", 10
        )
        self.use_sparse_regularization = kwargs.pop(
            "use_sparse_regularization", False
        )
        self.use_spm_loss = False
        self.freeze_original_weights = False
        self.regularization_type = kwargs.pop(
            "regularization_type", "L1 positive activation"
        )
        assert self.regularization_type in [
            "L2 activation",
            "L1 positive activation",
        ], f"Invalid regularization type: {self.regularization_type}"
        self.sparse_layers = []
        self.sparse_decoder_layers = []
        super(SparseTrainer, self).__init__(*args, **kwargs)

    def initialize_sparse_silu_layers(self, model):
        SparseMLP = get_mlp_class(model)
        self.sparse_layers = [
            m for m in model.modules() if isinstance(m, SparseMLP)
        ]

    def initialize_sparse_decoder_layers(self, model):
        SparseDecoder = get_decoder_class(model)
        self.sparse_decoder_layers = [
            m for m in model.modules() if isinstance(m, SparseDecoder)
        ]

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Override the huggingface's training_step function to add a regularization term.
        A regularization term is computed with intermediate values, which are freed after "backward()."
        You need to set `retain_graph=True` inside `backward` function to keep the values.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = (
                loss.mean()
            )  # mean() to average on multi-gpu parallel training

        if not self.freeze_original_weights:
            if loss is not None:
                self.accelerator.backward(loss, retain_graph=True)

        if self.use_sparse_regularization:
            regularization_loss = self.compute_regularization(model)
            if self.args.n_gpu > 1:
                regularization_loss = regularization_loss.mean()
            if regularization_loss is not None:
                self.accelerator.backward(
                    regularization_loss, retain_graph=True
                )
            loss += regularization_loss

        if self.use_spm_loss:
            spm_loss = self.compute_spm_loss(model)
            if self.args.n_gpu > 1:
                spm_loss = spm_loss.mean()
            if spm_loss is not None:
                self.accelerator.backward(spm_loss, retain_graph=False)
            loss += spm_loss

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_regularization(self, model):
        """
        Compute a sparse regularization loss for SiLU
        """
        loss = 0
        if len(self.sparse_layers) == 0:
            self.initialize_sparse_silu_layers(model)
        num_layers = len(self.sparse_layers)

        for module in self.sparse_layers:
            if module.activation_norm is not None:
                loss += module.activation_norm

        loss /= num_layers
        loss *= self.regularization_coefficient

        if self.state.global_step % 20 == 0 and loss != 0:
            print("Negative relularizer loss: ", loss.item())
        return loss

    def compute_spm_loss(self, model):
        loss = 0
        if len(self.sparse_decoder_layers) == 0:
            self.initialize_sparse_decoder_layers(model)
        for module in self.sparse_decoder_layers:
            if module.distill_loss != None:
                loss += module.distill_loss
        if self.state.global_step % 20 == 0 and loss != 0:
            print("Sparse Predictor Distillation loss: ", loss.item())
        return loss


class SparseSiLUTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(MARETrainer, self).__init__(*args, **kwargs)
        self.steps = 0
        self.relu_loss = torch.nn.MSELoss()

    def compute_penalty(self, model):
        loss = None
        self.steps += 1
        count = 0

        for is_decoder, network in enumerate(
            [
                model.transformer.encoder,
                model.transformer.decoder,
            ]
        ):
            for block_idx, block in enumerate(network.block):
                if is_decoder:
                    mlp_layer = block.layer[2].DenseReluDense
                    # continue
                else:
                    # if block_idx < 6:
                    #     continue
                    mlp_layer = block.layer[1].DenseReluDense

                if mlp_layer.__class__.__name__ != "MLPSparsityCheck":
                    continue

                count += 1

                if (
                    mlp_layer.routing_probs is None
                    or mlp_layer.activation is None
                ):
                    continue

                layer_loss = self.router_loss(
                    mlp_layer.routing_probs,
                    mlp_layer.activation,
                )

                if loss is None:
                    loss = layer_loss
                else:
                    loss += layer_loss

        if loss is not None:
            loss = loss / count

        if self.steps % 50 == 0:
            print(loss)

        return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")[:, 0]
        labels = labels.to(logits.dtype)

        # # get penalty term
        # lambda_coeff = 1
        # regularizer = lambda_coeff * self.compute_penalty(model)
        # loss = F.binary_cross_entropy(logits, labels)
        alpha = 1.0
        # penalty = self.compute_penalty(model)
        penalty = None
        loss = F.binary_cross_entropy(logits, labels)

        if penalty:
            loss = loss + alpha * penalty

        return (loss, outputs) if return_outputs else loss


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        # forward pass
        outputs = model(**inputs)

        logits = outputs.get("logits")[:, 0]
        labels = labels.to(logits.dtype)

        # Check if there are any NaN values in logits
        if torch.isnan(logits).any():
            # If there are NaN values, return a zero loss for this batch
            # This effectively "skips" optimizing this batch.
            # The zero gradients won't affect the optimizer update.
            print("NAN")
            loss = torch.tensor(0.0).to(logits.device).requires_grad_()
        else:
            loss = F.binary_cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss


class MARETrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(MARETrainer, self).__init__(*args, **kwargs)
        self.steps = 0
        self.router_loss = torch.nn.MSELoss()

    def compute_penalty(self, model):
        loss = None
        self.steps += 1
        count = 0

        for is_decoder, network in enumerate(
            [
                model.transformer.encoder,
                model.transformer.decoder,
            ]
        ):
            for block_idx, block in enumerate(network.block):
                if is_decoder:
                    mlp_layer = block.layer[2].DenseReluDense
                    # continue
                else:
                    # if block_idx < 6:
                    #     continue
                    mlp_layer = block.layer[1].DenseReluDense

                if mlp_layer.__class__.__name__ != "MLPSparsityCheck":
                    continue

                count += 1

                if (
                    mlp_layer.routing_probs is None
                    or mlp_layer.activation is None
                ):
                    continue

                layer_loss = self.router_loss(
                    mlp_layer.routing_probs,
                    mlp_layer.activation,
                )

                if loss is None:
                    loss = layer_loss
                else:
                    loss += layer_loss

        if loss is not None:
            loss = loss / count

        if self.steps % 50 == 0:
            print(loss)

        return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")[:, 0]
        labels = labels.to(logits.dtype)

        # # get penalty term
        # lambda_coeff = 1
        # regularizer = lambda_coeff * self.compute_penalty(model)
        # loss = F.binary_cross_entropy(logits, labels)
        alpha = 1.0
        # penalty = self.compute_penalty(model)
        penalty = None
        loss = F.binary_cross_entropy(logits, labels)

        if penalty:
            loss = loss + alpha * penalty

        return (loss, outputs) if return_outputs else loss
