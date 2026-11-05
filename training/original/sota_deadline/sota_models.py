#!/usr/bin/env python3
"""
Deadline-oriented modernized models.

Task 1:
    CT -> Swin visual tokens
       -> 18 learned label tokens cross-attend to image
       -> label-token self-attention (labels can interact)
       -> optional empirical label-graph mixer
       -> 18 logits

Task 2:
    CT -> Swin visual tokens
       -> Perceiver-style resampler (fixed visual token budget)
       -> projector into a small decoder-only instruction LM
       -> LoRA on LM attention
       -> report text
       + auxiliary 18-label pathology head

The Task-2 model deliberately uses a small causal LM (default Qwen2.5-0.5B)
rather than a 7B model because the deadline makes visual grounding/token
compression more important than blindly scaling the decoder.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import model_code_from_original as mc


class InteractingLabelTokenClassifier(nn.Module):
    def __init__(
        self,
        vision_params,
        num_labels=18,
        token_grid=(2, 4, 4),
        d_model=384,
        nhead=8,
        label_layers=2,
        dropout=0.1,
        label_graph: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_labels = int(num_labels)
        self.vision = mc.SwinUNETRVisualTokenizer(
            model_params=vision_params,
            token_grid=token_grid,
        )
        self.visual_proj = nn.LazyLinear(d_model)

        self.label_tokens = nn.Parameter(torch.randn(num_labels, d_model) * 0.02)
        self.cross_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.label_interaction = nn.TransformerEncoder(
            enc_layer, num_layers=label_layers
        )

        self.out_norm = nn.LayerNorm(d_model)
        self.label_head = nn.Linear(d_model, 1)

        if label_graph is not None:
            A = torch.as_tensor(label_graph, dtype=torch.float32)
            if A.shape != (num_labels, num_labels):
                raise ValueError(f"label_graph must be {num_labels}x{num_labels}")
            A = A.clone()
            A.fill_diagonal_(0)
            denom = A.abs().sum(dim=1, keepdim=True).clamp_min(1.0)
            A = A / denom
            self.register_buffer("label_graph", A)
            self.graph_alpha = nn.Parameter(torch.tensor(0.0))
        else:
            self.label_graph = None
            self.graph_alpha = None

    def forward(self, x, return_label_embeddings=False):
        visual = self.visual_proj(self.vision(x))
        B = x.shape[0]
        q = self.label_tokens.unsqueeze(0).expand(B, -1, -1)

        attn_out, _ = self.cross_attn(
            query=self.cross_norm(q),
            key=visual,
            value=visual,
            need_weights=False,
        )
        h = q + attn_out
        h = self.label_interaction(h)

        if self.label_graph is not None:
            graph_msg = torch.einsum("ij,bjd->bid", self.label_graph, h)
            h = h + torch.tanh(self.graph_alpha) * graph_msg

        logits = self.label_head(self.out_norm(h)).squeeze(-1)
        if return_label_embeddings:
            return logits, h
        return logits


class PerceiverResamplerLayer(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.self_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, latents, visual):
        z, _ = self.cross(
            self.q_norm(latents),
            self.kv_norm(visual),
            self.kv_norm(visual),
            need_weights=False,
        )
        latents = latents + z
        z, _ = self.self_attn(
            self.self_norm(latents),
            self.self_norm(latents),
            self.self_norm(latents),
            need_weights=False,
        )
        latents = latents + z
        latents = latents + self.ff(self.ff_norm(latents))
        return latents


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        num_latents=32,
        nhead=8,
        depth=2,
        dropout=0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.latents = nn.Parameter(torch.randn(num_latents, d_model) * 0.02)
        self.layers = nn.ModuleList([
            PerceiverResamplerLayer(d_model, nhead=nhead, dropout=dropout)
            for _ in range(depth)
        ])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, visual):
        visual = self.input_proj(visual)
        B = visual.shape[0]
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            latents = layer(latents, visual)
        return self.out_norm(latents)


class CTCausalReportGenerator(nn.Module):
    def __init__(
        self,
        vision_params,
        lm_name="Qwen/Qwen2.5-0.5B-Instruct",
        token_grid=(4, 6, 6),
        num_visual_tokens=32,
        resampler_depth=2,
        resampler_heads=8,
        num_labels=18,
        aux_pathology_weight=0.25,
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        freeze_vision=True,
        gradient_checkpointing=True,
    ):
        super().__init__()
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.lm_name = lm_name
        self.aux_pathology_weight = float(aux_pathology_weight)
        self.token_grid = tuple(token_grid)

        self.tokenizer = AutoTokenizer.from_pretrained(lm_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        self.lm = AutoModelForCausalLM.from_pretrained(
            lm_name,
            torch_dtype=dtype,
        )
        self.lm.config.use_cache = False

        if use_lora:
            try:
                from peft import LoraConfig, TaskType, get_peft_model
            except ImportError as e:
                raise ImportError(
                    "PEFT is required for LoRA. Install it without replacing Torch, "
                    "e.g. `python -m pip install --no-deps peft==0.13.2`."
                ) from e

            cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            self.lm = get_peft_model(self.lm, cfg)

        if gradient_checkpointing:
            if hasattr(self.lm, "gradient_checkpointing_enable"):
                self.lm.gradient_checkpointing_enable()
            if hasattr(self.lm, "enable_input_require_grads"):
                self.lm.enable_input_require_grads()

        hidden = getattr(self.lm.config, "hidden_size", None)
        if hidden is None:
            hidden = getattr(self.lm.config, "n_embd", None)
        if hidden is None:
            raise ValueError("Could not determine causal-LM hidden size.")

        self.vision = mc.SwinUNETRVisualTokenizer(
            model_params=vision_params,
            token_grid=token_grid,
        )

        # The Swin output channel dimension is inferred lazily so the code remains
        # compatible with the existing backbone configuration.
        self.visual_in = nn.LazyLinear(hidden)
        self.resampler = PerceiverResampler(
            input_dim=hidden,
            d_model=hidden,
            num_latents=num_visual_tokens,
            nhead=resampler_heads,
            depth=resampler_depth,
        )
        self.pathology_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_labels),
        )

        if freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad = False

        self.prefix_text = (
            "You are a radiologist. Generate a concise chest CT radiology report "
            "with FINDINGS and IMPRESSION. Base the report on the CT evidence.\n"
        )

    def encode_visual(self, x):
        v = self.vision(x)
        v = self.visual_in(v)
        v = self.resampler(v)
        pathology_logits = self.pathology_head(v.mean(dim=1))
        return v, pathology_logits

    def _embed_tokens(self, input_ids):
        return self.lm.get_input_embeddings()(input_ids)

    def forward(
        self,
        x,
        report_texts: Sequence[str],
        pathology_targets: Optional[torch.Tensor] = None,
        max_target_length=256,
    ):
        device = x.device
        B = x.shape[0]
        visual, pathology_logits = self.encode_visual(x)

        prefix = self.tokenizer(
            [self.prefix_text] * B,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        target = self.tokenizer(
            list(report_texts),
            padding=True,
            truncation=True,
            max_length=max_target_length,
            add_special_tokens=True,
            return_tensors="pt",
        )

        prefix_ids = prefix["input_ids"].to(device)
        prefix_mask = prefix["attention_mask"].to(device)
        target_ids = target["input_ids"].to(device)
        target_mask = target["attention_mask"].to(device)

        prefix_emb = self._embed_tokens(prefix_ids)
        target_emb = self._embed_tokens(target_ids)

        # Keep embeddings in the LM dtype.
        dtype = prefix_emb.dtype
        visual = visual.to(dtype)
        target_emb = target_emb.to(dtype)

        inputs_embeds = torch.cat([prefix_emb, visual, target_emb], dim=1)

        visual_mask = torch.ones(
            B, visual.shape[1], dtype=prefix_mask.dtype, device=device
        )
        attention_mask = torch.cat(
            [prefix_mask, visual_mask, target_mask], dim=1
        )

        ignore_prefix = torch.full(
            (B, prefix_ids.shape[1] + visual.shape[1]),
            -100,
            dtype=torch.long,
            device=device,
        )
        target_labels = target_ids.clone()
        target_labels[target_mask == 0] = -100
        labels = torch.cat([ignore_prefix, target_labels], dim=1)

        out = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        text_loss = out.loss
        pathology_loss = torch.tensor(0.0, device=device)

        if pathology_targets is not None:
            pathology_loss = F.binary_cross_entropy_with_logits(
                pathology_logits.float(),
                pathology_targets.float(),
            )

        total_loss = text_loss + self.aux_pathology_weight * pathology_loss
        return {
            "loss": total_loss,
            "text_loss": text_loss,
            "pathology_loss": pathology_loss,
            "pathology_logits": pathology_logits,
        }

    @torch.no_grad()
    def generate(
        self,
        x,
        max_new_tokens=256,
        num_beams=2,
        num_return_sequences=1,
    ):
        self.eval()
        device = x.device
        B = x.shape[0]
        visual, pathology_logits = self.encode_visual(x)

        prefix = self.tokenizer(
            [self.prefix_text] * B,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        prefix_ids = prefix["input_ids"].to(device)
        prefix_mask = prefix["attention_mask"].to(device)
        prefix_emb = self._embed_tokens(prefix_ids)
        visual = visual.to(prefix_emb.dtype)

        inputs_embeds = torch.cat([prefix_emb, visual], dim=1)
        visual_mask = torch.ones(
            B, visual.shape[1], dtype=prefix_mask.dtype, device=device
        )
        attention_mask = torch.cat([prefix_mask, visual_mask], dim=1)

        generated = self.lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=max(num_beams, num_return_sequences),
            num_return_sequences=num_return_sequences,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        texts = self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )
        return texts, torch.sigmoid(pathology_logits.float())

    def load_task1_encoder(self, task1_checkpoint_path):
        """
        Transfer the current Task-1 Swin encoder into Task 2 when compatible.
        Works with checkpoints whose state_dict uses the existing `encoder.*`
        naming convention.
        """
        ckpt = torch.load(task1_checkpoint_path, map_location="cpu")
        state = ckpt.get("model", ckpt)
        enc = {
            k[len("encoder."):]: v
            for k, v in state.items()
            if k.startswith("encoder.")
        }
        if not enc:
            raise ValueError("No `encoder.*` keys found in Task-1 checkpoint.")
        missing, unexpected = self.vision.encoder.load_state_dict(enc, strict=False)
        print("Transferred Task-1 encoder.")
        print("missing:", len(missing), "unexpected:", len(unexpected))
        return missing, unexpected
