from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from lean_dojo_v2.database import DynamicDatabase
from lean_dojo_v2.diffusion_training import DiffusionTrainingConfig
from lean_dojo_v2.diffusion_training.corruption import (
    corrupt_tactic_script,
    mask_target_tokens,
)
from lean_dojo_v2.diffusion_training.formatting import (
    format_infill_prompt,
    format_next_tactic_prompt,
    normalize_tactic_target,
)
from lean_dojo_v2.diffusion_training.objectives import masked_denoising_loss
from lean_dojo_v2.lean_dojo.data_extraction.lean import LeanGitRepo


class DiffusionTacticDataset(Dataset):
    """Dataset for objective O1: GoalState -> single-line tactic denoising."""

    def __init__(self, data_path: str):
        with open(data_path) as f:
            raw = json.load(f)
        self.data = self._process_data(raw)

    def _process_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        processed: List[Dict[str, str]] = []
        for item in data:
            for tactic in item.get("traced_tactics", []):
                target = normalize_tactic_target(tactic.get("tactic", ""))
                if not target or target in {"sorry", "admit"}:
                    continue
                processed.append(
                    {
                        "prompt": format_next_tactic_prompt(
                            tactic.get("state_before", "")
                        ),
                        "target": target,
                    }
                )
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


class DiffusionInfillDataset(Dataset):
    """Dataset for objective O2: tactic-script span infilling."""

    def __init__(
        self,
        data_path: str,
        seed: int = 42,
        num_holes: int = 1,
        max_hole_len: int = 3,
        hole_token_template: str = "<HOLE_{i}>",
    ):
        with open(data_path) as f:
            raw = json.load(f)
        self.rng = random.Random(seed)
        self.num_holes = num_holes
        self.max_hole_len = max_hole_len
        self.hole_token_template = hole_token_template
        self.data = self._process_data(raw)

    def _process_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        processed: List[Dict[str, str]] = []
        for item in data:
            tactics = [
                normalize_tactic_target(t.get("tactic", ""))
                for t in item.get("traced_tactics", [])
            ]
            tactics = [t for t in tactics if t and t not in {"sorry", "admit"}]
            if len(tactics) < 2:
                continue

            corrupted, targets = corrupt_tactic_script(
                tactics=tactics,
                num_holes=self.num_holes,
                max_hole_len=self.max_hole_len,
                hole_token_template=self.hole_token_template,
                rng=self.rng,
            )
            prompt = format_infill_prompt(
                corrupted_tactics=corrupted,
                theorem_statement=item.get("theorem_statement"),
            )
            target = "\n".join(
                f"{entry['hole_id']}: {' ; '.join(entry['original_span'])}"
                for entry in targets
            )
            processed.append({"prompt": prompt, "target": target})
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


class DiffusionTrainer:
    """Custom discrete denoising trainer for diffusion objectives."""

    def __init__(
        self,
        model_name: str,
        output_dir: str = "outputs-diffusion",
        objective: str = "next_tactic",
        epochs_per_repo: int = 1,
        batch_size: int = 1,
        lr: float = 2e-5,
        gradient_accumulation_steps: int = 1,
        max_length: int = 768,
        mask_prob: float = 0.3,
        seed: int = 42,
        num_holes: int = 1,
        max_hole_len: int = 3,
    ):
        self.config = DiffusionTrainingConfig(
            model_name=model_name,
            output_dir=output_dir,
            objective=objective,  # type: ignore[arg-type]
            epochs_per_repo=epochs_per_repo,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            lr=lr,
            max_length=max_length,
            mask_prob=mask_prob,
            seed=seed,
            num_holes=num_holes,
            max_hole_len=max_hole_len,
        )
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.tokenizer.mask_token_id is None:
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<MASK>"]})
            self.tokenizer.mask_token = "<MASK>"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def _build_dataset(self, train_json_path: str) -> Dataset:
        if self.config.objective == "next_tactic":
            return DiffusionTacticDataset(train_json_path)
        if self.config.objective == "infill":
            return DiffusionInfillDataset(
                train_json_path,
                seed=self.config.seed,
                num_holes=self.config.num_holes,
                max_hole_len=self.config.max_hole_len,
                hole_token_template=self.config.hole_token_template,
            )
        raise ValueError(f"Unsupported objective: {self.config.objective}")

    def _collate_batch(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        prompt_ids = self.tokenizer(
            [b["prompt"] for b in batch],
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            padding=True,
        )
        target_ids = self.tokenizer(
            [b["target"] for b in batch],
            return_tensors="pt",
            truncation=True,
            max_length=max(16, self.config.max_length // 3),
            padding=True,
            add_special_tokens=False,
        )

        input_ids_list: List[torch.Tensor] = []
        target_masks: List[torch.Tensor] = []
        attention_masks: List[torch.Tensor] = []
        max_len = 0

        for i in range(prompt_ids.input_ids.size(0)):
            prompt = prompt_ids.input_ids[i]
            target = target_ids.input_ids[i]
            target = target[target != self.tokenizer.pad_token_id]

            joined = torch.cat([prompt, target], dim=0)[: self.config.max_length]
            target_mask = torch.zeros_like(joined, dtype=torch.bool)
            target_start = min(prompt.size(0), joined.size(0))
            if target_start < joined.size(0):
                target_mask[target_start:] = True

            attn = torch.ones_like(joined)
            input_ids_list.append(joined)
            target_masks.append(target_mask)
            attention_masks.append(attn)
            max_len = max(max_len, joined.size(0))

        pad_id = self.tokenizer.pad_token_id
        padded_inputs = []
        padded_target_masks = []
        padded_attn = []
        for ids, mask, attn in zip(input_ids_list, target_masks, attention_masks):
            pad_len = max_len - ids.size(0)
            if pad_len > 0:
                ids = torch.cat(
                    [ids, torch.full((pad_len,), pad_id, dtype=torch.long)], dim=0
                )
                mask = torch.cat(
                    [mask, torch.zeros((pad_len,), dtype=torch.bool)], dim=0
                )
                attn = torch.cat([attn, torch.zeros((pad_len,), dtype=torch.long)], dim=0)
            padded_inputs.append(ids)
            padded_target_masks.append(mask)
            padded_attn.append(attn)

        input_ids = torch.stack(padded_inputs, dim=0)
        target_mask = torch.stack(padded_target_masks, dim=0)
        attention_mask = torch.stack(padded_attn, dim=0)

        g = torch.Generator(device="cpu").manual_seed(self.config.seed)
        corrupted_input_ids, labels = mask_target_tokens(
            input_ids=input_ids,
            target_mask=target_mask,
            mask_token_id=self.tokenizer.mask_token_id,  # type: ignore[arg-type]
            mask_prob=self.config.mask_prob,
            generator=g,
        )
        return {
            "input_ids": corrupted_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _train_one_dataset(self, dataset: Dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_batch,
        )
        self.model.train()
        self.optimizer.zero_grad()

        step = 0
        for _ in range(self.config.epochs_per_repo):
            for batch in tqdm(dataloader, desc="Diffusion training"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = masked_denoising_loss(outputs.logits.float(), labels)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                step += 1
                if step % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

    def train(
        self,
        repos: List[LeanGitRepo],
        database: DynamicDatabase,
        data_path: Path,
        objective: Optional[str] = None,
    ):
        if objective:
            self.config.objective = objective  # type: ignore[assignment]

        repos_to_process = []
        for repo in repos:
            repos_to_process.append(repo)
            database.export_merged_data(repos_to_process, data_path)
            train_json_path = os.path.join(data_path, "random", "train.json")
            dataset = self._build_dataset(train_json_path)
            if len(dataset) == 0:
                continue
            self._train_one_dataset(dataset)

        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

    def sample(self, prompt: str, n: int = 1, max_new_tokens: int = 64) -> List[str]:
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                num_return_sequences=n,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
