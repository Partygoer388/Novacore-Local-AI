"""LoRA 微调训练器(transformers + peft,均为可选依赖):
- 数据集:JSONL,每行 {"prompt","completion"} 或 {"text"}
- prompt 部分 label 置 -100(只学习回答部分)
- 流式日志 + 步级进度 + 手动停止(训练完当前步后安全中止并保存)
- 产出 LoRA 适配器目录(peft 格式,可与基座模型合并或配合推理框架加载)
"""
from __future__ import annotations

import json
import math
import threading
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal


class TrainAborted(RuntimeError):
    """用户手动停止(非错误)。"""


class TrainWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)          # 0-100
    done = pyqtSignal(bool, str)        # (成功, 消息/输出目录)

    def __init__(self, base_model: str, dataset_path: str, output_dir: str,
                 epochs: int = 3, lr: float = 1e-4, batch_size: int = 1,
                 lora_r: int = 8, max_samples: int = 0, max_len: int = 512,
                 parent=None):
        super().__init__(parent)
        self.base_model = str(base_model).strip()
        self.dataset_path = str(dataset_path).strip()
        self.output_dir = str(output_dir)
        self.epochs = max(1, int(epochs))
        self.lr = float(lr)
        self.batch_size = max(1, int(batch_size))
        self.lora_r = max(1, int(lora_r))
        self.max_samples = max(0, int(max_samples))
        self.max_len = max(64, int(max_len))
        self._stop_evt = threading.Event()

    # ---------- 取消 ----------
    def stop(self) -> None:
        self._stop_evt.set()

    def _check_stop(self) -> None:
        if self._stop_evt.is_set():
            raise TrainAborted()

    # ---------- 入口 ----------
    def run(self) -> None:
        try:
            self._train()
        except TrainAborted:
            self.log.emit("⚠️ 训练已手动停止")
            self.done.emit(False, "训练已手动停止")
        except Exception as e:
            self.log.emit(f"❌ 训练失败: {e}")
            self.done.emit(False, f"训练失败: {e}")

    # ---------- 依赖 ----------
    def _import_deps(self):
        try:
            import torch
        except ImportError as e:
            raise RuntimeError("未安装 torch(PyTorch),请先在「依赖管理」页安装") from e
        try:
            from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                      Trainer, TrainingArguments,
                                      TrainerCallback)
        except ImportError as e:
            raise RuntimeError("未安装 transformers,请先在「依赖管理」页安装") from e
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise RuntimeError("未安装 peft(LoRA 库),请先在「依赖管理」页安装:"
                               "pip install peft") from e
        return (torch, AutoModelForCausalLM, AutoTokenizer, Trainer,
                TrainingArguments, TrainerCallback, LoraConfig, get_peft_model)

    # ---------- 数据 ----------
    def _load_rows(self) -> list[tuple[str, str]]:
        p = Path(self.dataset_path)
        if not p.exists():
            raise RuntimeError(f"数据集文件不存在: {self.dataset_path}")
        rows: list[tuple[str, str]] = []
        with open(p, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"数据集第 {lineno} 行不是合法 JSON: {e}") from e
                if not isinstance(obj, dict):
                    continue
                if "text" in obj:
                    prompt, completion = "", str(obj["text"])
                else:
                    prompt = str(obj.get("prompt", ""))
                    completion = str(obj.get("completion", ""))
                if not (prompt or completion):
                    continue
                rows.append((prompt, completion))
                if self.max_samples and len(rows) >= self.max_samples:
                    break
        if not rows:
            raise RuntimeError(
                '数据集为空:需要 JSONL 文件,每行 {"prompt","completion"} 或 {"text"}')
        return rows

    # ---------- 主流程 ----------
    def _train(self) -> None:
        (torch, AutoModelForCausalLM, AutoTokenizer, Trainer,
         TrainingArguments, TrainerCallback, LoraConfig, get_peft_model) = \
            self._import_deps()

        base = self.base_model
        if not base:
            raise RuntimeError("请填写基座模型(HF 模型目录路径或仓库名)")
        if base.lower().endswith(".gguf"):
            raise RuntimeError("GGUF 模型不支持 LoRA 微调;请填写 HF 格式模型目录"
                               "(含 config.json/safetensors)或 HuggingFace 仓库名")

        self._check_stop()
        self.log.emit(f"📥 加载基座模型: {base}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(f"加载分词器失败({base}): {e}") from e
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        cuda = torch.cuda.is_available()
        dtype = torch.float16 if cuda else torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(
                base, torch_dtype=dtype, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(f"加载模型失败({base}): {e}") from e
        model.to("cuda" if cuda else "cpu")
        if tokenizer.pad_token == "[PAD]":
            model.resize_token_embeddings(len(tokenizer))

        self._check_stop()
        rows = self._load_rows()
        self.log.emit(f"📚 数据集样本数: {len(rows)}")

        examples: list[dict] = []
        for i, (prompt, completion) in enumerate(rows):
            self._check_stop()
            full_ids = tokenizer(prompt + completion, truncation=True,
                                 max_length=self.max_len,
                                 add_special_tokens=True)["input_ids"]
            if not full_ids:
                continue
            prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"] \
                if prompt else []
            plen = min(len(prompt_ids), len(full_ids))
            labels = list(full_ids)
            for j in range(plen):
                labels[j] = -100
            examples.append({"input_ids": full_ids, "labels": labels})
            if (i + 1) % 200 == 0:
                self.log.emit(f"   已编码 {i + 1}/{len(rows)} 条")
        if not examples:
            raise RuntimeError("数据集编码后为空,请检查内容")
        self.log.emit(f"✅ 编码完成,有效样本 {len(examples)} 条")

        self._check_stop()
        self.log.emit("🧩 附加 LoRA 适配器(r="
                      f"{self.lora_r})...")
        lora_cfg = LoraConfig(
            r=self.lora_r, lora_alpha=self.lora_r * 2, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        try:
            model = get_peft_model(model, lora_cfg)
        except Exception as e:
            raise RuntimeError(f"模型不支持所选拒绝目标模块(q/k/v/o_proj): {e}") from e
        model.print_trainable_parameters()

        steps_per_epoch = max(1, math.ceil(len(examples) / self.batch_size))
        total_steps = steps_per_epoch * self.epochs

        worker = self

        class _Cb(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kw):
                if logs and "loss" in logs:
                    worker.log.emit(f"   step {state.global_step}/{total_steps}"
                                    f"  loss={logs['loss']:.4f}")

            def on_step_end(self, args, state, control, **kw):
                if worker._stop_evt.is_set():
                    control.should_training_stop = True
                pct = int(state.global_step / max(total_steps, 1) * 100)
                worker.progress.emit(min(pct, 99))

        out_dir = Path(self.output_dir)
        tmp_dir = out_dir.parent / (out_dir.name + "_ckpt")
        args = TrainingArguments(
            output_dir=str(tmp_dir),
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.lr,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
        )
        trainer = Trainer(model=model, args=args, train_dataset=examples,
                          callbacks=[_Cb()])

        self.log.emit(f"🚀 开始训练:{self.epochs} 轮 × {steps_per_epoch} 步,"
                      f"共 {total_steps} 步(设备: {'CUDA' if cuda else 'CPU'})")
        self._check_stop()
        trainer.train()

        self._check_stop()
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        meta = {
            "base_model": base,
            "epochs": self.epochs,
            "learning_rate": self.lr,
            "batch_size": self.batch_size,
            "lora_r": self.lora_r,
            "samples": len(examples),
        }
        (out_dir / "train_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            if tmp_dir.exists():
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
        except OSError:
            pass
        self.progress.emit(100)
        self.log.emit(f"✅ 训练完成,LoRA 适配器已保存: {out_dir}")
        self.done.emit(True, str(out_dir))
