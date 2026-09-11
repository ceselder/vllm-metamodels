"""vllm-metamodels: merge a LoRA adapter INTO the served base weights, in place, on the worker.

Why: an RL rollout engine that serves the current policy as a rank-64 LoRA pays for the
LoRA shrink/expand kernels on every layer of every decode step.  Merging the adapter
(``W <- W0 + s * B @ A``) once per publish lets the same engine generate with plain
GEMMs and no LoRA kernels at all (``enable_lora`` can even be left off).  This module
does the merge on the worker from the adapter's ``(A, B)`` tensors -- nothing of the
size of the weights crosses a process boundary -- and keeps enough state to revert
or replace the adapter exactly:

``keep_base="gpu"``   a bf16 copy of every LoRA-targeted weight stays on the device
                      (~= the size of those weights, 48 GB on Qwen3.6-27B); every publish is
                      ``W = round(W0 + delta)``: ONE rounding, no drift, unmerge is a copy.
``keep_base="cpu"``   the same copy in pinned host memory; publish streams it back
                      (host->device bandwidth bound).
``keep_base="none"``  no copy: publish does ``W = round(W - delta_prev + delta_new)`` in fp32;
                      exact within one bf16 rounding per publish, so the base drifts as a
                      random walk of half-ulps (measure with ``lora_drift_test``).
``keep_base="auto"``  "gpu" if the copy fits with a 2 GB margin, else "cpu".

Layout knowledge comes from vLLM's own linear layers (duck-typed on attribute / class
names so it works across vLLM releases): fused ``qkv_proj`` = [q | k | v] rows,
``gate_up_proj`` = [gate | up] rows (``output_sizes``), column-parallel output shards and
row-parallel input shards for TP > 1 (implemented from the same rules
``weight_loader`` uses; only TP = 1 is GPU-tested -- TP > 1 requires
``VLLM_LENS_MERGE_ALLOW_TP=1``).  Quantized weights are refused.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

_PREFIX = "base_model.model."


@dataclass(slots=True)
class SubSlice:
    """One HF (un-fused) linear module inside a vLLM parameter."""

    hf_name: str
    """HF module name (``...layers.3.self_attn.q_proj``), as it appears in a PEFT adapter."""
    rows: tuple[int, int]
    """Row range of ``param`` holding this module's output shard."""
    row_src: tuple[int, int]
    """Row range of the FULL HF weight / LoRA B that this shard corresponds to."""
    col_src: tuple[int, int]
    """Column range of the FULL HF weight / LoRA A held by this TP rank."""
    out_full: int
    in_full: int


@dataclass(slots=True)
class LinearTarget:
    vllm_name: str
    param: torch.Tensor
    subs: list[SubSlice] = field(default_factory=list)
    kind: str = ""
    output_sizes: list[int] | None = None
    """Slice sizes of a ``merged_col_deferred`` layer (rows assigned from the adapter at merge time)."""


def _mro_names(obj: Any) -> set[str]:
    return {c.__name__ for c in type(obj).__mro__}


def _base_layer(mod: Any) -> Any:
    """LoRA wrapper (``*WithLoRA``) -> its ``base_layer``; plain linear -> itself."""
    base = getattr(mod, "base_layer", None)
    return base if base is not None and hasattr(mod, "output_slices") else mod


def _packed_mapping(model: Any) -> dict[str, list[str]]:
    m = getattr(model, "packed_modules_mapping", None) or {}
    return {k: list(v) for k, v in m.items()}


def discover_targets(model: torch.nn.Module, only: set[str] | None = None) -> list[LinearTarget]:
    """Every linear module of ``model`` that can carry a LoRA, with the HF sub-module
    layout of its weight.  ``only`` restricts to vLLM module names (or their HF sub-names)."""
    packed = _packed_mapping(model)
    out: list[LinearTarget] = []
    seen_weights: set[int] = set()
    for name, mod in model.named_modules():
        if name.endswith(".base_layer"):
            continue  # the linear inside a LoRA wrapper: already visited through the wrapper's name
        base = _base_layer(mod)
        names = _mro_names(base)
        if "LinearBase" not in names and not any(n.endswith("ParallelLinear") or n == "ReplicatedLinear" for n in names):
            continue
        if not hasattr(base, "weight") or not isinstance(getattr(base, "weight"), torch.Tensor):
            continue
        w: torch.Tensor = base.weight
        if w.dim() != 2 or not w.is_floating_point():
            continue  # quantized / packed storage: refuse silently here, loudly at merge time
        if id(w) in seen_weights:
            continue  # one target per weight tensor, whatever the module path
        seen_weights.add(id(w))
        suffix = name.rsplit(".", 1)[-1]
        prefix = name[: -len(suffix)] if suffix != name else ""
        hf_subs = packed.get(suffix, [suffix])
        tp_size = int(getattr(base, "tp_size", 1) or 1)
        tp_rank = int(getattr(base, "tp_rank", 0) or 0)
        in_size = int(getattr(base, "input_size", w.shape[1]))
        out_size = int(getattr(base, "output_size", w.shape[0]))
        subs: list[SubSlice] = []
        kind = ""
        if "QKVParallelLinear" in names:
            kind = "qkv"
            hs = int(base.head_size)
            q_shard = int(base.num_heads) * hs
            kv_shard = int(base.num_kv_heads) * hs
            q_full = int(base.total_num_heads) * hs
            kv_full = int(base.total_num_kv_heads) * hs
            repl = int(getattr(base, "num_kv_head_replicas", 1) or 1)
            kv_id = tp_rank // repl
            if len(hf_subs) != 3:  # unfused checkpoint naming: q_proj/k_proj/v_proj expected
                hf_subs = [f"{suffix}"] if len(hf_subs) == 1 else hf_subs
            sizes = [(q_shard, q_full, tp_rank * q_shard), (kv_shard, kv_full, kv_id * kv_shard), (kv_shard, kv_full, kv_id * kv_shard)]
            r = 0
            for hf, (shard, full, src0) in zip(hf_subs, sizes):
                subs.append(SubSlice(prefix + hf, (r, r + shard), (src0, src0 + shard), (0, in_size), full, in_size))
                r += shard
        elif "MergedColumnParallelLinear" in names or (hasattr(base, "output_sizes") and "ColumnParallelLinear" in names):
            kind = "merged_col"
            output_sizes = [int(s) for s in base.output_sizes]
            if len(hf_subs) == 1 and len(output_sizes) > 1:
                # ONE HF module whose rows span several vLLM output slices (Qwen3-Next's GDN
                # in_proj_qkvz = [q|k|v|z], in_proj_ba = [b|a]; vLLM's "variable slice" LoRA layer).
                # On TP = 1 the param rows are the HF rows in order; TP > 1 interleaves per-slice shards.
                if tp_size > 1:
                    raise NotImplementedError(
                        f"vllm-lens: {name}: a single LoRA module spanning {len(output_sizes)} output slices is only "
                        "supported at tensor_parallel_size=1"
                    )
                total = sum(output_sizes)
                subs.append(SubSlice(prefix + hf_subs[0], (0, total), (0, total), (0, in_size), total, in_size))
                kind = "merged_col_single"
            elif 1 < len(hf_subs) < len(output_sizes):
                # Several HF modules over MORE vLLM slices (vLLM 0.19 Qwen3-Next without LoRA: in_proj_qkvz =
                # [q|k|v|z] with HF modules in_proj_qkv + in_proj_z).  Which slices belong to which module is
                # only known from the adapter's B row counts -> resolved at merge time (``_per_param``),
                # all of the param's HF modules must then be present.  TP = 1 only.
                if tp_size > 1:
                    raise NotImplementedError(f"vllm-lens: {name}: deferred slice partition needs tensor_parallel_size=1")
                for hf in hf_subs:
                    subs.append(SubSlice(prefix + hf, (-1, -1), (-1, -1), (0, in_size), -1, in_size))
                kind = "merged_col_deferred"
            else:
                if len(hf_subs) != len(output_sizes):
                    raise ValueError(
                        f"vllm-lens: {name}: packed_modules_mapping lists {len(hf_subs)} HF modules but the layer has "
                        f"{len(output_sizes)} output slices"
                    )
                r = 0
                for hf, full in zip(hf_subs, output_sizes):
                    shard = full // tp_size
                    subs.append(SubSlice(prefix + hf, (r, r + shard), (tp_rank * shard, (tp_rank + 1) * shard), (0, in_size), full, in_size))
                    r += shard
        elif "RowParallelLinear" in names:
            kind = "row"
            ips = int(getattr(base, "input_size_per_partition", w.shape[1]))
            subs.append(SubSlice(prefix + hf_subs[0], (0, w.shape[0]), (0, out_size), (tp_rank * ips, (tp_rank + 1) * ips), out_size, in_size))
        elif "ColumnParallelLinear" in names:
            kind = "col"
            ops = int(getattr(base, "output_size_per_partition", w.shape[0]))
            subs.append(SubSlice(prefix + hf_subs[0], (0, ops), (tp_rank * ops, (tp_rank + 1) * ops), (0, in_size), out_size, in_size))
        elif "ReplicatedLinear" in names or "LinearBase" in names:
            kind = "replicated"
            subs.append(SubSlice(prefix + hf_subs[0], (0, w.shape[0]), (0, w.shape[0]), (0, w.shape[1]), w.shape[0], w.shape[1]))
        else:
            continue
        if only is not None and name not in only and not any(s.hf_name in only for s in subs):
            continue
        tgt = LinearTarget(name, w, subs, kind)
        if kind == "merged_col_deferred":
            tgt.output_sizes = [int(s) for s in base.output_sizes]
        out.append(tgt)
    return out


def layout_summary(targets: list[LinearTarget], with_norms: bool = False) -> list[dict[str, Any]]:
    """JSON-able description (for adapter synthesis / introspection)."""
    rows = []
    for t in targets:
        subs = []
        for s in t.subs:
            deferred = s.out_full < 0
            d: dict[str, Any] = {"hf_name": s.hf_name, "out": None if deferred else s.out_full, "in": s.in_full, "rows": None if deferred else list(s.rows)}
            if with_norms and not deferred:
                d["frob"] = float(t.param[s.rows[0] : s.rows[1], :].float().norm())
            subs.append(d)
        rows.append({"vllm_name": t.vllm_name, "kind": t.kind, "shape": list(t.param.shape), "dtype": str(t.param.dtype), "subs": subs,
                     **({"output_sizes": t.output_sizes} if t.output_sizes else {})})
    return rows


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------


def _strip(name: str) -> str:
    return name[len(_PREFIX) :] if name.startswith(_PREFIX) else name


def parse_adapter_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """``{"base_model.model.X.lora_A.weight": A, ".lora_B.weight": B}`` -> ``{X: {"A": A, "B": B}}``.
    Accepts PEFT's ``lora_A.weight`` / ``lora_A.default.weight`` / ``lora_A.<adapter>.weight``."""
    out: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in tensors.items():
        parts = _strip(k).split(".")
        if parts[-1] != "weight":
            raise ValueError(f"unsupported LoRA tensor name {k!r}")
        idx = next((i for i, p in enumerate(parts) if p in ("lora_A", "lora_B")), None)
        if idx is None:
            raise ValueError(f"unsupported LoRA tensor name {k!r} (no lora_A / lora_B)")
        module = ".".join(parts[:idx])
        out.setdefault(module, {})["A" if parts[idx] == "lora_A" else "B"] = v
    for m, ab in out.items():
        if "A" not in ab or "B" not in ab:
            raise ValueError(f"LoRA module {m!r} is missing lora_A or lora_B")
        if ab["A"].dim() != 2 or ab["B"].dim() != 2 or ab["A"].shape[0] != ab["B"].shape[1]:
            raise ValueError(f"LoRA module {m!r}: A {tuple(ab['A'].shape)} / B {tuple(ab['B'].shape)} are not [r, in] / [out, r]")
    return out


def scaling_from_config(cfg: dict[str, Any]) -> float:
    r = int(cfg["r"])
    alpha = float(cfg.get("lora_alpha", r))
    return alpha / math.sqrt(r) if cfg.get("use_rslora", False) else alpha / r


def load_adapter_dir(path: str) -> tuple[dict[str, dict[str, torch.Tensor]], float, dict[str, Any]]:
    """PEFT adapter directory -> ``(modules, scaling, config)``."""
    p = Path(path)
    cfg = json.loads((p / "adapter_config.json").read_text())
    st = p / "adapter_model.safetensors"
    if st.exists():
        from safetensors.torch import load_file

        tensors = load_file(str(st))
    else:
        bin_ = p / "adapter_model.bin"
        if not bin_.exists():
            raise FileNotFoundError(f"no adapter_model.safetensors / adapter_model.bin in {path}")
        tensors = torch.load(bin_, map_location="cpu", weights_only=True)
    return parse_adapter_tensors(tensors), scaling_from_config(cfg), cfg


def resolve_adapter(
    targets: list[LinearTarget], modules: dict[str, dict[str, torch.Tensor]], name_map: Any = None
) -> dict[str, list[tuple[LinearTarget, SubSlice]]]:
    """Adapter module name -> the (target, sub-slice) it lands in.  Unknown adapter
    modules raise (silently ignoring a trained module is how RL runs go wrong)."""
    by_hf: dict[str, tuple[LinearTarget, SubSlice]] = {}
    for t in targets:
        for s in t.subs:
            by_hf[s.hf_name] = (t, s)
    out: dict[str, list[tuple[LinearTarget, SubSlice]]] = {}
    missing: list[str] = []
    for m in modules:
        cands = [m]
        if name_map is not None:
            try:
                mapped = name_map(m)
                if mapped and mapped != m:
                    cands.append(mapped)
            except Exception:  # noqa: BLE001
                pass
        if m.startswith("model.") and not m.startswith("model.language_model."):
            # multimodal wrappers serve the text stack under a prefix; both layouts seen across vLLM versions:
            #   Qwen3_5ForConditionalGeneration: "language_model.model.layers.N.*" (vLLM 0.21) / "model.language_model.layers.N.*" (older)
            cands.append("model.language_model." + m[len("model."):])
            cands.append("language_model." + m)
        hit = next((c for c in cands if c in by_hf), None)
        if hit is None:
            missing.append(m)
            continue
        out[m] = [by_hf[hit]]
    if missing:
        raise ValueError(
            f"vllm-lens: {len(missing)} LoRA module(s) do not match any linear layer of the served model "
            f"(first: {missing[:5]}); known: {sorted(by_hf)[:5]} ... -- refusing to merge a partial adapter"
        )
    return out


def _check_shapes(m: str, ab: dict[str, torch.Tensor], s: SubSlice) -> None:
    A, B = ab["A"], ab["B"]
    if s.out_full < 0:  # deferred partition: rows are assigned from B in _per_param
        if A.shape[1] != s.in_full:
            raise ValueError(f"vllm-lens: LoRA module {m!r}: A {tuple(A.shape)} does not match the served input size {s.in_full}")
        return
    if A.shape[1] != s.in_full or B.shape[0] != s.out_full:
        raise ValueError(
            f"vllm-lens: LoRA module {m!r}: A {tuple(A.shape)} / B {tuple(B.shape)} do not match the served "
            f"weight [{s.out_full}, {s.in_full}]"
        )


def delta_for(ab: dict[str, torch.Tensor], s: SubSlice, scaling: float, device: torch.device) -> torch.Tensor:
    """``scaling * B[rows] @ A[:, cols]`` in float32 for this rank's shard."""
    A = ab["A"][:, s.col_src[0] : s.col_src[1]].to(device=device, dtype=torch.float32)
    B = ab["B"][s.row_src[0] : s.row_src[1], :].to(device=device, dtype=torch.float32)
    return (B @ A) * float(scaling)


# ---------------------------------------------------------------------------
# merge state
# ---------------------------------------------------------------------------


class LoRAMerger:
    """Owns the base-weight copies and the currently merged adapter for one worker.

    ``merge(..., keep_base=mode)``: ``gpu`` / ``cpu`` = exact ``round(W0 + delta)`` from a stored
    copy of the targeted weights (snapshotted on first use, on that device kind; ``auto`` picks
    gpu when it fits); ``none`` = ``round(W - delta_prev + delta_new)`` without copies (the
    previous adapter's A/B are kept on the host for that).  ``unmerge()`` copies W0 back when
    copies exist, else subtracts the merged adapter.  ``release_base()`` frees the copies."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.targets = discover_targets(model)
        self.by_name = {t.vllm_name: t for t in self.targets}
        self.base: dict[str, torch.Tensor] = {}  # vllm param name -> copy of W0 (gpu or pinned cpu)
        self.base_where: str | None = None
        self.mode: str = "none"  # mode of the last merge
        self.merged: dict[str, dict[str, torch.Tensor]] | None = None  # adapter currently in the weights (A/B on host)
        self.merged_scaling: float = 1.0
        self.merged_targets: list[str] = []
        self.publishes: int = 0
        self.last_publish_s: float = 0.0
        self.last_publish_breakdown: dict[str, float] = {}
        self.tp_size = 1
        for _n, mod in model.named_modules():
            base = _base_layer(mod)
            if hasattr(base, "tp_size"):
                self.tp_size = int(base.tp_size or 1)
                break
        self.name_map = None
        mapper = getattr(model, "hf_to_vllm_mapper", None)
        if mapper is not None and hasattr(mapper, "_map_name"):
            self.name_map = mapper._map_name

    # -- base copies -------------------------------------------------------

    def _device(self) -> torch.device:
        return self.targets[0].param.device if self.targets else torch.device("cpu")

    def _pick_mode(self, names: list[str], keep_base: str) -> str:
        if keep_base not in ("gpu", "cpu", "none", "auto"):
            raise ValueError(f"keep_base must be gpu|cpu|none|auto, got {keep_base!r}")
        if keep_base != "auto":
            return keep_base
        if self.base_where is not None:
            return self.base_where
        need = [n for n in names if n not in self.base]
        nbytes = sum(self.by_name[n].param.numel() * self.by_name[n].param.element_size() for n in need)
        dev = self._device()
        if dev.type != "cuda":
            return "gpu"
        free, _ = torch.cuda.mem_get_info(dev)
        return "gpu" if free > nbytes + (2 << 30) else "cpu"

    def _ensure_base(self, names: list[str], where: str) -> None:
        """Snapshot W0 for the params in ``names`` that have no copy yet, on the device kind of the
        existing copies (or ``where`` if there are none)."""
        where = self.base_where or where
        for n in names:
            if n in self.base:
                continue
            p = self.by_name[n].param.detach()
            if where == "gpu":
                self.base[n] = p.clone()
            else:
                try:
                    host = torch.empty(p.shape, dtype=p.dtype, pin_memory=True)
                except RuntimeError:  # pragma: no cover - no pinned memory
                    host = torch.empty(p.shape, dtype=p.dtype)
                host.copy_(p)
                self.base[n] = host
        self.base_where = where

    def release_base(self) -> int:
        n = len(self.base)
        self.base.clear()
        self.base_where = None
        return n

    def base_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.base.values())

    # -- merge / unmerge ---------------------------------------------------

    def _per_param(self, modules: dict[str, dict[str, torch.Tensor]]) -> dict[str, list[tuple[str, SubSlice]]]:
        per_param: dict[str, list[tuple[str, SubSlice]]] = {}
        for m, hits in resolve_adapter(self.targets, modules, self.name_map).items():
            for t, s in hits:
                _check_shapes(m, modules[m], s)
                per_param.setdefault(t.vllm_name, []).append((m, s))
        for pname, hits in per_param.items():
            t = self.by_name[pname]
            if t.kind != "merged_col_deferred":
                continue
            # every HF module of the param must be present; assign contiguous slice groups in mapping order
            present = {s.hf_name: m for m, s in hits}
            missing = [s.hf_name for s in t.subs if s.hf_name not in present]
            if missing:
                raise ValueError(f"vllm-lens: {pname}: adapter must cover all of {[s.hf_name for s in t.subs]} (missing {missing})")
            sizes = list(t.output_sizes or [])
            row, si = 0, 0
            resolved: list[tuple[str, SubSlice]] = []
            for s in t.subs:
                m = present[s.hf_name]
                out = int(modules[m]["B"].shape[0])
                acc, sj = 0, si
                while sj < len(sizes) and acc < out:
                    acc += sizes[sj]
                    sj += 1
                if acc != out:
                    raise ValueError(f"vllm-lens: {pname}: LoRA module {m!r} has {out} output rows, which is not a run of "
                                     f"output slices {sizes} starting at slice {si}")
                resolved.append((m, SubSlice(s.hf_name, (row, row + out), (0, out), s.col_src, out, s.in_full)))
                row, si = row + out, sj
            if row != int(t.param.shape[0]):
                raise ValueError(f"vllm-lens: {pname}: adapter modules cover {row} rows of a {int(t.param.shape[0])}-row weight")
            per_param[pname] = resolved
        return per_param

    @torch.no_grad()
    def merge(self, modules: dict[str, dict[str, torch.Tensor]], scaling: float, keep_base: str = "auto") -> dict[str, Any]:
        """Install ``modules`` (``{hf_module: {"A", "B"}}``) as the served adapter, replacing any
        previous one.  Returns timing / mode / counts."""
        t0 = time.perf_counter()
        per_param = self._per_param(modules)
        names = sorted(per_param)
        mode = self._pick_mode(names, keep_base)
        t_resolve = time.perf_counter()
        if mode in ("gpu", "cpu"):
            self._ensure_base(names, mode)
        t_snapshot = time.perf_counter()
        dev = self._device()
        prev_per_param: dict[str, list[tuple[str, SubSlice]]] = {}
        if mode == "none" and self.merged is not None:
            prev_per_param = self._per_param(self.merged)
        n_rows = 0
        for pname in sorted(set(per_param) | set(prev_per_param)):
            W = self.by_name[pname].param
            if mode == "none":
                acc = W.float()
                for m, s in prev_per_param.get(pname, ()):  # remove the previous adapter's contribution
                    acc[s.rows[0] : s.rows[1], :] -= delta_for(self.merged[m], s, self.merged_scaling, dev)  # type: ignore[index]
            else:
                src = self.base[pname]
                acc = src.to(device=W.device, non_blocking=True).float() if src.device != W.device else src.float()
            for m, s in per_param.get(pname, ()):
                acc[s.rows[0] : s.rows[1], :] += delta_for(modules[m], s, scaling, dev)
                n_rows += s.rows[1] - s.rows[0]
            W.copy_(acc.to(W.dtype))
            del acc
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t_apply = time.perf_counter()
        # keep A/B (tiny, host) so a later "none"-mode publish or unmerge can subtract them
        self.merged = {m: {"A": ab["A"].detach().to("cpu"), "B": ab["B"].detach().to("cpu")} for m, ab in modules.items()}
        self.merged_scaling = float(scaling)
        self.merged_targets = names
        self.mode = mode
        self.publishes += 1
        self.last_publish_s = t_apply - t0
        self.last_publish_breakdown = {"resolve_s": t_resolve - t0, "snapshot_s": t_snapshot - t_resolve, "apply_s": t_apply - t_snapshot}
        return {
            "mode": mode,
            "n_modules": len(modules),
            "n_params": len(names),
            "n_rows": n_rows,
            "scaling": float(scaling),
            "publish_s": self.last_publish_s,
            **self.last_publish_breakdown,
            "base_bytes": self.base_bytes(),
            "base_where": self.base_where,
            "publishes": self.publishes,
        }

    @torch.no_grad()
    def unmerge(self, release: bool = False, how: str = "auto") -> dict[str, Any]:
        """Restore the base weights: copy W0 back when copies exist (exact), else subtract the
        merged adapter (one more rounding).  ``how`` forces ``"copy"`` (error without copies) or
        ``"subtract"``.  ``release=True`` also frees the base copies."""
        t0 = time.perf_counter()
        n = 0
        if how not in ("auto", "copy", "subtract"):
            raise ValueError(f"how must be auto|copy|subtract, got {how!r}")
        have_copies = bool(self.base) and all(name in self.base for name in self.merged_targets)
        if how == "copy" and self.merged is not None and not have_copies:
            raise ValueError("unmerge(how='copy') needs base copies for every merged parameter")
        if self.merged is not None:
            if have_copies and how != "subtract":
                how = "copy"
                for name in self.merged_targets:
                    W = self.by_name[name].param
                    W.copy_(self.base[name], non_blocking=True)
                    n += 1
            else:
                how = "subtract"
                dev = self._device()
                for pname, hits in self._per_param(self.merged).items():
                    W = self.by_name[pname].param
                    acc = W.float()
                    for m, s in hits:
                        acc[s.rows[0] : s.rows[1], :] -= delta_for(self.merged[m], s, self.merged_scaling, dev)
                    W.copy_(acc.to(W.dtype))
                    n += 1
            if self._device().type == "cuda":
                torch.cuda.synchronize(self._device())
        else:
            how = "nothing"
        self.merged = None
        self.merged_targets = []
        if release:
            self.release_base()
        return {"restored_params": n, "how": how, "unmerge_s": time.perf_counter() - t0, "base_bytes": self.base_bytes()}

    def status(self) -> dict[str, Any]:
        return {
            "merged": self.merged is not None,
            "n_modules": len(self.merged or {}),
            "n_params": len(self.merged_targets),
            "mode": self.mode,
            "base_where": self.base_where,
            "base_bytes": self.base_bytes(),
            "publishes": self.publishes,
            "last_publish_s": self.last_publish_s,
            "n_targets": len(self.targets),
            "tp_size": self.tp_size,
        }

    # -- diagnostics ---------------------------------------------------------

    @torch.no_grad()
    def fingerprint(self, names: list[str] | None = None) -> dict[str, list[float]]:
        """Per-parameter (sum, sum of squares) in float64 -- cheap equality evidence."""
        out: dict[str, list[float]] = {}
        for n, t in self.by_name.items():
            if names is not None and n not in names:
                continue
            w = t.param.double()
            out[n] = [float(w.sum()), float((w * w).sum())]
        return out

    @torch.no_grad()
    def compare_to_exact(self, modules: dict[str, dict[str, torch.Tensor]] | None, scaling: float = 1.0) -> dict[str, Any]:
        """Drift of the CURRENT weights vs an exact ``round(W0 + delta)`` merge of ``modules``
        (``None`` = vs the base itself; needs base copies).  Reports, over the targeted params:
        max |diff| (absolute, and in units of the bf16 spacing at the parameter's RMS magnitude,
        ``rms * 2^-7`` -- a "local ulp" of a near-zero element would inflate ordinary rounding
        noise), relative Frobenius error and the fraction of changed elements."""
        if not self.base:
            raise ValueError("compare_to_exact needs base copies (merge with keep_base gpu|cpu first)")
        per_param = self._per_param(modules) if modules else {n: [] for n in self.base}
        dev = self._device()
        max_ulps = max_abs = num = den = 0.0
        changed = total = 0
        for pname, hits in per_param.items():
            W = self.by_name[pname].param
            exact = self.base[pname].to(W.device).float()
            for m, s in hits:
                exact[s.rows[0] : s.rows[1], :] += delta_for(modules[m], s, scaling, dev)  # type: ignore[index]
            exact_b = exact.to(W.dtype).float()
            diff = W.float() - exact_b
            rms = float(exact_b.pow(2).mean().sqrt()) or 1e-30
            max_ulps = max(max_ulps, float(diff.abs().max()) / (rms * 2.0 ** -7))
            max_abs = max(max_abs, float(diff.abs().max()))
            num += float((diff * diff).sum())
            den += float((exact_b * exact_b).sum())
            changed += int((diff != 0).sum())
            total += diff.numel()
            del exact, exact_b, diff
        return {
            "max_abs_diff": max_abs,
            "max_diff_ulps": max_ulps,
            "rel_frobenius": math.sqrt(num / den) if den else 0.0,
            "frac_changed": changed / total if total else 0.0,
            "n_params": len(per_param),
        }


def synth_adapter(layout: list[dict[str, Any]], rank: int, rel_norm: float, seed: int, alpha: float = 16.0,
                  rslora: bool = True, dtype: torch.dtype = torch.bfloat16) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """A random PEFT-format LoRA whose per-module ``||s B A||_F / ||W||_F`` is ``rel_norm``
    (``layout`` from ``layout_summary(..., with_norms=True)``).  Returns ``(tensors, adapter_config)``."""
    g = torch.Generator().manual_seed(seed)
    cfg = {"peft_type": "LORA", "r": rank, "lora_alpha": alpha, "use_rslora": rslora, "lora_dropout": 0.0, "bias": "none",
           "task_type": "CAUSAL_LM", "fan_in_fan_out": False, "init_lora_weights": True, "target_modules": []}
    scaling = scaling_from_config(cfg)
    tensors: dict[str, torch.Tensor] = {}
    targets: set[str] = set()
    for t in layout:
        for s in t["subs"]:
            A = torch.randn(rank, s["in"], generator=g) / math.sqrt(s["in"])
            B = torch.randn(s["out"], rank, generator=g)
            delta_norm = float(scaling) * float((B @ A).norm())
            want = rel_norm * float(s.get("frob") or math.sqrt(s["out"] * s["in"]) * 0.02)
            B *= want / max(delta_norm, 1e-12)
            tensors[f"{_PREFIX}{s['hf_name']}.lora_A.weight"] = A.to(dtype)
            tensors[f"{_PREFIX}{s['hf_name']}.lora_B.weight"] = B.to(dtype)
            targets.add(s["hf_name"].rsplit(".", 1)[-1])
    cfg["target_modules"] = sorted(targets)
    return tensors, cfg


def save_adapter(path: str, tensors: dict[str, torch.Tensor], cfg: dict[str, Any]) -> None:
    from safetensors.torch import save_file

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(p / "adapter_model.safetensors"), metadata={"format": "pt"})
    (p / "adapter_config.json").write_text(json.dumps(cfg, indent=1))


def tp_allowed(tp_size: int) -> bool:
    return tp_size <= 1 or os.environ.get("VLLM_LENS_MERGE_ALLOW_TP", "").strip().lower() in ("1", "true", "yes", "on")
