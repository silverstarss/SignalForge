"""Transactional standalone experiment execution for RewardScope."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rewardscope.config import RunConfig, load_run_config_with_requested
from rewardscope.datasets import DatasetExample, load_dataset_result
from rewardscope.extraction import NumericExtractionConfig
from rewardscope.io import write_rollouts_jsonl
from rewardscope.io.atomic import atomic_write_json, atomic_write_jsonl
from rewardscope.metrics import PromptGroupMetricsResult, PromptGroupSummary
from rewardscope.reports.analysis import AnalysisArtifacts, analyze_rollouts_jsonl, write_analysis_report
from rewardscope.rollouts import (
    RolloutInput,
    build_math_verify_latex_rollout,
    build_math_verify_numeric_rollout,
    build_numeric_rollout,
)
from rewardscope.sampling import GeneratedResponse, TransformersSampler
from rewardscope.sampling.transformers import build_generation_kwargs

if TYPE_CHECKING:
    from rewardscope.reports.plots import AnalysisPlotArtifacts


@dataclass(frozen=True)
class ExperimentArtifacts:
    output_dir: Path
    inputs_jsonl: Path
    rendered_prompt_json: Path
    rollouts_jsonl: Path
    config_snapshot_json: Path
    provenance_json: Path
    manifest_json: Path
    metrics_result: PromptGroupMetricsResult
    summary: PromptGroupSummary
    report: AnalysisArtifacts
    plots: AnalysisPlotArtifacts | None


def run_experiment(config: RunConfig) -> ExperimentArtifacts:
    """Run one complete diagnostics experiment with transactional output commit."""
    return _run_experiment(config, requested_config=_serialize_value(config))


def run_experiment_from_yaml(path: str | Path) -> ExperimentArtifacts:
    """Load strict YAML then run an experiment while preserving requested fields."""
    config, requested = load_run_config_with_requested(path)
    return _run_experiment(config, requested_config=requested)


def _run_experiment(config: RunConfig, *, requested_config: dict[str, Any]) -> ExperimentArtifacts:
    if not isinstance(config, RunConfig):
        raise TypeError("config must be a RunConfig.")

    final_dir = config.output.output_dir.expanduser().resolve()
    _validate_final_directory(final_dir)
    staging_dir = _create_staging_directory(final_dir, config.output.run_id)
    phase = "preflight"
    try:
        if config.analysis.write_plots:
            _preflight_plotting()

        phase = "dataset"
        dataset_result = load_dataset_result(config.dataset)
        examples = list(dataset_result.examples)
        _validate_examples(examples)

        phase = "model"
        sampler = TransformersSampler.from_pretrained(config.model)
        resolved_config = _resolved_config(config, final_dir)
        inputs_path = staging_dir / "inputs.jsonl"
        rendered_prompt_path = staging_dir / "rendered_prompt.json"
        config_path = staging_dir / "config_snapshot.json"
        provenance_path = staging_dir / "provenance.json"
        atomic_write_json(config_path, {"requested": requested_config, "resolved": resolved_config})
        atomic_write_jsonl(inputs_path, [_input_row(example, config) for example in examples])
        atomic_write_json(rendered_prompt_path, _rendered_prompt_row(examples[0], sampler))
        atomic_write_json(provenance_path, _build_provenance(config, sampler, dataset_result, examples))

        phase = "generation"
        responses = sampler.generate([_prompt_input(example) for example in examples], config.sampling)
        _validate_generated_responses(responses, prompt_count=len(examples), num_samples=config.sampling.num_samples)

        phase = "verification"
        records = []
        for response in responses:
            example = examples[response.prompt_index]
            rollout_input = RolloutInput(
                run_id=config.output.run_id,
                prompt_id=example.prompt_id,
                sample_id=response.sample_index,
                model_name=config.model.name,
                dataset_name=example.dataset_name,
                split=example.split,
                generation_seed=config.sampling.generation_seed,
                temperature=config.sampling.temperature,
                top_p=config.sampling.top_p,
                max_new_tokens=config.sampling.max_new_tokens,
                batch_size=config.sampling.batch_size,
                prompt=example.prompt,
                response=response.response,
                ground_truth=example.ground_truth,
                prompt_tokens=response.prompt_tokens,
                response_tokens=response.response_tokens,
                finish_reason=response.finish_reason,
                hit_max_length=response.hit_max_length,
            )
            records.append(_build_rollout(config, rollout_input))
        if not records:
            raise ValueError("Sampler produced no rollout records.")

        phase = "rollouts"
        rollouts_path = staging_dir / "rollouts.jsonl"
        write_rollouts_jsonl(rollouts_path, records)

        phase = "analysis"
        result, summary = analyze_rollouts_jsonl(
            rollouts_path,
            expected_group_size=config.sampling.num_samples,
            strict=config.analysis.strict,
            k_values=config.analysis.k_values,
        )
        if summary is None:
            raise ValueError("Successful experiments require a non-empty analysis summary.")
        report = write_analysis_report(staging_dir / "analysis", result, summary)

        plots = None
        if config.analysis.write_plots:
            phase = "plots"
            from rewardscope.reports.plots import write_analysis_plots
            plots = write_analysis_plots(staging_dir / "analysis", result)

        phase = "manifest"
        manifest_path = staging_dir / "manifest.json"
        atomic_write_json(manifest_path, _build_manifest(staging_dir))
        _commit_staging_directory(staging_dir, final_dir)
        return ExperimentArtifacts(
            output_dir=final_dir,
            inputs_jsonl=final_dir / "inputs.jsonl",
            rendered_prompt_json=final_dir / "rendered_prompt.json",
            rollouts_jsonl=final_dir / "rollouts.jsonl",
            config_snapshot_json=final_dir / "config_snapshot.json",
            provenance_json=final_dir / "provenance.json",
            manifest_json=final_dir / "manifest.json",
            metrics_result=result,
            summary=summary,
            report=_relocate_report(report, final_dir / "analysis"),
            plots=_relocate_plots(plots, final_dir / "analysis"),
        )
    except BaseException as error:
        _handle_failure(staging_dir, config.output.keep_failed_run, phase, error)
        raise


def _validate_final_directory(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Final output directory must be absent or empty: {path}")


def _create_staging_directory(final_dir: Path, run_id: str) -> Path:
    root = final_dir.parent / ".staging"
    root.mkdir(parents=True, exist_ok=True)
    staging_dir = root / f"{run_id}-{uuid.uuid4().hex}"
    staging_dir.mkdir()
    return staging_dir


def _commit_staging_directory(staging_dir: Path, final_dir: Path) -> None:
    if final_dir.exists():
        final_dir.rmdir()
    os.replace(staging_dir, final_dir)


def _handle_failure(staging_dir: Path, keep_failed_run: bool, phase: str, error: BaseException) -> None:
    if not staging_dir.exists():
        return
    if keep_failed_run:
        atomic_write_json(staging_dir / "failure.json", {
            "status": "failed", "phase": phase,
            "error_type": type(error).__name__, "message": str(error),
        })
        return
    shutil.rmtree(staging_dir)


def _validate_examples(examples: list[DatasetExample]) -> None:
    if not examples:
        raise ValueError("Dataset selection produced no examples.")
    prompt_ids = [example.prompt_id for example in examples]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("Dataset selection contains duplicate prompt_id values.")


def _validate_generated_responses(
    responses: object, *, prompt_count: int, num_samples: int
) -> list[GeneratedResponse]:
    if not isinstance(responses, list) or len(responses) != prompt_count * num_samples:
        raise ValueError("Sampler response count violates the prompt/sample contract.")
    seen: set[tuple[int, int]] = set()
    for expected_prompt in range(prompt_count):
        for expected_sample in range(num_samples):
            response = responses[expected_prompt * num_samples + expected_sample]
            if not isinstance(response, GeneratedResponse):
                raise ValueError("Sampler must return GeneratedResponse objects.")
            if (response.prompt_index, response.sample_index) != (expected_prompt, expected_sample):
                raise ValueError("Sampler responses must be prompt-major and sample-minor.")
            pair = (response.prompt_index, response.sample_index)
            if pair in seen:
                raise ValueError("Sampler returned a duplicate prompt/sample pair.")
            seen.add(pair)
            if response.prompt_tokens < 0 or response.response_tokens < 0:
                raise ValueError("Sampler token counts must be non-negative.")
            if response.finish_reason not in {"eos", "length"}:
                raise ValueError("Sampler finish_reason must be eos or length.")
    return responses


def _input_row(example: DatasetExample, config: RunConfig) -> dict[str, Any]:
    return {
        "prompt_id": example.prompt_id, "source_index": example.source_index,
        "question": example.question, "ground_truth": example.ground_truth,
        "rendered_prompt": example.prompt, "dataset_name": example.dataset_name,
        "conversation_messages": (
            [message.to_dict() for message in example.messages]
            if example.messages is not None else None
        ),
        "dataset_config": config.dataset.config, "split": example.split,
        "revision": config.dataset.revision,
    }


def _rendered_prompt_row(example: DatasetExample, sampler: Any) -> dict[str, Any]:
    prompt_format = sampler._resolve_prompt_format()
    return {
        "prompt_id": example.prompt_id,
        "source_index": example.source_index,
        "prompt_format": prompt_format,
        "add_generation_prompt": prompt_format == "chat",
        "dataset_prompt": example.prompt,
        "conversation_messages": (
            [message.to_dict() for message in example.messages]
            if example.messages is not None else None
        ),
        "model_input_prompt": sampler.render_prompt(_prompt_input(example)),
    }


def _prompt_input(example: DatasetExample) -> str | list[dict[str, str]]:
    if example.messages is None:
        return example.prompt
    return [message.to_dict() for message in example.messages]


def _resolved_config(config: RunConfig, output_dir: Path) -> dict[str, Any]:
    resolved = _serialize_value(config)
    resolved["output"]["output_dir"] = str(output_dir)
    for key in ("name", "tokenizer_name"):
        value = resolved["model"].get(key)
        if isinstance(value, str) and Path(value).expanduser().exists():
            resolved["model"][key] = str(Path(value).expanduser().resolve())
    return resolved


def _build_provenance(config: RunConfig, sampler: Any, dataset_result: Any, examples: list[DatasetExample]) -> dict[str, Any]:
    model = sampler._model
    tokenizer = sampler._tokenizer
    chat_template = getattr(tokenizer, "chat_template", None)
    return {
        "rewardscope_version": _package_version("rewardscope"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "model": {
            "requested_name": config.model.name,
            "resolved_path": _local_path_or_none(config.model.name),
            "tokenizer_requested_name": config.model.tokenizer_name or config.model.name,
            "model_class": type(model).__name__, "tokenizer_class": type(tokenizer).__name__,
            "dtype": str(getattr(model, "dtype", None)),
            "prompt_format": sampler._resolve_prompt_format(),
            "padding_side": getattr(tokenizer, "padding_side", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "eos_token_id": _json_value(sampler._eos_token_id()),
            "chat_template_sha256": _sha256_text(chat_template) if isinstance(chat_template, str) else None,
            "files": _model_file_manifest(config.model.name),
        },
        "dataset": {
            "name": config.dataset.name, "config": config.dataset.config,
            "split": config.dataset.split, "revision": config.dataset.revision,
            "selection": config.dataset.selection, "dataset_seed": config.dataset.dataset_seed,
            "levels": list(config.dataset.levels) if config.dataset.levels else None,
            "hf_endpoint": config.dataset.hf_endpoint,
            "data_source": config.dataset.data_source,
            "requested_source_indices": list(config.dataset.source_indices) if config.dataset.source_indices else None,
            "source_count": dataset_result.source_count, "selected_count": len(examples),
            "selected_source_indices": [example.source_index for example in examples],
            "selected_prompt_ids": [example.prompt_id for example in examples],
            "fingerprint": dataset_result.fingerprint,
            "gold_parse_attempt_count": dataset_result.gold_parse_attempt_count,
            "gold_parse_failure_count": dataset_result.gold_parse_failure_count,
            "gold_parse_failure_rate": dataset_result.gold_parse_failure_rate,
        },
        "generation": {"generation_seed": config.sampling.generation_seed, "kwargs": build_generation_kwargs(config.sampling)},
        "verification": {
            "backend": config.verification.backend,
            "mode": config.verification.mode,
            "math_verify_version": _package_version("math-verify"),
            "percentage_policy": _extraction_config_for_dataset(config).percentage_policy,
            "gold_parser": "latex" if config.verification.backend == "math_verify_latex" else "expression",
        },
        "runtime": _runtime_provenance(),
    }


def _build_manifest(directory: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            artifacts.append({"path": path.relative_to(directory).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    return {"status": "completed", "artifacts": artifacts}


def _preflight_plotting() -> None:
    from rewardscope.reports.plots import _load_pyplot
    _load_pyplot()


def _runtime_provenance() -> dict[str, Any]:
    runtime: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform(), "pytorch": None, "transformers": _package_version("transformers"), "datasets": _package_version("datasets")}
    try:
        import torch
        runtime["pytorch"] = torch.__version__
        runtime["cuda_version"] = torch.version.cuda
        runtime["cuda_available"] = torch.cuda.is_available()
        runtime["gpus"] = [
            {"name": torch.cuda.get_device_name(index), "capability": list(torch.cuda.get_device_capability(index))}
            for index in range(torch.cuda.device_count())
        ] if torch.cuda.is_available() else []
    except ModuleNotFoundError:
        runtime["cuda_available"] = False
        runtime["gpus"] = []
    return runtime


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if is_dataclass(value): return {key: _serialize_value(item) for key, item in asdict(value).items()}
    if isinstance(value, tuple): return [_serialize_value(item) for item in value]
    if isinstance(value, dict): return {key: _serialize_value(item) for key, item in value.items()}
    return value


def _json_value(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


def _package_version(name: str) -> str | None:
    try: return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: return None


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError): return None


def _local_path_or_none(value: str) -> str | None:
    return str(Path(value).expanduser().resolve()) if Path(value).expanduser().exists() else None


def _model_file_manifest(model_name: str) -> list[dict[str, Any]]:
    root = Path(model_name).expanduser()
    if not root.is_dir(): return []
    small_metadata = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
    return [
        {"path": path.name, "bytes": path.stat().st_size,
         "sha256": _sha256_file(path) if path.name in small_metadata else None}
        for path in sorted(root.iterdir()) if path.is_file()
    ]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relocate_report(report: AnalysisArtifacts, directory: Path) -> AnalysisArtifacts:
    return AnalysisArtifacts(directory / "prompt_group_metrics.csv", directory / "summary.json", directory / "issues.jsonl", report.group_count, report.issue_count)


def _relocate_plots(plots: Any | None, directory: Path) -> Any | None:
    if plots is None: return None
    return type(plots)(*(directory / path.name if path is not None else None for path in plots.__dict__.values()))


def _extraction_config_for_dataset(config: RunConfig) -> NumericExtractionConfig:
    return NumericExtractionConfig(
        percentage_policy="literal" if config.dataset.name.lower() == "gsm8k" else "reject"
    )


def _build_rollout(config: RunConfig, rollout_input: RolloutInput):
    if config.verification.backend == "math_verify_latex":
        return build_math_verify_latex_rollout(
            rollout_input,
            reward_config=config.reward,
            mode=config.verification.mode,
        )
    if config.verification.backend == "math_verify":
        return build_math_verify_numeric_rollout(
            rollout_input,
            reward_config=config.reward,
            mode=config.verification.mode,
        )
    return build_numeric_rollout(
        rollout_input,
        reward_config=config.reward,
        extraction_config=_extraction_config_for_dataset(config),
    )
