"""
Day049-Day050: Custom MLflow pyfunc model packaging (`ScratchLmPyfuncAdapter`),
immutable package manifest with SHA-256 verification, clean-process reload,
and MLflow Model Registry candidate registration with retry reconciliation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote

import httpx
import torch

from scratchlm.config import ModelConfig, load_model_config
from scratchlm.inference.sampling import generate_tokens
from scratchlm.model.gpt import ScratchGPT
from scratchlm.tokenizer.bpe_tokenizer import BPETokenizer
from scratchlm.training.checkpoints import LineageHashes
from scratchlm.training.sft import ChatTemplateSpec
from scratchlm.training.tracker import SpoolingMlflowRestTracker, validate_logical_artifact_uri


@dataclass(frozen=True)
class ModelPackageManifest:
    """Immutable manifest for a packaged ScratchGPT model artifact (Day049)."""

    schema_version: str
    model_name: str
    model_stage: str  # "base_pretrain" or "narrow_sft"
    source_run_id: str
    parent_base_run_id: str | None
    lineage: LineageHashes
    files_sha256: dict[str, str]
    package_sha256: str
    validation_gate_passed: bool
    serving_promotion_approved: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True)
class RegistryCandidateRecord:
    """Persisted reconciliation result for an MLflow Model Registry candidate."""

    model_name: str
    selected_version: str
    source_run_id: str
    artifact_logical_uri: str
    package_sha256: str
    tokenizer_sha256: str
    candidate_alias_assigned: bool
    champion_alias_assigned: bool
    labeled_duplicate_versions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_mlflow_pyfunc_package(
    *,
    output_dir: Path,
    model_name: str,
    model_stage: str,
    source_run_id: str,
    parent_base_run_id: str | None,
    model: ScratchGPT,
    tokenizer: BPETokenizer,
    chat_template: ChatTemplateSpec,
    lineage: LineageHashes,
    validation_gate_passed: bool,
) -> ModelPackageManifest:
    """
    Package `ScratchGPT` weights, `model_config.json`, `tokenizer.json`, `chat_template.json`,
    `requirements.txt`, `MLmodel` descriptor, and `package_manifest.json` into `output_dir`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save pure state_dict tensor weights (safe for weights_only=True loading)
    weights_path = output_dir / "weights.pt"
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(cpu_state, weights_path)

    # 2. Save model_config.json
    cfg_path = output_dir / "model_config.json"
    cfg_path.write_text(json.dumps(asdict(model.config), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # 3. Save tokenizer.json
    tok_path = output_dir / "tokenizer.json"
    tokenizer.save(tok_path)

    # 4. Save chat_template.json
    tmpl_path = output_dir / "chat_template.json"
    tmpl_path.write_text(json.dumps(chat_template.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # 5. Save pinned requirements.txt
    req_path = output_dir / "requirements.txt"
    req_path.write_text(
        "torch==2.6.0+cu118\nnumpy==1.26.4\npydantic==2.9.2\n",
        encoding="utf-8",
    )

    # 6. Save MLflow pyfunc MLmodel metadata descriptor
    mlmodel_path = output_dir / "MLmodel"
    mlmodel_content = (
        "artifact_path: model_package\n"
        "flavors:\n"
        "  python_function:\n"
        "    loader_module: scratchlm.serving.package\n"
        "    data: .\n"
        "    env:\n"
        "      virtualenv: python_env.yaml\n"
        "    python_version: 3.12.0\n"
        f"run_id: {source_run_id}\n"
    )
    mlmodel_path.write_text(mlmodel_content, encoding="utf-8")

    tracked_files = [
        "weights.pt",
        "model_config.json",
        "tokenizer.json",
        "chat_template.json",
        "requirements.txt",
        "MLmodel",
    ]
    files_sha: dict[str, str] = {}
    combo_hasher = hashlib.sha256()
    for fname in sorted(tracked_files):
        raw = (output_dir / fname).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        files_sha[fname] = sha
        combo_hasher.update(f"{fname}:{sha}\n".encode("utf-8"))

    pkg_sha = combo_hasher.hexdigest()
    manifest = ModelPackageManifest(
        schema_version="1.0.0",
        model_name=model_name,
        model_stage=model_stage,
        source_run_id=source_run_id,
        parent_base_run_id=parent_base_run_id,
        lineage=lineage,
        files_sha256=files_sha,
        package_sha256=pkg_sha,
        validation_gate_passed=validation_gate_passed,
        serving_promotion_approved=False,
    )
    manifest_path = output_dir / "package_manifest.json"
    manifest_path.write_text(manifest.canonical_json(), encoding="utf-8")
    return manifest


class ScratchLmPyfuncAdapter:
    """
    Reloadable MLflow pyfunc-compatible inference adapter (Day049).
    Verifies every file checksum in `package_manifest.json` and loads weights using
    restricted `weights_only=True` deserialization so untrusted pickles cannot execute code.
    """

    def __init__(
        self,
        *,
        model: ScratchGPT,
        tokenizer: BPETokenizer,
        chat_template: ChatTemplateSpec,
        manifest: ModelPackageManifest,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.manifest = manifest

    @classmethod
    def load_package(cls, package_dir: Path, device: str = "cpu") -> ScratchLmPyfuncAdapter:
        manifest_path = package_dir / "package_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing package_manifest.json in {package_dir}")

        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files_sha: dict[str, str] = raw_manifest["files_sha256"]

        combo_hasher = hashlib.sha256()
        for fname in sorted(files_sha.keys()):
            fpath = package_dir / fname
            if not fpath.exists():
                raise FileNotFoundError(f"Package file '{fname}' missing in {package_dir}")
            actual_sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
            if actual_sha != files_sha[fname]:
                raise ValueError(
                    f"Package file checksum mismatch for '{fname}': expected {files_sha[fname]}, got {actual_sha}"
                )
            combo_hasher.update(f"{fname}:{actual_sha}\n".encode("utf-8"))

        if combo_hasher.hexdigest() != raw_manifest["package_sha256"]:
            raise ValueError("Overall package_sha256 mismatch")

        lineage = LineageHashes(**raw_manifest["lineage"])
        manifest = ModelPackageManifest(
            schema_version=raw_manifest["schema_version"],
            model_name=raw_manifest["model_name"],
            model_stage=raw_manifest["model_stage"],
            source_run_id=raw_manifest["source_run_id"],
            parent_base_run_id=raw_manifest.get("parent_base_run_id"),
            lineage=lineage,
            files_sha256=files_sha,
            package_sha256=raw_manifest["package_sha256"],
            validation_gate_passed=bool(raw_manifest["validation_gate_passed"]),
            serving_promotion_approved=bool(raw_manifest["serving_promotion_approved"]),
        )

        cfg: ModelConfig = load_model_config(package_dir / "model_config.json")
        tok = BPETokenizer.load(package_dir / "tokenizer.json", expected_sha256=lineage.tokenizer_sha256)
        tmpl = ChatTemplateSpec(**json.loads((package_dir / "chat_template.json").read_text(encoding="utf-8")))

        model = ScratchGPT(cfg)
        state_dict = torch.load(package_dir / "weights.pt", map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        if cfg.tie_word_embeddings:
            model.lm_head.weight = model.embeddings.token_embedding.weight
        model.to(device)
        model.eval()

        return cls(model=model, tokenizer=tok, chat_template=tmpl, manifest=manifest)

    @torch.no_grad()
    def predict(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int = 16,
        temperature: float = 0.0,
        seed: int = 1337,
    ) -> list[dict[str, Any]]:
        """Batch-generation smoke contract for MLflow pyfunc evaluation."""
        outputs: list[dict[str, Any]] = []
        for prompt in prompts:
            prompt_ids = self.tokenizer.encode(prompt, add_bos=False, add_eos=False)
            res = generate_tokens(
                self.model,
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=seed,
                eos_token_id=self.tokenizer.eos_token_id,
                tokenizer=self.tokenizer,  # type: ignore[arg-type]
            )
            outputs.append(
                {
                    "prompt": prompt,
                    "generated_text": res.output_text,
                    "finish_reason": res.finish_reason,
                    "package_sha256": self.manifest.package_sha256,
                    "tokenizer_sha256": self.manifest.lineage.tokenizer_sha256,
                }
            )
        return outputs


def _load_model(path: str | Path) -> ScratchLmPyfuncAdapter:
    """MLflow python_function loader_module entrypoint."""
    return ScratchLmPyfuncAdapter.load_package(Path(path))


class MlflowRegistryCandidateManager:
    """
    Day049-Day050: Registers evaluated model packages in the MLflow Model Registry and
    reconciles timeout-induced duplicate versions by `(source_run_id, package_sha256)`.
    Enforces that:
      1. Failed validation (`validation_gate_passed == False`) blocks candidate registration.
      2. Retries reconcile to one canonical version and label any duplicates.
      3. Only the `candidate` alias is set; active serving (`champion`) is NEVER automatically set.
    """

    def __init__(self, tracking_uri: str = "http://127.0.0.1:5001", timeout_seconds: float = 5.0) -> None:
        self.tracking_uri = tracking_uri.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _req(self, method: str, path: str, json_body: dict[str, Any] | None = None) -> httpx.Response:
        url = f"{self.tracking_uri}{path}"
        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.request(method, url, json=json_body)
            resp.raise_for_status()
            return resp

    def ensure_registered_model(self, model_name: str) -> None:
        try:
            self._req("GET", f"/api/2.0/mlflow/registered-models/get?name={quote(model_name)}")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            self._req(
                "POST",
                "/api/2.0/mlflow/registered-models/create",
                json_body={
                    "name": model_name,
                    "description": "ScratchGPT decoder-only language model trained from random weights.",
                },
            )

    def create_raw_version(
        self,
        *,
        model_name: str,
        source_run_id: str,
        artifact_logical_uri: str,
        manifest: ModelPackageManifest,
    ) -> str:
        validate_logical_artifact_uri(artifact_logical_uri)
        tags = [
            {"key": "package_sha256", "value": manifest.package_sha256},
            {"key": "tokenizer_sha256", "value": manifest.lineage.tokenizer_sha256},
            {"key": "dataset_manifest_sha256", "value": manifest.lineage.dataset_manifest_sha256},
            {"key": "validation_gate", "value": "PASSED" if manifest.validation_gate_passed else "FAILED"},
            {"key": "serving_approved", "value": "false"},
        ]
        resp = self._req(
            "POST",
            "/api/2.0/mlflow/model-versions/create",
            json_body={
                "name": model_name,
                "source": artifact_logical_uri,
                "run_id": source_run_id,
                "tags": tags,
            },
        )
        return str(resp.json()["model_version"]["version"])

    def list_versions_for_model(self, model_name: str) -> list[dict[str, Any]]:
        filter_str = quote(f"name='{model_name}'")
        resp = self._req("GET", f"/api/2.0/mlflow/model-versions/search?filter={filter_str}")
        data = resp.json()
        versions: list[dict[str, Any]] = data.get("model_versions", [])
        # Fetch full version details so tags are populated across MLflow 3.x responses
        detailed: list[dict[str, Any]] = []
        for v in versions:
            ver_num = str(v["version"])
            detail_resp = self._req(
                "GET",
                f"/api/2.0/mlflow/model-versions/get?name={quote(model_name)}&version={quote(ver_num)}",
            )
            detailed.append(detail_resp.json()["model_version"])
        return detailed

    def set_version_tag(self, model_name: str, version: str, key: str, value: str) -> None:
        self._req(
            "POST",
            "/api/2.0/mlflow/model-versions/set-tag",
            json_body={"name": model_name, "version": str(version), "key": key, "value": value},
        )

    def register_or_reconcile_candidate(
        self,
        *,
        package_dir: Path,
        manifest: ModelPackageManifest,
        tracker: SpoolingMlflowRestTracker,
    ) -> RegistryCandidateRecord:
        if not manifest.validation_gate_passed:
            raise PermissionError(
                f"Cannot register candidate for '{manifest.model_name}': validation_gate_passed is False"
            )

        # Upload all package files under `model_package/` in MLflow artifacts
        last_uri = ""
        for p in sorted(package_dir.iterdir()):
            if p.is_file():
                last_uri = tracker.log_artifact(p, artifact_subpath="model_package")
        pkg_logical_uri = last_uri.rsplit("/", 1)[0]
        validate_logical_artifact_uri(pkg_logical_uri)

        self.ensure_registered_model(manifest.model_name)
        existing_versions = self.list_versions_for_model(manifest.model_name)

        matching_versions: list[int] = []
        for v in existing_versions:
            if str(v.get("run_id", "")) != manifest.source_run_id:
                continue
            tag_map = {t["key"]: t["value"] for t in v.get("tags", [])}
            existing_hash = tag_map.get("package_sha256")
            if existing_hash and existing_hash != manifest.package_sha256:
                raise ValueError(
                    f"Registry conflict for run_id={manifest.source_run_id}: existing version {v['version']} "
                    f"has package_sha256={existing_hash} != {manifest.package_sha256}"
                )
            if existing_hash == manifest.package_sha256:
                matching_versions.append(int(v["version"]))

        if not matching_versions:
            created_v = self.create_raw_version(
                model_name=manifest.model_name,
                source_run_id=manifest.source_run_id,
                artifact_logical_uri=pkg_logical_uri,
                manifest=manifest,
            )
            matching_versions.append(int(created_v))

        matching_versions.sort()
        canonical_ver = str(matching_versions[0])
        duplicate_vers = [str(v) for v in matching_versions[1:]]

        self.set_version_tag(manifest.model_name, canonical_ver, "reconciliation_status", "selected_canonical")
        for dup_v in duplicate_vers:
            self.set_version_tag(manifest.model_name, dup_v, "reconciliation_status", "duplicate")
            self.set_version_tag(manifest.model_name, dup_v, "duplicate_of_version", canonical_ver)

        # Assign 'candidate' alias ONLY (never 'champion')
        self._req(
            "POST",
            "/api/2.0/mlflow/registered-models/alias",
            json_body={
                "name": manifest.model_name,
                "alias": "candidate",
                "version": canonical_ver,
            },
        )

        return RegistryCandidateRecord(
            model_name=manifest.model_name,
            selected_version=canonical_ver,
            source_run_id=manifest.source_run_id,
            artifact_logical_uri=pkg_logical_uri,
            package_sha256=manifest.package_sha256,
            tokenizer_sha256=manifest.lineage.tokenizer_sha256,
            candidate_alias_assigned=True,
            champion_alias_assigned=False,
            labeled_duplicate_versions=duplicate_vers,
        )
