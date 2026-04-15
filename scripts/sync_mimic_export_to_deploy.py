#!/usr/bin/env python3
"""Sync a trained export (ONNX + deploy.yaml) into deploy policy folder.

Example:
  python scripts/sync_mimic_export_to_deploy.py \
      unitree_g1_23dof_mimic_dance_102/2026-03-25_10-19-47
    python scripts/sync_mimic_export_to_deploy.py \
            unitree_g1_23dof_velocity/2026-03-28_14-44-45
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

FIXED_JOINT_IDS_MAP = (
    "joint_ids_map: [0, 6, 12, 1, 7, 15, 22, 2, 8, 16, 23, 3, 9, 17, 24, 4, 10, "
    "18, 25, 5, 11, 19, 26]"
)


def infer_target_subpath(experiment_name: str, robot: str, target_dance_name: str | None) -> tuple[Path, str]:
    mimic_prefix = f"unitree_{robot}_mimic_"
    velocity_name = f"unitree_{robot}_velocity"

    if experiment_name.startswith(mimic_prefix):
        dance_name = target_dance_name or experiment_name[len(mimic_prefix) :]
        if not dance_name:
            raise ValueError("Inferred empty dance name from mimic experiment.")
        return Path("mimic") / dance_name, f"mimic/{dance_name}"

    if experiment_name == velocity_name:
        return Path("velocity") / "v0", "velocity/v0"

    raise ValueError(
        "Cannot infer destination from experiment name "
        f"'{experiment_name}'. Expected '{mimic_prefix}<dance_name>' or '{velocity_name}'."
    )


def confirm_overwrite(paths: list[Path]) -> bool:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return True

    print("[WARN] Destination file(s) already exist:")
    for p in existing:
        print(f"  - {p}")

    while True:
        answer = input("Replace existing file(s)? [y/n]: ").strip().lower()
        if answer == "y":
            return True
        if answer == "n":
            return False
        print("Please input 'y' or 'n'.")


def replace_joint_ids_map_in_yaml(yaml_path: Path) -> None:
    text = yaml_path.read_text(encoding="utf-8")

    # Replace full joint_ids_map block (supports one-line or wrapped multi-line YAML list).
    pattern = re.compile(r"(?ms)^joint_ids_map:\s*\[.*?\]\s*\n(?=\w+:)")
    if pattern.search(text):
        text = pattern.sub(FIXED_JOINT_IDS_MAP + "\n", text, count=1)
    else:
        # Fallback: ensure the first line is the fixed map.
        text = FIXED_JOINT_IDS_MAP + "\n" + text

    yaml_path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy policy.onnx + deploy.yaml from logs/rsl_rl to deploy folder and patch joint_ids_map."
    )
    parser.add_argument(
        "run_path",
        help=(
            "Relative run path under logs/rsl_rl, e.g. "
            "unitree_g1_23dof_mimic_dance_102/2026-03-25_10-19-47 or "
            "unitree_g1_23dof_velocity/2026-03-28_14-44-45"
        ),
    )
    parser.add_argument(
        "--robot",
        default="g1_23dof",
        help="Robot key used in run prefix and deploy destination (default: g1_23dof)",
    )
    parser.add_argument(
        "--target-dance-name",
        default=None,
        help="Optional override for destination dance folder name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing files.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    logs_root = project_root / "logs" / "rsl_rl"

    run_rel = Path(args.run_path)
    if len(run_rel.parts) < 2:
        raise ValueError("run_path must include experiment/timestamp, e.g. exp_name/2026-xx-xx_xx-xx-xx")

    exp_name = run_rel.parts[0]
    run_dir = logs_root / run_rel

    target_subpath, target_label = infer_target_subpath(exp_name, args.robot, args.target_dance_name)

    src_policy = run_dir / "exported" / "policy.onnx"
    src_deploy_yaml = run_dir / "params" / "deploy.yaml"

    dst_root = project_root / "deploy" / "robots" / args.robot / "config" / "policy" / target_subpath
    dst_policy = dst_root / "exported" / "policy.onnx"
    dst_deploy_yaml = dst_root / "params" / "deploy.yaml"

    missing = [p for p in (src_policy, src_deploy_yaml) if not p.exists()]
    if missing:
        for p in missing:
            print(f"[ERROR] Missing source file: {p}")
        return 1

    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Source run:   {run_dir}")
    print(f"[INFO] Target type:  {target_label}")
    print(f"[INFO] Dest root:    {dst_root}")

    if args.dry_run:
        print(f"[DRY-RUN] Copy {src_policy} -> {dst_policy}")
        print(f"[DRY-RUN] Copy {src_deploy_yaml} -> {dst_deploy_yaml}")
        print(f"[DRY-RUN] Patch joint_ids_map in {dst_deploy_yaml}")
        return 0

    if not confirm_overwrite([dst_policy, dst_deploy_yaml]):
        print("[INFO] User chose not to replace existing files. Exit.")
        return 0

    dst_policy.parent.mkdir(parents=True, exist_ok=True)
    dst_deploy_yaml.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src_policy, dst_policy)
    shutil.copy2(src_deploy_yaml, dst_deploy_yaml)
    replace_joint_ids_map_in_yaml(dst_deploy_yaml)

    print(f"[DONE] policy.onnx synced to: {dst_policy}")
    print(f"[DONE] deploy.yaml synced to: {dst_deploy_yaml}")
    print(f"[DONE] joint_ids_map replaced with fixed MuJoCo mapping.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise
