#!/usr/bin/env python3
"""
Migrate stray MLflow experiment/run checkpoint directories that were
accidentally written at the repo root (e.g. ./560573378411910460/<run_id>/checkpoints/)
into their proper location under mlruns/<experiment_id>/<run_id>/checkpoints/.

This mirrors the path convention already used by
cardiac_motion/utils/mlflow_read_helpers.py (get_all_ckpt_paths), which looks
for checkpoints at os.path.join(os.path.dirname(artifact_uri), "checkpoints"),
i.e. a sibling of the run's "artifacts" directory inside mlruns/.

Usage:
    python scripts/migrate_mlruns_checkpoints.py            # dry run (default), just prints the plan
    python scripts/migrate_mlruns_checkpoints.py --apply     # actually move the files
    python scripts/migrate_mlruns_checkpoints.py --apply --force  # overwrite files that already exist at the destination

Safety:
- Only touches top-level directories whose name matches an MLflow experiment id
  (digits only) AND that already exist as a directory under mlruns/.
- Only touches run subdirectories whose name already exists under
  mlruns/<experiment_id>/.
- Refuses to overwrite an existing destination file unless --force is given.
- Removes source run/experiment directories only after they are empty.
"""
import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MLRUNS_DIR = REPO_ROOT / "mlruns"


def find_candidate_experiments(repo_root: Path, mlruns_dir: Path):
    for entry in sorted(repo_root.iterdir()):
        if not entry.is_dir():
            continue
        if not entry.name.isdigit():
            continue
        if entry == mlruns_dir:
            continue
        target_exp_dir = mlruns_dir / entry.name
        if not target_exp_dir.is_dir():
            print(f"[skip] {entry.name}: no matching experiment directory under mlruns/, leaving untouched")
            continue
        yield entry, target_exp_dir


def plan_moves(exp_dir: Path, target_exp_dir: Path):
    moves = []
    for run_dir in sorted(exp_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        target_run_dir = target_exp_dir / run_id
        if not target_run_dir.is_dir():
            print(f"[skip] {exp_dir.name}/{run_id}: no matching run directory under mlruns/{exp_dir.name}/, leaving untouched")
            continue

        src_ckpt_dir = run_dir / "checkpoints"
        if not src_ckpt_dir.is_dir():
            print(f"[skip] {exp_dir.name}/{run_id}: no checkpoints/ subdirectory found")
            continue

        dst_ckpt_dir = target_run_dir / "checkpoints"
        for src_file in sorted(src_ckpt_dir.iterdir()):
            dst_file = dst_ckpt_dir / src_file.name
            moves.append((src_file, dst_file))
    return moves


def apply_moves(moves, force: bool):
    for src_file, dst_file in moves:
        if dst_file.exists() and not force:
            print(f"[error] destination already exists, skipping (use --force to overwrite): {dst_file}")
            continue
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_file), str(dst_file))
        print(f"[moved] {src_file} -> {dst_file}")


def cleanup_empty_dirs(exp_dir: Path):
    for run_dir in sorted(exp_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        ckpt_dir = run_dir / "checkpoints"
        if ckpt_dir.is_dir() and not any(ckpt_dir.iterdir()):
            ckpt_dir.rmdir()
            print(f"[cleanup] removed empty {ckpt_dir}")
        if run_dir.is_dir() and not any(run_dir.iterdir()):
            run_dir.rmdir()
            print(f"[cleanup] removed empty {run_dir}")
    if exp_dir.is_dir() and not any(exp_dir.iterdir()):
        exp_dir.rmdir()
        print(f"[cleanup] removed empty {exp_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Actually perform the moves (default is dry run)")
    parser.add_argument("--force", action="store_true", help="Overwrite destination files that already exist")
    args = parser.parse_args()

    if not MLRUNS_DIR.is_dir():
        raise SystemExit(f"mlruns/ directory not found at {MLRUNS_DIR}")

    all_moves = []
    all_exp_dirs = []
    for exp_dir, target_exp_dir in find_candidate_experiments(REPO_ROOT, MLRUNS_DIR):
        moves = plan_moves(exp_dir, target_exp_dir)
        all_moves.extend(moves)
        all_exp_dirs.append(exp_dir)

    if not all_moves:
        print("Nothing to migrate.")
        return

    print(f"\n{len(all_moves)} file(s) to move:")
    for src_file, dst_file in all_moves:
        print(f"  {src_file.relative_to(REPO_ROOT)} -> {dst_file.relative_to(REPO_ROOT)}")

    if not args.apply:
        print("\nDry run only, no changes made. Re-run with --apply to perform the migration.")
        return

    print()
    apply_moves(all_moves, force=args.force)

    for exp_dir in all_exp_dirs:
        cleanup_empty_dirs(exp_dir)


if __name__ == "__main__":
    main()
