#!/usr/bin/env python3
from __future__ import annotations
import argparse
import io
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', 'data', 'dataset', 'datasets',
                 'weight', 'weights', 'outputs', 'output', 'checkpoints', 'logs', 'releases', 'eval_runs',
                 'node_modules', 'dist', 'results'}
EXTENSIONS = {'.py', '.md', '.toml', '.yaml', '.yml', '.txt', '.sh', '.java', '.png', '.json'}


def release_files(root=ROOT):
    paths = subprocess.check_output(['git', 'ls-files', '-c', '-o', '--exclude-standard', '-z'], cwd=root).decode().split('\0')
    files = []
    for name in sorted(set(paths)):
        if not name:
            continue
        rel = Path(name)
        path = root / rel
        if any(part in EXCLUDED_DIRS or part.startswith('.') for part in rel.parts[:-1]):
            continue
        if rel.name.startswith('.env') and rel.name != '.env.example':
            continue
        if rel.as_posix() == 'config/local.yaml' or not path.is_file() or path.is_symlink():
            continue
        if path.suffix not in EXTENSIONS and rel.name not in {'.env.example', '.gitignore', 'LICENSE', 'NOTICE'}:
            continue
        if path.suffix == '.json' and not name.startswith('synthea_extensions/modules/'):
            continue
        files.append(rel)
    return files


def package(output, root=ROOT):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = release_files(root)
    if not files:
        raise ValueError('No release files found')
    with tarfile.open(output, 'w:gz') as archive:
        for rel in files:
            archive.add(root / rel, arcname=str(Path('ClinVerse') / rel), recursive=False)
        payload = json.dumps({'version': '0.3.0', 'files': [p.as_posix() for p in files]}, indent=2).encode()
        info = tarfile.TarInfo('ClinVerse/release-manifest.json')
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return len(files)


def main():
    parser = argparse.ArgumentParser(description='Package ClinVerse source, configurations and documentation')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(f'{package(args.output)} files: {args.output.resolve()}')


if __name__ == '__main__':
    main()
