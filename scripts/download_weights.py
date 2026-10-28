#!/usr/bin/env python3
"""Download, verify and extract VLM3D model weights using Python 3."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BLOCK = 8 * 1024 * 1024


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(BLOCK), b''):
            h.update(block)
    return h.hexdigest()


def fetch(url, target):
    if target.exists():
        print('Using downloaded file:', target.name, flush=True)
        return
    temporary = target.with_name(target.name + '.downloading')
    print('Downloading:', target.name, flush=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'VLM3D-weight-download'})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as out:
            expected = int(response.headers.get('Content-Length', '0'))
            received = 0
            last_report = 0
            while True:
                block = response.read(BLOCK)
                if not block:
                    break
                out.write(block)
                received += len(block)
                if received - last_report >= 128 * 1024 * 1024:
                    print(f'  {received / 1024**2:.0f} MiB' +
                          (f' / {expected / 1024**2:.0f} MiB' if expected else ''), flush=True)
                    last_report = received
            if expected and received != expected:
                raise ValueError('Incomplete download: ' + target.name)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def extract(archive, target):
    if target.exists():
        raise ValueError(f'{target} already exists. Choose a new --output-dir for a fresh extraction.')
    stage = Path(tempfile.mkdtemp(prefix=target.name + '-extract-', dir=target.parent))
    try:
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                name = PurePosixPath(info.filename)
                if name.is_absolute() or '..' in name.parts or '\\' in info.filename:
                    raise ValueError('Unsafe archive path: ' + info.filename)
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Archive symlink is not supported: ' + info.filename)
            z.extractall(stage)
        stage.rename(target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def download(task, destination):
    config = json.loads((ROOT / 'configs' / task / 'weights.json').read_text())
    cache = destination / 'downloads'
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / config['filename']
    if not archive.exists():
        for url, name in zip(config['download_urls'], config['assets']):
            fetch(url, cache / name)
        if len(config['assets']) > 1:
            temporary = archive.with_name(archive.name + '.joining')
            try:
                with temporary.open('wb') as out:
                    for name in config['assets']:
                        with (cache / name).open('rb') as part:
                            shutil.copyfileobj(part, out, BLOCK)
                os.replace(temporary, archive)
            finally:
                temporary.unlink(missing_ok=True)
    print('Verifying:', archive.name, flush=True)
    if archive.stat().st_size != config['bytes'] or sha256(archive) != config['sha256']:
        raise ValueError(f'Checksum/size mismatch: {archive}. Remove that download and its numbered parts before retrying.')
    extract(archive, destination / task)
    print('Ready:', destination / task, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['task1', 'task2', 'task3', 'all'])
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'weights')
    args = parser.parse_args()
    try:
        for task in (['task1', 'task2', 'task3'] if args.task == 'all' else [args.task]):
            download(task, args.output_dir.expanduser().resolve())
        return 0
    except urllib.error.HTTPError as error:
        print(f'Download failed (HTTP {error.code}). Check the public release Assets.', file=sys.stderr)
    except (OSError, ValueError, zipfile.BadZipFile, urllib.error.URLError) as error:
        print('ERROR:', error, file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
