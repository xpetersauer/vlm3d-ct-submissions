#!/usr/bin/env python3
"""Prepare, upload, and document the three exact CT weight bundles with gh.

Preparation is local only. Upload creates a DRAFT release in the chosen repo.
Finalize refuses private repositories/draft releases and checks uploaded assets.
Neither phase changes visibility, publishes a release, commits, or pushes Git.
"""
import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / '.release_plan.json'
WEIGHTS = [
    ('task1', 'cd1082ec', 'task1_weights.zip', '5f252b1b2da96516303762fde1ec28af428849edb83afb7150435562b71d2d27'),
    ('task2', 'a8799c78', 'task2_weights.zip', 'ab8417b0e0b7395988ecdd748ad1fc543fc6f1d9866a420b95150d28bf793ebc'),
    ('task3', '888a2309', 'task3_weights_full_47149.zip', 'b06dea5916646cda857f815d595e609523384566fcd602cbc34ad3decf13722a'),
]
CHUNK = 1900 * 1024 * 1024  # smaller than GitHub's 2 GiB release-asset ceiling
MAX_ASSET = 2 * 1024**3

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def gh(*args):
    return subprocess.check_output(['gh', *args], text=True)

def save(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')

def valid_repo(value):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', value):
        raise ValueError('Use OWNER/REPOSITORY, not a URL')
    return value

def prepare(args):
    if PLAN.exists():
        raise ValueError('A release plan already exists. Use upload/finalize; preparation will not overwrite it.')
    repo = valid_repo(args.repo)
    weights_dir = args.weights_dir.expanduser().resolve()
    sources = []
    for task, sid, name, expected in WEIGHTS:
        path = weights_dir / name
        if not path.is_file() or path.is_symlink(): raise ValueError('Missing/unsafe weights: ' + str(path))
        print('Verifying exact weights:', name, flush=True)
        before = path.stat()
        if sha(path) != expected: raise ValueError('Weight SHA-256 mismatch: ' + name)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns): raise ValueError('Weights changed during hashing')
        sources.append((task, sid, path, expected, after))
    staging_parent = args.staging_parent.expanduser().resolve()
    if not staging_parent.is_dir(): raise ValueError('Staging parent directory does not exist')
    staging = Path(tempfile.mkdtemp(prefix='vlm3d_ct_release_', dir=staging_parent))
    assets = []
    records = []
    for task, sid, path, expected, stats in sources:
        names = []
        if path.stat().st_size >= MAX_ASSET:
            print('Splitting in a new staging directory:', path.name, flush=True)
            whole = hashlib.sha256()
            with path.open('rb') as f:
                index = 0
                while True:
                    first = f.read(min(8 * 1024 * 1024, CHUNK))
                    if not first: break
                    target = staging / (path.name + f'.part{index:03d}')
                    h = hashlib.sha256(); used = 0
                    with target.open('xb') as out:
                        block = first
                        while block:
                            out.write(block); h.update(block); whole.update(block); used += len(block)
                            if used >= CHUNK: break
                            block = f.read(min(8 * 1024 * 1024, CHUNK-used))
                    assets.append(dict(path=str(target), name=target.name, bytes=used, sha256=h.hexdigest()))
                    names.append(target.name); index += 1
            if whole.hexdigest() != expected: raise ValueError('Split-file verification failed; original left unchanged')
        else:
            assets.append(dict(path=str(path), name=path.name, bytes=stats.st_size, sha256=expected))
            names.append(path.name)
        records.append(dict(task=task, submission=sid, original_filename=path.name, original_sha256=expected, original_bytes=stats.st_size, assets=names))
    sums = staging / 'SHA256SUMS.txt'
    sums.write_text(''.join(a['sha256'] + '  ' + a['name'] + '\n' for a in assets))
    assets.append(dict(path=str(sums), name=sums.name, bytes=sums.stat().st_size, sha256=sha(sums)))
    plan = dict(schema='vlm3d-github-release-v1', repo=repo, tag=args.tag, staging=str(staging), assets=assets, weights=records)
    save(PLAN, plan)
    print('Preparation complete. Originals unchanged. Task2 parts use about 2.9 GB extra disk.')
    print('Next: python3 scripts/github_release.py upload')

def read_plan():
    if not PLAN.is_file(): raise ValueError('Run prepare first')
    plan = json.loads(PLAN.read_text()); valid_repo(plan['repo'])
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', plan['tag']): raise ValueError('Invalid release tag')
    return plan

def release(plan, create=False):
    result = subprocess.run(['gh', 'api', f"repos/{plan['repo']}/releases/tags/{plan['tag']}"], text=True, capture_output=True)
    if result.returncode == 0: return json.loads(result.stdout)
    if not create or '404' not in result.stderr: raise ValueError('Cannot inspect release: ' + result.stderr.strip())
    notes = ROOT / 'docs/RELEASE_NOTES.md'
    print(gh('release', 'create', plan['tag'], '--repo', plan['repo'], '--draft', '--title', 'VLM3D CT submitted weights', '--notes-file', str(notes)))
    return json.loads(gh('api', f"repos/{plan['repo']}/releases/tags/{plan['tag']}"))

def matching(asset, expected):
    if asset.get('size') != expected['bytes']: return False
    digest = asset.get('digest')
    return digest is None or digest == 'sha256:' + expected['sha256']

def upload():
    plan = read_plan()
    obj = release(plan, create=True)
    if not obj.get('draft'): raise ValueError('Release is already published; refusing to mutate it')
    existing = {a['name']: a for a in obj.get('assets', [])}
    for asset in plan['assets']:
        path = Path(asset['path'])
        if not path.is_file() or sha(path) != asset['sha256']: raise ValueError('Local asset changed/missing: ' + asset['name'])
        if asset['name'] in existing:
            found = existing[asset['name']]
            if not matching(found, asset) or not found.get('digest'):
                raise ValueError('Existing remote asset cannot be hash-verified; no overwrite performed: ' + asset['name'])
            print('Verified existing upload:', asset['name']); continue
        print('Uploading DRAFT release asset:', asset['name'], flush=True)
        subprocess.run(['gh', 'release', 'upload', plan['tag'], str(path), '--repo', plan['repo']], check=True)
    print('Draft assets uploaded. Nothing was made public by this script.')
    print('See UPLOAD_TO_GITHUB.md for public visibility/release publication, then run finalize.')

def finalize():
    plan = read_plan()
    repo = json.loads(gh('api', 'repos/' + plan['repo']))
    if repo.get('private') is not False: raise ValueError('Repository is private; do not claim public availability yet')
    obj = release(plan)
    if obj.get('draft'): raise ValueError('Release is draft; publish it before finalizing public links')
    found = {a['name']: a for a in obj.get('assets', [])}
    for expected in plan['assets']:
        if expected['name'] not in found or not matching(found[expected['name']], expected):
            raise ValueError('Remote asset missing/size or digest mismatch: ' + expected['name'])
    base = f"https://github.com/{plan['repo']}/releases/download/{plan['tag']}/"
    text = '# Final submission weights\n\nPublic release: ' + obj['html_url'] + '\n\n'
    links = {}
    for record in plan['weights']:
        text += '## ' + record['task'] + ' — ' + record['submission'] + '\n\n'
        for name in record['assets']: text += '- [' + name + '](' + base + name + ')\n'
        text += '\nReconstructed/original filename: `' + record['original_filename'] + '`\n\nSHA-256: `' + record['original_sha256'] + '`\n\n'
        links[record['task']] = base + record['assets'][0] if len(record['assets']) == 1 else obj['html_url']
    text += '[Asset checksums](' + base + 'SHA256SUMS.txt)\n\n'
    text += '## Task2 reassembly\n\nDownload both numbered parts into one directory, then run:\n\n```bash\ncat task2_weights.zip.part000 task2_weights.zip.part001 > task2_weights.zip\nshasum -a 256 task2_weights.zip\n```\n\nCompare the reconstructed hash with the Task2 value above before extraction.\n'
    (ROOT / 'docs/WEIGHT_DOWNLOADS.md').write_text(text)
    form = (ROOT / 'CHALLENGE_FORM.md').read_text()
    form = form.replace('**Current selection:** Not available / access restricted (until an authorized public URL exists).', '**Current selection:** GitHub — public release assets (see docs/WEIGHT_DOWNLOADS.md).')
    parts = form.split('## Task ')
    for i in range(1, len(parts)):
        key = 'task' + parts[i][0]
        if key in links:
            parts[i] = parts[i].replace('Public URL: not yet supplied.', 'Public URL: ' + links[key] + ' .')
    form = '## Task '.join(parts)
    form = form.replace('The existing weights remain on the participant\'s Mac; public download URLs have\nnot been supplied. Keep those availability statements accurate when submitting.', 'The original weights remain on the participant\'s Mac. Verified public release assets and reassembly instructions are documented in docs/WEIGHT_DOWNLOADS.md.')
    form = form.split('\n\n## GitHub repository and release')[0]
    form += '\n\n## GitHub repository and release\n\nRepository: https://github.com/' + plan['repo'] + '\n\nWeights: ' + obj['html_url'] + '\n'
    (ROOT / 'CHALLENGE_FORM.md').write_text(form)
    readme = (ROOT / 'README.md').read_text()
    readme = readme.replace('Authorized public weight hosting is still required for a fully\nreproducible public release.', 'Published weight links and Task2 reassembly instructions are in docs/WEIGHT_DOWNLOADS.md. Remaining training/provenance gaps still limit full reproducibility.')
    (ROOT / 'README.md').write_text(readme)
    manifest_path = ROOT / 'provenance/ARTIFACTS.json'
    manifest = json.loads(manifest_path.read_text())
    for s in manifest['submissions']: s['weight_url'] = links.get('task' + str(s['task']))
    save(manifest_path, manifest)
    print('Updated CHALLENGE_FORM.md, README.md, docs/WEIGHT_DOWNLOADS.md and provenance/ARTIFACTS.json.')
    print('Remote names/sizes checked; server SHA-256 checked when supplied by API. No remote asset was changed.')
    print('Commit/push those local documentation changes next.')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    a = sub.add_parser('prepare')
    a.add_argument('--repo', required=True)
    a.add_argument('--tag', default='weights-v1')
    a.add_argument('--weights-dir', type=Path, default=Path.home() / 'Downloads/vlm3d_final/vlm3d_weeklong_deliverables')
    a.add_argument('--staging-parent', type=Path, default=Path.home() / 'Downloads')
    sub.add_parser('upload'); sub.add_parser('finalize')
    args = p.parse_args()
    if args.mode == 'prepare' and not re.fullmatch(r'[A-Za-z0-9_.-]+', args.tag): p.error('Invalid tag')
    try:
        if args.mode == 'prepare': prepare(args)
        elif args.mode == 'upload': upload()
        else: finalize()
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as e:
        print('ERROR:', e); return 2

if __name__ == '__main__': raise SystemExit(main())
