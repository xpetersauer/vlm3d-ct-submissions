#!/usr/bin/env python3
"""Verify a local weight archive against the participant's collected SHA-256."""
import argparse
import hashlib
from pathlib import Path

HASHES = {
    'task1': '5f252b1b2da96516303762fde1ec28af428849edb83afb7150435562b71d2d27',
    'task2': 'ab8417b0e0b7395988ecdd748ad1fc543fc6f1d9866a420b95150d28bf793ebc',
    'task3': 'b06dea5916646cda857f815d595e609523384566fcd602cbc34ad3decf13722a',
}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=HASHES, required=True)
    p.add_argument('--file', type=Path, required=True)
    a = p.parse_args()
    h = hashlib.sha256()
    with a.file.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    print(h.hexdigest(), a.file.name)
    if h.hexdigest() != HASHES[a.task]:
        print('MISMATCH: do not silently substitute this file.'); return 1
    print('MATCH: local bytes equal collected archive; platform identity/rights are separate checks.')
    return 0

if __name__ == '__main__': raise SystemExit(main())
