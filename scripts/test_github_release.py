"""Offline fixture tests; never calls GitHub or loads neural models."""
import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('release_tool', Path(__file__).with_name('github_release.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class ReleaseTests(unittest.TestCase):
    def test_split_exact_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); weights = root / 'weights'; weights.mkdir()
            w = weights / 'task2_weights.zip'; w.write_bytes(bytes(range(101)))
            digest = m.sha(w)
            args = argparse.Namespace(repo='fixture/project', tag='weights-v1', weights_dir=weights, staging_parent=root)
            with patch.object(m, 'ROOT', root), patch.object(m, 'PLAN', root / '.release_plan.json'), patch.object(m, 'WEIGHTS', [('task2', 'fixture', w.name, digest)]), patch.object(m, 'CHUNK', 64), patch.object(m, 'MAX_ASSET', 100):
                m.prepare(args)
                plan = json.loads(m.PLAN.read_text())
                parts = [Path(x['path']).read_bytes() for x in plan['assets'] if '.part' in x['name']]
                self.assertEqual(b''.join(parts), w.read_bytes())
                self.assertEqual([len(x) for x in parts], [64, 37])
                self.assertEqual(plan['weights'][0]['original_sha256'], digest)
                with self.assertRaises(ValueError): m.prepare(args)

    def test_hash_mismatch_before_staging(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / 'a.zip').write_bytes(b'fixture')
            args = argparse.Namespace(repo='fixture/project', tag='weights-v1', weights_dir=root, staging_parent=root)
            with patch.object(m, 'PLAN', root / 'plan'), patch.object(m, 'WEIGHTS', [('task1', 'test', 'a.zip', '0'*64)]):
                with self.assertRaises(ValueError): m.prepare(args)
                self.assertFalse(m.PLAN.exists())

    def test_finalize_refuses_private_or_draft(self):
        plan = {'repo':'fixture/project', 'tag':'weights-v1'}
        with patch.object(m, 'read_plan', return_value=plan), patch.object(m, 'gh', return_value='{"private":true}'):
            with self.assertRaises(ValueError): m.finalize()
        with patch.object(m, 'read_plan', return_value=plan), patch.object(m, 'gh', return_value='{"private":false}'), patch.object(m, 'release', return_value={'draft':True}):
            with self.assertRaises(ValueError): m.finalize()

    def test_asset_size_and_optional_digest(self):
        e = {'bytes':123, 'sha256':'a'*64}
        self.assertTrue(m.matching({'size':123, 'digest':'sha256:'+'a'*64}, e))
        self.assertFalse(m.matching({'size':122}, e))
        self.assertFalse(m.matching({'size':123, 'digest':'sha256:'+'b'*64}, e))

    def test_finalize_writes_real_links_only_after_gate(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root/'docs').mkdir(); (root/'provenance').mkdir()
            (root/'README.md').write_text('Fixture repository')
            (root/'CHALLENGE_FORM.md').write_text('## Task 1\n**Current selection:** Not available / access restricted (until an authorized public URL exists).\nPublic URL: not yet supplied.\n')
            (root/'provenance/ARTIFACTS.json').write_text('{"submissions":[{"task":1}]}')
            plan = {'repo':'fixture/project','tag':'weights-v1', 'assets':[{'name':'a.zip','bytes':123,'sha256':'a'*64}], 'weights':[{'task':'task1','submission':'fixture','original_filename':'a.zip','original_sha256':'a'*64,'assets':['a.zip']}]}
            remote = {'draft':False,'html_url':'https://github.com/fixture/project/releases/tag/weights-v1','assets':[{'name':'a.zip','size':123,'digest':'sha256:'+'a'*64}]}
            with patch.object(m, 'ROOT', root), patch.object(m, 'read_plan', return_value=plan), patch.object(m, 'gh', return_value='{"private":false}'), patch.object(m, 'release', return_value=remote):
                m.finalize(); m.finalize()
            form = (root/'CHALLENGE_FORM.md').read_text()
            self.assertIn('**Current selection:** GitHub', form)
            self.assertIn('https://github.com/fixture/project/releases/download/weights-v1/a.zip', form)
            self.assertEqual(form.count('## GitHub repository and release'),1)

if __name__ == '__main__': unittest.main()
