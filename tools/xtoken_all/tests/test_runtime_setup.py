"""Stdlib fixtures for setup/reuse; never touch /opt or download a runtime."""

import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[3]


def load_helper():
    text = (ROOT / 'ensure_runtime.sh').read_text(encoding='utf-8')
    code = text.split("<<'XTOKEN_ENSURE_PY'\n", 1)[1].split('\nXTOKEN_ENSURE_PY\n', 1)[0]
    module = ModuleType('runtime_setup_fixture')
    exec(compile(code, 'ensure_runtime.sh:python', 'exec'), module.__dict__)
    return module


class RuntimeSetup(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.m = load_helper()
        self.m.DRIVER = self.base / 'node/opt/driver'
        self.ray = self.base / 'node/opt/ray'
        self.services = self.base / 'node/opt/services'
        self.m.RAY_STORE, self.m.GYM_STORE = self.ray, self.services
        self.m.PYTHON_HOME = self.base / 'node/root/python'
        self.m.SOURCE_ALIAS = self.base / 'node/opt/source-alias'
        self.m.ROLES = tuple((name, self.ray / name) for name in ('policy', 'generation', 'gym_actor'))
        self.m.SERVICES = tuple((name, self.services / 'kind' / name / '.venv') for name in ('model', 'agent', 'qa'))
        self.required = [self.m.DRIVER, *[p for _, p in self.m.ROLES], *[p for _, p in self.m.SERVICES], self.m.PYTHON_HOME]
        self.roots = [self.m.DRIVER, self.ray, self.services, self.m.PYTHON_HOME]
        self.m.DRIVER.parent.mkdir(parents=True)
        self.source = self.base / 'source'
        self.super = self.base / 'super'
        self.gym = self.super / '3rdparty/Gym-workspace/Gym'
        self.args = SimpleNamespace(source_root=self.source, super_source=self.super,
            nltk_data=self.base / 'nltk', archive=self.base / 'runtime.tar.gz', check_only=False)
        self.m.validate_sources = Mock(return_value=(self.source, self.super, self.gym, {'fixture': True}))
        self.real_occupied = self.m.occupied
        self.m.occupied = Mock(return_value=[])

    def ready_directories(self):
        for prefix in self.required:
            prefix.mkdir(parents=True)
        self.m.SOURCE_ALIAS.mkdir()

    def archive(self):
        self.m.MIN_FREE_BYTES = 1
        with tarfile.open(self.args.archive, 'w:gz') as bundle:
            for prefix in self.required:
                names = ('bin/python3.12', 'lib/python3.12/encodings/__init__.py') if prefix == self.m.PYTHON_HOME else ('pyvenv.cfg',)
                for relative in names:
                    content = b'fixture-' + relative.encode()
                    info = tarfile.TarInfo((prefix / relative).as_posix().lstrip('/'))
                    info.size = len(content)
                    bundle.addfile(info, io.BytesIO(content))
            content = b'shared fixture bytes'
            first = tarfile.TarInfo((self.m.DRIVER / 'shared-one').as_posix().lstrip('/'))
            first.size = len(content)
            bundle.addfile(first, io.BytesIO(content))
            second = tarfile.TarInfo((self.m.DRIVER / 'shared-two').as_posix().lstrip('/'))
            second.type, second.linkname = tarfile.LNKTYPE, first.name
            bundle.addfile(second)
            alias = tarfile.TarInfo((self.m.DRIVER / 'bin/python').as_posix().lstrip('/'))
            alias.type, alias.linkname = tarfile.SYMTYPE, str(self.m.PYTHON_HOME / 'bin/python3.12')
            bundle.addfile(alias)
        self.m.ARCHIVE_BYTES = self.args.archive.stat().st_size
        self.m.ARCHIVE_SHA256 = self.m.digest(self.args.archive)

    def test_healthy_reuse_needs_no_archive_or_marker(self):
        self.ready_directories()
        self.m.occupied.return_value = [123]
        self.m.check_runtime = Mock(return_value=[{'actual_check': True}])
        self.m.install_missing = Mock(side_effect=AssertionError('Must not extract'))
        result = self.m.ensure(self.args)
        self.assertTrue(result['reused'])
        self.assertIsNone(result['archive_sha256'])
        self.m.check_runtime.assert_called_once()
        self.m.install_missing.assert_not_called()
        self.m.occupied.assert_not_called()

    def test_existing_directories_do_not_bypass_a_failed_check(self):
        self.ready_directories()
        self.m.check_runtime = Mock(side_effect=RuntimeError('wrong native build'))
        with self.assertRaisesRegex(RuntimeError, 'wrong native build'):
            self.m.ensure(self.args)

    def test_check_only_missing_is_read_only(self):
        self.args.check_only = True
        self.m.install_missing = Mock(side_effect=AssertionError('Must not extract'))
        with self.assertRaisesRegex(FileNotFoundError, 'not provisioned'):
            self.m.ensure(self.args)
        self.assertFalse(self.m.SOURCE_ALIAS.exists())
        self.assertFalse(self.args.archive.exists())
        self.m.install_missing.assert_not_called()

    def test_missing_cache_does_not_create_source_alias(self):
        with self.assertRaisesRegex(FileNotFoundError, 'shared storage'):
            self.m.ensure(self.args)
        self.assertFalse(self.m.SOURCE_ALIAS.exists())

    def test_partial_incompatible_environment_is_preserved(self):
        self.m.DRIVER.mkdir()
        sentinel = self.m.DRIVER / 'keep'
        sentinel.write_text('original')
        self.m.SOURCE_ALIAS.mkdir()
        self.args.archive.write_bytes(b'not used')
        self.m.check_runtime = Mock(side_effect=RuntimeError('version mismatch'))
        self.m.install_missing = Mock(side_effect=AssertionError('Must not extract'))
        with self.assertRaisesRegex(RuntimeError, 'version mismatch'):
            self.m.ensure(self.args)
        self.assertEqual(sentinel.read_text(), 'original')
        self.m.install_missing.assert_not_called()

    def test_active_runtime_blocks_source_binding_without_killing(self):
        self.m.occupied.return_value = [123]
        with self.assertRaisesRegex(RuntimeError, '123'):
            self.m.ensure_source_alias(self.super, False)
        self.assertFalse(self.m.SOURCE_ALIAS.exists())

    def test_check_only_never_creates_missing_source_binding(self):
        with self.assertRaises(FileNotFoundError):
            self.m.ensure_source_alias(self.super, True)
        self.assertFalse(self.m.SOURCE_ALIAS.exists())

    def test_existing_regular_source_file_is_not_replaced(self):
        self.m.SOURCE_ALIAS.write_text('keep')
        with self.assertRaises(NotADirectoryError):
            self.m.ensure_source_alias(self.super, False)
        self.assertEqual(self.m.SOURCE_ALIAS.read_text(), 'keep')

    def test_checksum_rejection_occurs_before_tar(self):
        self.args.archive.write_bytes(b'wrong')
        with patch.object(self.m.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'SHA256'):
                self.m.install_missing(self.args.archive, self.roots, self.required)
            run.assert_not_called()

    def test_driver_version_mismatch_is_not_ready(self):
        self.m.DRIVER.joinpath('bin').mkdir(parents=True)
        self.m.DRIVER.joinpath('bin/python').write_text('fixture')
        self.m.probe = Mock(return_value={'python': '3.12.12', 'prefix': str(self.m.DRIVER),
            'base_prefix': str(self.m.PYTHON_HOME), 'versions': {**self.m.COMMON, 'torch': 'wrong'}})
        with self.assertRaisesRegex(RuntimeError, 'versions differ'):
            self.m.check_runtime(self.source, self.gym, self.args.nltk_data, True)

    @unittest.skipUnless(os.name == 'posix', 'Actual /proc process detection on Linux')
    def test_real_process_guard_detects_only_our_fixture_child(self):
        child = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(20)', str(self.m.DRIVER / 'bin/python')])
        try:
            seen = []
            for _ in range(20):
                seen = self.real_occupied([self.m.DRIVER])
                if child.pid in seen:
                    break
                time.sleep(0.02)
            self.assertIn(child.pid, seen)
        finally:
            child.terminate()
            child.wait(timeout=5)

    @unittest.skipUnless(os.name == 'posix', 'fcntl-backed node receipt on Linux')
    def test_failed_recheck_invalidates_old_ready_receipt(self):
        receipt = self.base / ('xtoken-runtime-ready-' + self.m.socket.gethostname() + '-' + str(os.geteuid()) + '.json')
        receipt.write_text('{"success":true}')
        self.m.ensure = Mock(side_effect=RuntimeError('native API changed'))
        argv = ['ensure_runtime.sh', '--source-root', str(self.source), '--archive', str(self.args.archive),
            '--super-source', str(self.super), '--nltk-data', str(self.args.nltk_data)]
        with patch.object(self.m, 'NODE_STATE_DIR', self.base), patch.object(sys, 'argv', argv):
            with self.assertRaises(SystemExit) as error:
                self.m.main()
        self.assertEqual(error.exception.code, 1)
        self.assertFalse(json.loads(receipt.read_text())['success'])

    @unittest.skipUnless(os.name == 'posix', 'fcntl-backed setup serialization on Linux')
    def test_concurrent_setup_stops_without_rewriting_receipt(self):
        import fcntl
        receipt = self.base / ('xtoken-runtime-ready-' + self.m.socket.gethostname() + '-' + str(os.geteuid()) + '.json')
        receipt.write_text('{"success":true,"owner":"other setup"}')
        self.m.ensure = Mock(side_effect=AssertionError('Must not enter setup'))
        argv = ['ensure_runtime.sh', '--source-root', str(self.source), '--archive', str(self.args.archive),
            '--super-source', str(self.super), '--nltk-data', str(self.args.nltk_data)]
        with receipt.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(self.m, 'NODE_STATE_DIR', self.base), patch.object(sys, 'argv', argv):
                with self.assertRaises(SystemExit):
                    self.m.main()
        self.assertEqual(json.loads(receipt.read_text())['owner'], 'other setup')
        self.m.ensure.assert_not_called()

    @unittest.skipUnless(os.name == 'posix', 'GNU tar/hardlink publication is exercised on Linux')
    def test_cold_restore_then_reuse_and_hardlinks(self):
        self.archive()
        self.m.SOURCE_ALIAS.mkdir()
        self.m.check_runtime = Mock(return_value=[{'fixture_import': True}])
        result = self.m.ensure(self.args)
        self.assertFalse(result['reused'])
        self.assertEqual(len(result['restored_prefixes']), len(self.required))
        self.assertEqual((self.m.DRIVER / 'shared-one').stat().st_ino, (self.m.DRIVER / 'shared-two').stat().st_ino)
        self.assertEqual((self.m.DRIVER / 'bin/python').resolve(), self.m.PYTHON_HOME / 'bin/python3.12')
        self.assertFalse(list(self.m.DRIVER.parent.glob('.xtoken-runtime-stage-*')))
        self.args.archive.unlink()
        with patch.object(self.m, 'install_missing', side_effect=AssertionError('No second extraction')):
            self.assertTrue(self.m.ensure(self.args)['reused'])

    @unittest.skipUnless(os.name == 'posix', 'GNU tar fixture')
    def test_wrong_existing_base_python_blocks_all_publication(self):
        self.archive()
        (self.m.PYTHON_HOME / 'bin').mkdir(parents=True)
        (self.m.PYTHON_HOME / 'bin/python3.12').write_text('different build')
        with self.assertRaisesRegex(RuntimeError, 'base Python differs'):
            self.m.install_missing(self.args.archive, self.roots, self.required)
        self.assertFalse(self.m.DRIVER.exists())
        self.assertEqual((self.m.PYTHON_HOME / 'bin/python3.12').read_text(), 'different build')


if __name__ == '__main__':
    unittest.main()
