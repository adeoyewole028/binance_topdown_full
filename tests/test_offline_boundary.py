"""Verify the disposable HTTP/evaluator boundary itself cannot escape."""
from pathlib import Path
import socket

import pytest

from tests.legacy_sandbox import LegacySandbox, OfflineViolation, ROOT


def test_socket_transport_is_blocked() -> None:
    with LegacySandbox() as sandbox:
        with socket.socket() as transport:
            with pytest.raises(OfflineViolation):
                transport.connect(('127.0.0.1', 9))
        assert sandbox.network_attempts == 1


def test_dns_resolution_is_blocked() -> None:
    with LegacySandbox() as sandbox:
        with pytest.raises(OfflineViolation):
            socket.getaddrinfo('example.com', 443)
        assert sandbox.network_attempts == 1


def test_real_env_read_is_blocked() -> None:
    with LegacySandbox() as sandbox:
        with pytest.raises(OfflineViolation):
            (ROOT / '.env').read_text(encoding='utf-8')
        assert sandbox.forbidden_file_attempts == 1


def test_non_disposable_write_is_blocked(tmp_path: Path) -> None:
    forbidden = tmp_path / 'must-not-be-created.txt'
    with LegacySandbox() as sandbox:
        with pytest.raises(OfflineViolation):
            forbidden.write_text('test', encoding='utf-8')
        assert sandbox.forbidden_file_attempts == 1
    assert not forbidden.exists()


def test_non_disposable_directory_is_blocked(tmp_path: Path) -> None:
    forbidden = tmp_path / 'must-not-be-created'
    with LegacySandbox():
        with pytest.raises(OfflineViolation):
            forbidden.mkdir()
    assert not forbidden.exists()


@pytest.mark.parametrize('name', ['dashboard.py', 'topdown_daemon.py', 'backtester.py', 'utils.py',
                                  'operator_security.py', 'runtime_environment.py', 'wsgi.py', 'run_scanner.py'])
def test_application_source_compiles_without_import(name: str) -> None:
    source = (ROOT / name).read_text(encoding='utf-8')
    compile(source, name, 'exec')
