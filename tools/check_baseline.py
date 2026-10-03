"""Run BT01 offline tests and emit redacted, explicit baseline evidence."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional redacted JSON evidence file')
    parser.add_argument('--ticket', default='offline regression suite', help='Ticket owning this validation snapshot')
    args = parser.parse_args()
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith(('BINANCE_', 'OPERATOR_', 'ALLOW_', 'TRADING_')) or name in (
            'API_KEY', 'API_SECRET', 'FLASK_SECRET_KEY', 'DATA_DIR', 'APP_ENV', 'FLASK_ENV',
            'FLASK_DEBUG', 'FLASK_RUN_DEBUG', 'FLASK_RUN_RELOAD', 'PAPER_TRADING', 'TESTNET', 'CREDENTIAL_VENUE',
        ):
            environment.pop(name, None)
    environment['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    with tempfile.TemporaryDirectory(prefix='bt01-report-') as directory:
        report = Path(directory) / 'junit.xml'
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', '-q', f'--junitxml={report}'],
            cwd=ROOT, env=environment, check=False, timeout=300,
        )
        suites = ET.parse(report).getroot()
        cases = suites.findall('.//testcase')
        failures = sum(case.find('failure') is not None or case.find('error') is not None for case in cases)
        xfails = []
        for case in cases:
            skipped = case.find('skipped')
            if skipped is not None and skipped.get('type') == 'pytest.xfail':
                xfails.append(case.attrib['name'])
        other_skips = sum(case.find('skipped') is not None for case in cases) - len(xfails)
        evidence = {
            'ticket': args.ticket, 'scope': 'offline behavior and safety checks; not deployed/provider acceptance',
            'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'python': platform.python_version(),
            'dependencies': {name: version(name) for name in ('ccxt', 'pandas', 'Flask', 'gunicorn', 'python-dotenv', 'pytest', 'pyflakes', 'mypy')},
            'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ('dashboard.py', 'topdown_daemon.py', 'backtester.py', 'utils.py', 'operator_security.py', 'runtime_environment.py', 'wsgi.py', 'run_scanner.py', 'gunicorn.conf.py', 'state_store.py', 'execution_owner.py', 'execution_worker.py', 'state_admin.py', 'manage_state.py', 'run_worker.py', 'configuration.py', 'operations_policy.py', 'market_runtime.py', 'market_adapter.py', 'request_budget.py', 'paper_policy.py', 'paper_filters.py', 'exchange_adapter.py', 'exchange_execution.py')},
            'fixture_sha256': hashlib.sha256((ROOT / 'tests/fixtures/legacy_strategy_v1.json').read_bytes()).hexdigest(),
            'tests': len(cases), 'passed': len(cases) - failures - len(xfails) - other_skips,
            'known_defects_expected_failure': xfails, 'failures': failures,
            'other_skips': other_skips, 'pytest_exit_code': result.returncode,
            'boundaries': 'Fake providers and pinned CCXT with replaced raw transports only; socket/DNS guards; cleared credentials; temporary state; guarded writes; unmodified normal imports; explicit guarded launch checks.',
        }
    print(json.dumps(evidence, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    return result.returncode if result.returncode else int(failures > 0 or other_skips > 0)


if __name__ == '__main__':
    raise SystemExit(main())
