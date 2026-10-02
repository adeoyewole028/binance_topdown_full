"""Explicit production HTTP launcher. It never owns a scanner or execution loop."""
from runtime_environment import load_launcher_environment

runtime = load_launcher_environment()
if runtime.development:
    raise ValueError('The production WSGI launcher requires APP_ENV=production')

from dashboard import initialize_runtime

app = initialize_runtime()
