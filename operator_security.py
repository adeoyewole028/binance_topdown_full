"""Single configured operator, signed expiring sessions and shared login throttling."""
from __future__ import annotations

from datetime import timedelta
from contextlib import closing
import hashlib
import hmac
from pathlib import Path
import secrets
import sqlite3
import time
from typing import Any

from flask import Flask, Response, redirect, render_template_string, request, session
from werkzeug.security import check_password_hash
from werkzeug.wrappers import Response as WerkzeugResponse

SESSION_SECONDS = 1800
LOGIN_WINDOW_SECONDS = 900
LOGIN_ATTEMPTS = 5


def configure_operator(app: Flask, username: str, password_hash: str, secret: str,
                       directory: Path, *, development: bool = False) -> None:
    if not username or not password_hash.startswith(('scrypt:', 'pbkdf2:')) or len(password_hash.split('$')) != 3:
        raise ValueError('A configured operator and Werkzeug password hash are required')
    if len(secret) < 32 or secret.lower() in {'supersecretkey123', 'change_me'}:
        raise ValueError('A unique session secret of at least 32 characters is required')
    app.config.update(
        SECRET_KEY=secret, OPERATOR_USERNAME=username, OPERATOR_PASSWORD_HASH=password_hash,
        AUTH_REVISION=hashlib.sha256((username + password_hash + secret).encode()).hexdigest(),
        AUTH_DB=str(directory / 'operator-auth.sqlite3'), SESSION_COOKIE_SECURE=not development,
        SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Strict',
        PERMANENT_SESSION_LIFETIME=timedelta(seconds=SESSION_SECONDS),
        SESSION_REFRESH_EACH_REQUEST=False, MAX_CONTENT_LENGTH=16 * 1024,
    )
    directory.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(app.config['AUTH_DB'])) as connection, connection:
        connection.execute('CREATE TABLE IF NOT EXISTS attempts (occurred REAL NOT NULL)')
        connection.execute('CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, expires REAL NOT NULL, revision TEXT NOT NULL)')


def install_operator_security(app: Flask) -> None:
    def csrf_token() -> str:
        if 'csrf' not in session:
            session['csrf'] = secrets.token_urlsafe(32)
        return str(session['csrf'])

    app.jinja_env.globals['csrf_token'] = csrf_token

    @app.before_request
    def protect_request() -> WerkzeugResponse | tuple[dict[str, str], int] | None:
        public = request.endpoint == 'operator_login'
        authenticated = (
            app.config.get('AUTH_REVISION') is not None
            and hmac.compare_digest(str(session.get('operator', '')), app.config['AUTH_REVISION'])
            and 0 <= time.time() - float(session.get('signed_in_at', 0)) < SESSION_SECONDS
        )
        if authenticated:
            try:
                with closing(sqlite3.connect(app.config['AUTH_DB'], timeout=5)) as connection:
                    authenticated = connection.execute(
                        'SELECT 1 FROM sessions WHERE id = ? AND expires > ? AND revision = ?',
                        (session.get('session_id', ''), time.time(), app.config['AUTH_REVISION']),
                    ).fetchone() is not None
            except sqlite3.Error:
                return {'error': 'Operator access temporarily unavailable'}, 503
        if not public and not authenticated:
            if request.method == 'GET' and request.path == '/':
                return redirect('/login')
            return {'error': 'Sign-in required'}, 401
        if not app.config.get('AUTH_REVISION'):
            return {'error': 'Operator access is not configured'}, 503
        if request.method not in {'GET', 'HEAD', 'OPTIONS'}:
            supplied = request.form.get('csrf_token', '') or request.headers.get('X-CSRF-Token', '')
            expected = str(session.get('csrf', ''))
            origin = request.headers.get('Origin')
            if (not expected or not hmac.compare_digest(expected.encode(), supplied.encode())
                    or (origin is not None and origin != request.host_url.rstrip('/'))):
                return {'error': 'Request verification failed'}, 403
        return None

    @app.after_request
    def protect_response(response: Response) -> Response:
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers.pop('Access-Control-Allow-Origin', None)
        return response

    @app.route('/login', methods=['GET', 'POST'], endpoint='operator_login')
    def login() -> Any:
        if request.method == 'POST':
            now = time.time()
            # Global account throttle also covers fresh cookies, forged usernames,
            # concurrent HTTP workers and worker restarts. Never trust forwarded IPs.
            try:
                with closing(sqlite3.connect(app.config['AUTH_DB'], timeout=5)) as connection, connection:
                    connection.execute('BEGIN IMMEDIATE')
                    connection.execute('DELETE FROM attempts WHERE occurred <= ?', (now - LOGIN_WINDOW_SECONDS,))
                    count = connection.execute('SELECT count(*) FROM attempts').fetchone()[0]
                    if count >= LOGIN_ATTEMPTS:
                        return {'error': 'Sign-in temporarily unavailable; try again later'}, 429, {'Retry-After': str(LOGIN_WINDOW_SECONDS)}
                    connection.execute('INSERT INTO attempts VALUES (?)', (now,))
            except sqlite3.Error:
                return {'error': 'Operator access temporarily unavailable'}, 503
            password_ok = check_password_hash(app.config['OPERATOR_PASSWORD_HASH'], request.form.get('password', ''))
            username_ok = hmac.compare_digest(app.config['OPERATOR_USERNAME'].encode(), request.form.get('username', '').encode())
            if not (password_ok and username_ok):
                return {'error': 'Invalid sign-in'}, 401
            session_id = secrets.token_urlsafe(32)
            try:
                with closing(sqlite3.connect(app.config['AUTH_DB'], timeout=5)) as connection, connection:
                    connection.execute('DELETE FROM sessions WHERE expires <= ? OR revision != ?', (now, app.config['AUTH_REVISION']))
                    connection.execute('INSERT INTO sessions VALUES (?, ?, ?)',
                                       (session_id, now + SESSION_SECONDS, app.config['AUTH_REVISION']))
            except sqlite3.Error:
                return {'error': 'Operator access temporarily unavailable'}, 503
            session.clear()
            session.permanent = True
            session['operator'] = app.config['AUTH_REVISION']
            session['signed_in_at'] = now
            session['session_id'] = session_id
            csrf_token()
            return redirect('/')
        return render_template_string('''<!doctype html><html lang="en"><title>Operator sign-in</title>
            <h1>Operator sign-in</h1><form method="post">
            <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
            <label>Operator <input name="username" autocomplete="username" required></label>
            <label>Password <input name="password" type="password" autocomplete="current-password" required></label>
            <button type="submit">Sign in</button></form></html>''')

    @app.post('/logout')
    def logout() -> Any:
        try:
            with closing(sqlite3.connect(app.config['AUTH_DB'], timeout=5)) as connection, connection:
                connection.execute('DELETE FROM sessions WHERE id = ?', (session.get('session_id', ''),))
        except sqlite3.Error:
            return {'error': 'Sign-out temporarily unavailable'}, 503
        session.clear()
        return redirect('/login')
