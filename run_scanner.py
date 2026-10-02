"""Explicit isolated development scanner; production ownership comes in BT03."""
from runtime_environment import load_launcher_environment


def main() -> None:
    runtime = load_launcher_environment()
    if not runtime.development or runtime.mode != 'paper':
        raise ValueError('Standalone scanner is restricted to isolated development paper mode')
    from dashboard import initialize_runtime, bot_loop
    initialize_runtime()
    bot_loop()


if __name__ == '__main__':
    main()
