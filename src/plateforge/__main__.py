"""Run PlateForge as a module: python -m plateforge."""

try:
    from plateforge.cli import main
except ImportError:  # cli.py arrives in a later commit

    def main():
        from plateforge import __version__

        print(f"PlateForge {__version__}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
