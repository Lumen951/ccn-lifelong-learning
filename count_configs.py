import sys

from experiment.config import CountConfig

if __name__ == "__main__":
    try:
        CountConfig(sys.argv[1:])
    except (ValueError, OSError) as exc:
        print(f"Cannot count configurations: {exc}", file=sys.stderr)
        sys.exit(1)
