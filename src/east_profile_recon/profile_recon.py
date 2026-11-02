import sys
from pathlib import Path

if __package__:
    from .core import main
else:
    src_dir = str(Path(__file__).resolve().parents[1])
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from east_profile_recon.core import main


if __name__ == "__main__":
    main()
