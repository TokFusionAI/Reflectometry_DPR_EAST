from pathlib import Path

from east_profile_recon.core import run_single_timepoint


EXAMPLE_DIR = Path(__file__).resolve().parent
DATA_DIR = EXAMPLE_DIR / "data"


def main():
    output_file = run_single_timepoint(
        shot=151858,
        IT=10735,
        time_point=5.0,
        qvw_path=DATA_DIR / "data_151858_5s.h5",
        vacuum_path=DATA_DIR / "vacuum_data.h5",
        output_root=EXAMPLE_DIR.parent / "outputs",
        device=None,
    )
    print(f"Saved reconstruction to: {output_file}")


if __name__ == "__main__":
    main()
