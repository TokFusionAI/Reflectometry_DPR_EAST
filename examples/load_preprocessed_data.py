from pathlib import Path

from east_profile_recon.data import load_QVWdata, load_vacuum_data


DATA_DIR = Path(__file__).resolve().parent / "data"
PLASMA_FILE = DATA_DIR / "data_151858_5s.h5"
VACUUM_FILE = DATA_DIR / "vacuum_data.h5"


def main():
    time_ms, q_data, v_data, w_data = load_QVWdata(PLASMA_FILE)
    vacuum_data = load_vacuum_data(VACUUM_FILE)

    print(f"Plasma data: {PLASMA_FILE}")
    print(f"  time: shape={time_ms.shape}, unit=ms")
    print(f"  Q: shape={q_data.shape}, dtype={q_data.dtype}")
    print(f"  V: shape={v_data.shape}, dtype={v_data.dtype}")
    print(f"  W: shape={w_data.shape}, dtype={w_data.dtype}")
    print(f"  time range: {time_ms.min():.6f} to {time_ms.max():.6f} ms")

    vacuum_names = (
        "fb_Q",
        "fb_err_Q",
        "fb_V",
        "fb_err_V",
        "fb_W",
        "fb_err_W",
    )
    print(f"Vacuum reference: {VACUUM_FILE}")
    for name, values in zip(vacuum_names, vacuum_data):
        print(f"  {name}: shape={values.shape}, dtype={values.dtype}")


if __name__ == "__main__":
    main()
