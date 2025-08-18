import os
import shlex
import subprocess
import sys


BLACKHOLE_DEVICE_NAME = os.getenv("BLACKHOLE_NAME", "BlackHole 2ch")
OUTPUT_PATH = os.getenv("OUTPUT", os.path.join(os.getcwd(), "capture.wav"))
DURATION_SECONDS = 150


def run(cmd: str) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return proc.returncode, proc.stdout, proc.stderr


def main() -> None:
    # Build base SoX command using CoreAudio device by name
    base_cmd = f"sox -t coreaudio {shlex.quote(BLACKHOLE_DEVICE_NAME)} {shlex.quote(OUTPUT_PATH)}"

    # If duration provided, add trim to stop automatically
    if DURATION_SECONDS:
        try:
            dur = float(DURATION_SECONDS)
        except ValueError:
            print("DURATION must be a number (seconds)", file=sys.stderr)
            sys.exit(2)
        cmd = f"{base_cmd} trim 0 {dur:.2f}"
    else:
        cmd = base_cmd

    print(
        f"Recording{f' {DURATION_SECONDS}s' if DURATION_SECONDS else ''} from '{BLACKHOLE_DEVICE_NAME}' → {OUTPUT_PATH}"
    )
    print(f"Command: {cmd}")
    code, out, err = run(cmd)
    if code != 0:
        print(err, file=sys.stderr)
        sys.exit(code)
    print("Saved.")


if __name__ == "__main__":
    main()
