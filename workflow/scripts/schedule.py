"""Show or remove the old timer that ran workflow ticks.

    python3 workflow/scripts/schedule.py show|uninstall

The persistent coordinator run (`task run`, workflow/scripts/coordinate.py)
replaces the timer, so `install` is refused.

The timer runs workflow/scripts/tick.py every `SCHEDULE_EVERY` minutes
from this checkout: a cron line or a systemd user timer on Linux, a
launchd agent on macOS (`SCHEDULE_KIND` in config/local.env to choose).
The units are written from where this checkout is and the PATH it is
installed from, so nothing machine-specific is committed.
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "workflow"))

from flowlib import localconfig, schedule  # noqa: E402

UNITS = Path.home() / ".config" / "systemd" / "user"
AGENTS = Path.home() / "Library" / "LaunchAgents"


def main() -> int:
    """Show, install or remove the timer."""
    action = sys.argv[1] if len(sys.argv) > 1 else "show"
    if action == "install":
        # The persistent coordinator run replaces the timer.
        print(
            "no timer is installed any more: run `task run` once (under"
            " nohup or tmux); `ACTION=uninstall` removes an old timer"
        )
        return 2
    cfg = localconfig.read()
    every = int(cfg.get("SCHEDULE_EVERY") or 60)
    path = shutil.which("uv") and str(Path(shutil.which("uv")).parent)
    path = ":".join(
        p for p in [path, "/usr/local/bin", "/usr/bin", "/bin"] if p
    )
    kind = schedule.timer_kind(
        cfg, sys.platform, has_systemd=shutil.which("systemctl") is not None
    )
    if kind == "cron":
        line = schedule.cron_line(ROOT, every, path)
        if action == "show":
            print(line)
            return 0
        old = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        text = schedule.merge_crontab(
            old.stdout if old.returncode == 0 else "",
            line if action == "install" else None,
        )
        subprocess.run(["crontab", "-"], input=text, text=True, check=True)
    elif kind == "systemd":
        service, timer = schedule.systemd_units(ROOT, every, path)
        if action == "show":
            print(service, timer, sep="\n")
            return 0
        name = schedule.LABEL
        if action == "install":
            UNITS.mkdir(parents=True, exist_ok=True)
            (UNITS / f"{name}.service").write_text(service)
            (UNITS / f"{name}.timer").write_text(timer)
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
            subprocess.run(
                ["systemctl", "--user", "enable", "--now", f"{name}.timer"],
                check=True,
            )
        else:
            subprocess.run(
                ["systemctl", "--user", "disable", "--now", f"{name}.timer"],
                check=False,
            )
            for f in (f"{name}.service", f"{name}.timer"):
                (UNITS / f).unlink(missing_ok=True)
    elif kind == "launchd":
        plist = schedule.launchd_plist(ROOT, every, path)
        if action == "show":
            print(plist.decode())
            return 0
        target = AGENTS / f"{schedule.LABEL}.plist"
        subprocess.run(
            ["launchctl", "unload", str(target)],
            check=False,
            capture_output=True,
        )
        if action == "install":
            AGENTS.mkdir(parents=True, exist_ok=True)
            target.write_bytes(plist)
            subprocess.run(["launchctl", "load", str(target)], check=True)
        else:
            target.unlink(missing_ok=True)
    else:
        print(f"unknown SCHEDULE_KIND {kind!r}: cron, systemd or launchd")
        return 2
    print(f"{action}: {kind} every {every} min for {ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
