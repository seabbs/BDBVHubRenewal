"""Scheduled workflow runs: what a tick runs, and the timer that runs it."""

import plistlib

import pytest
from flowlib import schedule


def test_a_tick_runs_this_machine_s_targets_in_order():
    cfg = {
        "SCHEDULE_TARGETS": "score land",
        "CORES": "4",
        "MEM_MB": "32768",
        "SIM_SLOTS": "2",
    }
    got = schedule.plan(cfg)
    assert [t for t, _ in got] == ["score", "land"]
    score = got[0][1]
    assert score[:3] == ["nice", "-n", "19"]
    assert "--forcerun" in score
    assert "release_list" in score
    assert "workflow/profiles/default" in score
    assert score[score.index("--cores") :][:2] == ["--cores", "4"]
    assert "mem_mb=32768" in score


def test_every_target_uses_the_default_profile():
    got = schedule.plan({"SCHEDULE_TARGETS": "simulate finetune publish"})
    assert all("workflow/profiles/default" in cmd for _, cmd in got)


def snakemake_targets(cmd):
    """The targets of a planned Snakemake command."""
    start = cmd.index("snakemake") + 1
    return cmd[start : cmd.index("--workflow-profile")]


def test_the_pool_targets_are_one_snakemake_run():
    # One run over the machine's whole budget, so a free core or the GPU
    # takes whichever queued job fits.
    got = schedule.plan(
        {"SCHEDULE_TARGETS": "simulate finetune score land publish"}
    )
    assert [t for t, _ in got] == ["simulate finetune score", "land", "publish"]
    pool = got[0][1]
    assert snakemake_targets(pool) == ["simulate", "finetune", "score"]
    assert pool.count("--workflow-profile") == 1
    assert "release_list" in pool


def test_the_outward_targets_keep_their_order_after_the_pool():
    got = schedule.plan({"SCHEDULE_TARGETS": "publish simulate release"})
    assert [t for t, _ in got] == ["simulate", "publish", "release"]


def test_a_pool_without_score_lists_no_releases():
    got = schedule.plan({"SCHEDULE_TARGETS": "simulate finetune"})
    assert "release_list" not in got[0][1]


def test_land_and_release_need_the_pool_publish_does_not():
    # publish reads only saved checkpoints, so it never retries a failed
    # job; land and release would rerun failed scoring or simulation.
    assert {"land", "release"} == schedule.NEEDS_POOL


def test_no_targets_means_nothing_runs():
    assert schedule.plan({"SCHEDULE_TARGETS": ""}) == []


def test_unknown_targets_are_an_error():
    with pytest.raises(ValueError, match="nope"):
        schedule.plan({"SCHEDULE_TARGETS": "score nope"})


def test_a_tick_waits_for_room(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable:   1048576 kB\n")  # 1 GB
    cfg = {"MIN_FREE_GB": "20", "MIN_DISK_GB": "0"}
    ok, why = schedule.room(cfg, tmp_path, meminfo)
    assert not ok
    assert "memory" in why
    ok, _ = schedule.room(
        {"MIN_FREE_GB": "0", "MIN_DISK_GB": "0"}, tmp_path, meminfo
    )
    assert ok


def test_no_meminfo_skips_the_memory_check(tmp_path):
    ok, _ = schedule.room(
        {"MIN_FREE_GB": "999", "MIN_DISK_GB": "0"}, tmp_path, tmp_path / "none"
    )
    assert ok


def test_runs_never_overlap(tmp_path):
    lock = tmp_path / "tick.lock"
    with schedule.locked(lock) as first:
        assert first
        with schedule.locked(lock) as second:
            assert not second
    with schedule.locked(lock) as again:
        assert again


def test_the_interval_becomes_a_cron_line(tmp_path):
    assert schedule.cron_spec(30) == "*/30 * * * *"
    assert schedule.cron_spec(120) == "7 */2 * * *"
    assert schedule.cron_spec(60) == "7 * * * *"
    line = schedule.cron_line(tmp_path, 30)
    assert line.startswith("*/30 * * * * cd ")
    assert str(tmp_path) in line
    assert line.endswith(schedule.MARK)


def test_installing_cron_keeps_other_lines_and_replaces_its_own(tmp_path):
    old = "0 1 * * * backup\n*/5 * * * * cd /x && tick " + schedule.MARK + "\n"
    new = schedule.merge_crontab(old, schedule.cron_line(tmp_path, 30))
    assert "0 1 * * * backup" in new
    assert new.count(schedule.MARK) == 1
    assert schedule.merge_crontab(new, None).count(schedule.MARK) == 0


def test_systemd_user_units(tmp_path):
    service, timer = schedule.systemd_units(tmp_path, 45)
    assert f"WorkingDirectory={tmp_path}" in service
    assert "workflow/scripts/tick.py" in service
    assert "OnUnitActiveSec=45min" in timer
    assert "Persistent=true" in timer


def test_a_launchd_agent(tmp_path):
    data = plistlib.loads(schedule.launchd_plist(tmp_path, 60))
    assert data["StartInterval"] == 3600
    assert data["WorkingDirectory"] == str(tmp_path)
    assert data["ProgramArguments"][-1] == "workflow/scripts/tick.py"


def test_the_timer_kind_follows_the_platform():
    assert schedule.timer_kind({"SCHEDULE_KIND": "cron"}, "linux") == "cron"
    assert schedule.timer_kind({}, "darwin") == "launchd"
    assert schedule.timer_kind({}, "linux", has_systemd=True) == "systemd"
    assert schedule.timer_kind({}, "linux", has_systemd=False) == "cron"


def test_timers_run_with_the_installing_path(tmp_path):
    assert "PATH=/opt/uv/bin:/usr/bin python3" in schedule.cron_line(
        tmp_path, 30, "/opt/uv/bin:/usr/bin"
    )
    service, _ = schedule.systemd_units(tmp_path, 30, "/opt/uv/bin")
    assert "Environment=PATH=/opt/uv/bin" in service
    data = plistlib.loads(schedule.launchd_plist(tmp_path, 30, "/opt/uv/bin"))
    assert data["EnvironmentVariables"]["PATH"] == "/opt/uv/bin"


@pytest.mark.parametrize("source", ["release", "remote"])
def test_a_finetune_run_lists_the_batches_it_can_have_afresh(source):
    got = schedule.plan(
        {"SCHEDULE_TARGETS": "finetune score", "SIMS_FROM": source}
    )
    pool = got[0][1]
    assert pool.count("--forcerun") == 1
    forced = pool[pool.index("--forcerun") + 1 :]
    assert "release_list" in forced
    assert "available_batches" in forced


def test_a_machine_with_the_rounds_lists_no_batches():
    # --forcerun adds a rule to the run even when nothing needs it.
    got = schedule.plan({"SCHEDULE_TARGETS": "finetune"})
    assert "--forcerun" not in got[0][1]


def test_the_paper_is_not_a_tick_target():
    # The paper run pins the checkpoint into tracked config and copies
    # into docs/, so it is run on purpose (`task paper`), not on a timer.
    with pytest.raises(ValueError, match="paper"):
        schedule.plan({"SCHEDULE_TARGETS": "score paper"})


def test_a_tick_trains_one_segment_a_run():
    got = schedule.plan({"SCHEDULE_TARGETS": "simulate finetune"})
    pool = got[0][1]
    assert pool[pool.index("--config") + 1] == "next_segment_only=True"


def test_a_tick_without_finetune_sets_no_config():
    got = schedule.plan({"SCHEDULE_TARGETS": "score"})
    assert "--config" not in got[0][1]
