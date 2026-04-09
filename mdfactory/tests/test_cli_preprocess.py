# ABOUTME: Smoke tests for the analysis preprocess CLI command, verifying
# ABOUTME: dry-run behavior and preprocess configuration generation.
"""Smoke tests for analysis preprocess CLI."""

import pandas as pd

from mdfactory import cli


def test_analysis_preprocess_dry_run(monkeypatch, tmp_path):
    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    script_path = tmp_path / "script.sh"
    script_path.write_text("#!/bin/sh\necho done\n")
    script_path.chmod(0o755)

    class DummyBuildInput:
        hash = "HASH"
        simulation_type = "bilayer"

    class DummySimulation:
        build_input = DummyBuildInput()

    df = pd.DataFrame(
        {
            "path": [str(sim_dir)],
            "simulation": [DummySimulation()],
        }
    )

    def fake_discover(self):
        self._discovery_df = df
        return df

    monkeypatch.setattr(cli.SimulationStore, "discover", fake_discover)
    monkeypatch.setattr(cli, "resolve_simulation_paths", lambda *args, **kwargs: [sim_dir])

    cli.analysis_preprocess(
        source=sim_dir,
        script=script_path,
        output=None,
        dry_run=True,
    )


def test_analysis_preprocess_requires_script(tmp_path):
    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()

    try:
        cli.analysis_preprocess(source=sim_dir)
    except ValueError as exc:
        assert "--script" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing script")
