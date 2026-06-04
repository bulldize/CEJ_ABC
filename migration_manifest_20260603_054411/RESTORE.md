# Restore Notes

## Expected Layout After Extraction

```text
/root/workspace/CEJ_ABC
/root/ToothFairy3
/root/cej_runs/run_unsup_001
/root/cej_isolated_runs/C/...
/root/backup_models
/root/xray
/root/clash
```

## Environment

The original environment used Python 3.10.16. CUDA toolkit reported `12.1`.

Recommended restore sequence:

```bash
cd /root/workspace/CEJ_ABC
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r migration_manifest_20260603_054411/env_manifest/pip_freeze_venv.txt
```

If exact package resolution fails, use the `pip_freeze_venv.txt` file as the authoritative package list from this server.

## Proxy / VPN

The original server had an xray process like:

```text
/root/xray/xray run -c /root/xray/config.json
```

Sensitive proxy configuration is included because this migration requested server configuration preservation. Review it before sharing the archive.

## Paper Workflow

For paper writing, start from:

```text
/root/workspace/CEJ_ABC/migration_manifest_20260603_054411/paper_artifacts/source_paths/
```

This folder contains copied metrics, manifests, summaries, logs, configs, and selected HTML figure sources without requiring a search through the full run tree.
