# Local Compute Status

Date: 2026-06-16

## WSL Memory Change

The local WSL configuration was changed to support a slow FS8 fallback run:

```text
[wsl2]
memory=8GB
swap=64GB
swapFile=D:\\WSL\\swap\\swap.vhdx
localhostForwarding=true
```

After `wsl --shutdown`, WSL reported:

```text
Mem: 7.6 GiB
Swap: 64 GiB
```

This makes the run possible, but not fast. Swap is much slower than physical RAM.

## FS8 Job

Started:

```text
work/run_fs82_local.sh
```

PID recorded in:

```text
work/fs82_local_driver.pid
```

Driver logs:

```text
work/fs82_local_driver.log
work/fs82_local_driver.err.log
```

FS8 outputs:

```text
D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\fs82
```

FS8 logs:

```text
D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\reprocessed_2026\logs_fs82
```

Run policy:

- image: `freesurfer/freesurfer:8.2.0`
- threads: `2`
- order: `2018`, `2022`, `kate_fs82_base`, `2018.long.kate_fs82_base`,
  `2022.long.kate_fs82_base`, then `2024_cross_probe`
- 2024 is not included in the longitudinal base.

Current observed status when created:

- Docker daemon: running manually through `dockerd`
- FS8 image: pulling
- FS8 subjects: not started yet
- WSL swap: available, not used yet

Status check command:

```bash
wsl bash -lc "bash /mnt/c/Users/Lenovo/Documents/Codex/2026-06-11/prior-conversation-with-codex-conversation-role/work/check_fs82_status.sh"
```

## ReconAny

Prepared script:

```text
work/run_reconany_local.sh
```

Do not run concurrently with FS8 on this machine. It should be run only after the
FS8 job finishes or is intentionally stopped. The script checks whether
`run_recon-any` exists in the selected FreeSurfer image/build and exits with a
clear error if not.
