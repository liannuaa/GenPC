# Workspace Notes

## Default Python Environment
- This project defaults to the conda environment `genpc`.
- Do not assume `python` or `pip` from `PATH` points to that environment.
- Prefer the explicit interpreter and pip paths below when installing deps or running code:
  - `/root/autodl-tmp/conda-envs/genpc/bin/python`
  - `/root/autodl-tmp/conda-envs/genpc/bin/pip`

## Command Convention
- Install packages into `genpc`, not `base`.
- Run project scripts with `/root/autodl-tmp/conda-envs/genpc/bin/python ...`.
- If a command needs `pip`, use `/root/autodl-tmp/conda-envs/genpc/bin/python -m pip ...`.
