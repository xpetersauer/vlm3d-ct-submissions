FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
WORKDIR /opt/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY model_code_from_original.py vlm3d_common.py sota_models.py model_def.py process.py ./
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && useradd -m appuser && chown -R appuser:appuser /opt/app
ENV PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# VLM3D_TASK1_METRIC_CALIBRATION_COPY_V1
COPY task1_calibrate_output.py /opt/app/task1_calibrate_output.py

# VLM3D_NINJA_PLATFORM_REPAIR_V2
RUN python -m pip uninstall -y ninja || true; python -c "import pathlib,site,shutil; roots={pathlib.Path(p) for p in [*site.getsitepackages(),site.getusersitepackages()] if p}; stale=[p for r in roots if r.exists() for p in r.iterdir() if p.name.lower().startswith('ninja-') and (p.name.lower().endswith('.dist-info') or p.name.lower().endswith('.egg-info'))]; print('Removing stale Ninja metadata:',*[str(p) for p in stale]); [shutil.rmtree(p) if p.is_dir() else p.unlink() for p in stale]"; python -m pip install --no-cache-dir --only-binary=:all: ninja==1.11.1.4; python -c "import importlib.metadata as m,platform,shutil,subprocess; ds=[d for d in m.distributions() if (d.metadata.get('Name') or '').lower()=='ninja']; assert len(ds)==1,[(d.version,str(d._path)) for d in ds]; assert ds[0].version=='1.11.1.4',ds[0].version; p=shutil.which('ninja'); assert p; print('NINJA PLATFORM REPAIR PASS',platform.machine(),ds[0].version,p,subprocess.check_output([p,'--version'],text=True).strip())"

# VLM3D_FORITHMUS_STANDALONE_MAC_V4: task1 Docker gate
COPY forithmus_contract_guard.py /opt/app/forithmus_contract_guard.py
RUN python -m py_compile /opt/app/process.py /opt/app/forithmus_contract_guard.py && python -c "import torch,numpy,nibabel,transformers; import process; print('TASK1 BUILD IMPORTS PASS')"

# VLM3D_TASK1_METRIC_CALIBRATION_BUILD_GUARD_V2
RUN test -f /opt/app/task1_calibrate_output.py && python -m py_compile /opt/app/task1_calibrate_output.py

USER appuser
ENTRYPOINT ["/entrypoint.sh"]
