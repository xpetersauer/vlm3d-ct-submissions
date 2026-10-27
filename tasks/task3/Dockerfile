FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /opt/app

COPY requirements.txt /opt/app/requirements.txt
RUN pip install --no-cache-dir -r /opt/app/requirements.txt

COPY model_code_from_original.py /opt/app/model_code_from_original.py
COPY vlm3d_common.py /opt/app/vlm3d_common.py
COPY forithmus_validate_outputs.py /opt/app/forithmus_validate_outputs.py
COPY task3_report_to_ct.py /opt/app/task3_report_to_ct.py
COPY process.py /opt/app/process.py
COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh && useradd -m appuser && chown -R appuser:appuser /opt/app

ENV PYTHONUNBUFFERED=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV FORITHMUS_INPUT=/input
ENV FORITHMUS_OUTPUT=/output

# TASK3_OFFLINE_DISTILBERT_CACHE_V1
ENV HF_HOME=/opt/hf-cache
ENV TRANSFORMERS_CACHE=/opt/hf-cache/transformers

RUN mkdir -p /opt/hf-cache && \
    HF_HOME=/opt/hf-cache \
    HF_HUB_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=0 \
    python -c "from transformers import AutoModel,AutoTokenizer; name='distilbert-base-uncased'; AutoModel.from_pretrained(name); AutoTokenizer.from_pretrained(name); print('Downloaded',name)" && \
    chmod -R a+rX /opt/hf-cache

RUN HF_HOME=/opt/hf-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    python -c "from transformers import AutoModel,AutoTokenizer; name='distilbert-base-uncased'; m=AutoModel.from_pretrained(name,local_files_only=True); t=AutoTokenizer.from_pretrained(name,local_files_only=True); print('TASK3 DISTILBERT OFFLINE PASS',m.config.hidden_size,len(t))"

# VLM3D_FORITHMUS_STANDALONE_MAC_V4: task3 Docker gate
COPY forithmus_contract_guard.py /opt/app/forithmus_contract_guard.py
ENV HF_HOME=/opt/app/hf_cache \
    HUGGINGFACE_HUB_CACHE=/opt/app/hf_cache/hub \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1
RUN TRANSFORMERS_OFFLINE=0 HF_HUB_OFFLINE=0 python -c "from transformers import AutoModel,AutoTokenizer; AutoTokenizer.from_pretrained('distilbert-base-uncased'); AutoModel.from_pretrained('distilbert-base-uncased'); print('TASK3 DISTILBERT CACHE PASS')"
RUN python -m py_compile /opt/app/process.py /opt/app/task3_report_to_ct.py /opt/app/model_code_from_original.py /opt/app/forithmus_contract_guard.py && python -c "import torch,numpy,nibabel,matplotlib,transformers; import model_code_from_original,task3_report_to_ct; print('TASK3 BUILD IMPORTS PASS')"

# VLM3D_NINJA_PLATFORM_REPAIR_V2
RUN python -m pip uninstall -y ninja || true; python -c "import pathlib,site,shutil; roots={pathlib.Path(p) for p in [*site.getsitepackages(),site.getusersitepackages()] if p}; stale=[p for r in roots if r.exists() for p in r.iterdir() if p.name.lower().startswith('ninja-') and (p.name.lower().endswith('.dist-info') or p.name.lower().endswith('.egg-info'))]; print('Removing stale Ninja metadata:',*[str(p) for p in stale]); [shutil.rmtree(p) if p.is_dir() else p.unlink() for p in stale]"; python -m pip install --no-cache-dir --only-binary=:all: ninja==1.11.1.4; python -c "import importlib.metadata as m,platform,shutil,subprocess; ds=[d for d in m.distributions() if (d.metadata.get('Name') or '').lower()=='ninja']; assert len(ds)==1,[(d.version,str(d._path)) for d in ds]; assert ds[0].version=='1.11.1.4',ds[0].version; p=shutil.which('ninja'); assert p; print('NINJA PLATFORM REPAIR PASS',platform.machine(),ds[0].version,p,subprocess.check_output([p,'--version'],text=True).strip())"

# VLM3D_CROSS_TASK_DEPENDENCY_PARITY_V1_TASK3
RUN python -m pip check && python -c "import torch,torchvision,numpy,pandas,matplotlib,nibabel,einops,transformers,huggingface_hub,hf_transfer; import model_code_from_original as m,task3_report_to_ct; assert m.AutoTokenizer is not None and m.AutoModel is not None; print('TASK3 CROSS-TASK DEPENDENCY PARITY PASS')"

USER appuser
ENTRYPOINT ["/entrypoint.sh"]
