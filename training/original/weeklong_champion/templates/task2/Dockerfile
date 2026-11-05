FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
WORKDIR /opt/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY model_code_from_original.py vlm3d_common.py sota_models.py process.py ./
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && useradd -m appuser && chown -R appuser:appuser /opt/app
ENV PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
USER appuser
ENTRYPOINT ["/entrypoint.sh"]
