FROM vllm/vllm-openai:latest

ENV HF_HOME=/root/.cache/huggingface

EXPOSE 8000

ENTRYPOINT ["vllm", "serve"]
CMD ["Qwen/Qwen2.5-0.5B-Instruct", "--host", "0.0.0.0", "--port", "8000"]
