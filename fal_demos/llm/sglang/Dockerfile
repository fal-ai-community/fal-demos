FROM lmsysorg/sglang:latest

ENV HF_HOME=/root/.cache/huggingface

EXPOSE 30000

ENTRYPOINT ["python3", "-m", "sglang.launch_server"]
CMD ["--model-path", "Qwen/Qwen2.5-0.5B-Instruct", "--host", "0.0.0.0", "--port", "30000"]
