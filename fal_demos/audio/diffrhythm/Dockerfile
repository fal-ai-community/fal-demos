FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Install system dependencies.
RUN apt-get update && \
    apt-get install -y git espeak-ng ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies.
RUN pip install accelerate==1.4.0 \
    inflect==7.5.0 \
    torchdiffeq==0.2.5 \
    torchaudio==2.6.0 \
    x-transformers==2.1.2 \
    transformers==4.49.0 \
    numba==0.61.0 \
    llvmlite==0.44.0 \
    librosa==0.10.2.post1 \
    pyarrow==19.0.1 \
    pandas==2.2.3 \
    pylance==0.23.2 \
    ema-pytorch==0.7.7 \
    prefigure==0.0.10 \
    bitsandbytes==0.45.3 \
    muq==0.1.0 \
    mutagen==1.47.0 \
    pyopenjtalk==0.4.1 \
    pykakasi==2.3.0 \
    jieba==0.42.1 \
    cn2an==0.5.23 \
    pypinyin==0.53.0 \
    onnxruntime==1.20.1 \
    Unidecode==1.3.8 \
    phonemizer==3.3.0 \
    liger_kernel==0.5.4 \
    openai==1.65.2 \
    pydantic==2.10.6 \
    einops==0.8.1 \
    lazy_loader==0.4 \
    scipy==1.15.2 \
    ftfy==6.3.1 \
    torchdiffeq==0.2.5 \
    ffmpeg-python \
    LangSegment==0.2.0 \
    hydra-core==1.3.2
