ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE}
COPY src/tensorfold /usr/local/lib/python3.12/dist-packages/tensorfold
RUN pip install --no-cache-dir "transformers>=5.3,<6" huggingface_hub anyio click filelock fsspec
ENV TENSORFOLD_NO_UPDATE_CHECK=1
