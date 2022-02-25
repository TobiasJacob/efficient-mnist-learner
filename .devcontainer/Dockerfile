ARG VARIANT="22.01"
FROM nvcr.io/nvidia/pytorch:${VARIANT}-py3

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends nodejs

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN useradd -ms /bin/bash vscode
RUN chown vscode /opt/conda
USER vscode
