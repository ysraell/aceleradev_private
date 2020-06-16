FROM python:3.7-buster

RUN apt-get update && apt-get install -y \
  wget \
  nginx \
  ca-certificates \
  htop \
  vim \
  jed \
  nano \
  nodejs \
  zsh \
  mtr \
  whois \
  npm && rm -rf /var/cache/apt && rm -rf /var/lib/apt/lists/*

# terminal colors with xterm
ENV TERM xterm
# set the zsh theme
ENV ZSH_THEME agnoster
# run the installation script  
RUN wget https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O - | zsh || true

RUN pip install -U pip --no-cache-dir
COPY /requirements.txt /requirements.txt
RUN pip install -r requirements.txt --no-cache-dir
# For modules to other things
# For run https://github.com/rasbt/deeplearning-models
# Streamlit needs that:
RUN pip install streamlit --no-cache-dir

RUN jupyter nbextension enable --py --sys-prefix ipysankeywidget
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension
RUN jupyter serverextension enable --py jupyterlab --sys-prefix
RUN jupyter labextension install jupyterlab_templates
RUN jupyter notebook --generate-config
COPY /jupyterlab_templates_config.py /jupyterlab_templates_config.py
RUN cat /jupyterlab_templates_config.py >>/root/.jupyter/jupyter_notebook_config.py
COPY /JupyterTemplates/DS/*.ipynb /JupyterTemplates/DS/
RUN jupyter serverextension enable --py jupyterlab_templates


#Set environment for AceleraDev
RUN curl https://s3-us-west-1.amazonaws.com/codenation-cli/latest/codenation_linux.tar.gz | tar xvz
RUN mv codenation /usr/local/bin
RUN codenation config -t 2816bb242252e72b8f6adfec61272c2892405e07
#For test: cd /work/aceleradev_private/data-science-0  ; codenation test -c data-science-0

RUN pip install pytest

# Mount point of your $HOME
RUN mkdir /work

# All servers need be in:
COPY /servers.sh /servers.sh
WORKDIR /
EXPOSE 8888 8501
CMD ["/servers.sh"]
