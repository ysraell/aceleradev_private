FROM python:3.7-buster

RUN apt-get update && apt-get install -y \
  wget \
  nginx \
  ca-certificates \
  htop \
  vim \
  jed \
  nano \
  zsh \
  npm && rm -rf /var/cache/apt && rm -rf /var/lib/apt/lists/*

# terminal colors with xterm
ENV TERM xterm
# set the zsh theme
ENV ZSH_THEME agnoster
# run the installation script  
RUN wget https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O - | zsh || true

RUN pip install -U pip --no-cache-dir
COPY /requirements.txt /requirements.txt
RUN pip install -r requirements.txt --no-cache-d
#Set environment for AceleraDev
RUN curl https://s3-us-west-1.amazonaws.com/codenation-cli/latest/codenation_linux.tar.gz | tar xvz
RUN mv codenation /usr/local/bin
#RUN codenation config -t 
#For test: cd /work/aceleradev_private/data-science-0  ; codenation test -c data-science-0


#Using Eigen from Tuxfamily
#RUN wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.bz2 && tar xfjv eigen-3.3.7.tar.bz2 && rm eigen-3.3.7.tar.bz2
#RUN mv eigen-3.3.7 /usr/local/

# Mount point of your $HOME
RUN mkdir /work

# All servers need be in:
COPY /servers.sh /servers.sh
WORKDIR /
EXPOSE 8889
CMD ["/servers.sh"]
