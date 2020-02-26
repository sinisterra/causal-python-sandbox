FROM divkal/cdt-py3.7:latest

RUN  apt-get install python-dev graphviz libgraphviz-dev pkg-config -y
RUN pip3 install pygraphviz ray
WORKDIR /code
CMD ["/bin/bash"]