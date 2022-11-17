Installation of Environment Specific Dependencies:
```
conda install -c conda-forge openbabel pytdc
pip install matplotlib global-chem
git clone https://github.com/dockstring/dockstring.git && cd dockstring && pip install .
git clone https://github.com/aspuru-guzik-group/group-selfies.git && cd group-selfies && pip install .
pip install flask requests

RUN pip install selfies
RUN pip install SmilesPE
```