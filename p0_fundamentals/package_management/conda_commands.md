## Useful conda commands
A list of useful conda commands.

### save conda environment environment to file
conda env export > gzh_base_env.yml
conda env export > gzh_{NEW}_env.yml

### update conda environment from file
conda env update --prefix ./env --file gzh_base_env.yml  --prune

### create conda environment from file
conda env create -f gzh_base_env.yml

### create new conda environment w. python
conda create --name gzh_{NEW} python=3.12 --channel conda-forge --override-channels
mamba create --name gzh_{NEW} python=3.13

### clone
conda create --name gzh_{NEW} --clone gzh_{OLD}

### clean conda folder
conda clean -p

### delete conda environment
conda env remove -n gzh_{NEW}
conda remove -p PATH_to_env --all

### change location
[See this guide](https://conda.io/projects/conda/en/latest/user-guide/configuration/use-condarc.html#specify-environment-directories-envs-dirs)

### set channel to conda-forge per default
conda activate gzh_{NEW}
conda config --env --add channels conda-forge
conda config --env --remove channels defaults
conda config --env --set channel_priority strict

