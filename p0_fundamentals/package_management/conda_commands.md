## Useful conda commands
A list of useful conda commands.
Texts in capital enclosed by curly brackets are placeholders.
Notice that all conda environment names are prefixed with `gzh_` and corresponding 
yaml-files with `conda_env_gzh_`, which can of course be omitted.


#### save conda environment to file
`conda env export > conda_env_gzh_{NEW}.yml`

#### update conda environment from file
`conda env update --prefix ./env --file conda_env_{NEW}.yml  --prune`

#### create a conda environment from an environment file
`conda env create -f gzh_base_env.yml`

#### create a new base conda environment w. python
`conda create --name gzh_{NEW} python=3.12 --channel conda-forge --override-channels`

`conda create --name gzh_{NEW} python=3.13`

#### clone an existing conda environment
`conda create --name gzh_{NEW} --clone gzh_{OLD}`

#### clean the conda folder
`conda clean -p`

#### delete a conda environment
`conda env remove -n gzh_{NEW}`

`conda remove -p PATH_to_env --all`

#### change the location of conda environment files
[See this guide](https://conda.io/projects/conda/en/latest/user-guide/configuration/use-condarc.html#specify-environment-directories-envs-dirs)

#### set the download channel to conda-forge per default
`conda activate gzh_{NEW}`

`conda config --env --add channels conda-forge`

`conda config --env --remove channels defaults`

`conda config --env --set channel_priority strict`

