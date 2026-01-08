# Practical Generative AI – From Zero to Hero
This repository contains a collection of useful resources for practitioners, including learning material, workflows, code snippets.

The idea is to provide a comprehensive overview of the state-of-the-art in machine learning and GenAI, and to be self-contained.
The starting point is all fundamentals needed to efficiently work with ML/GenAI.
From thereon, the focus shifts to more advanced topics.
The philosophy is that the focus should be on the practical aspects of ML/GenAI, i.e. *learning by doing*.
As such, all necessary files are provided in this repository, with a clear structure and dependencies between parts that are clearly defined.

The topics intended to be covered in this repository are:
- Fundamentals: Technical basics, setup, mathematical foundations, best-practices
- Containerisation: Indispensable tools like Docker and Kubernetes are covered
- Data sourcing: How to collect and store data, including relevant tools for various data types
- Data processing: How to work with and process structured and unstructured data
- Visualisation: Presenting data and objects in a clear and understandable way
- Machine learning (classic): Data preprocessing, data quality, modelling techniques, evaluation, validation, deployment
- Neural networks: Various architectures, data processing, deployment
- Large language models (LLMs): Transformers and related architectures, GPT-like models, training, deployment
- Agents: Reinforcement learning, decision-making, planning

*The repository is dynamic and constantly evolving. The structure is not yet finalised.*

## Structure
### Parts, chapters and sections
On the highest level, all topics are grouped into folders, one for each part.
Each part has an associated folder indicated by the prefix `p{nm}_` where `nm` in the placeholder is 
a 2-digit number that indicates the order of the part within the repository.
Currently, the parts are:
- p00_fundamentals
- p01_containerisation
- p02_data_sourcing
- p03_data_processing
- p04_visualisation
- p05_ml_classic
- p06_neural_networks
- p07_llms
- p08_agents

The parts are furthermore divided into chapters (i.e. have the prefix `c{nm}_`) and potentially also sections (i.e. have the prefix `s{nm}_`).
This is to ensure that the structure is easily navigable and that the dependencies between parts are clear.

There are (sub)repositories found within the main repository. These are used to store the code snippets and workflows.

### Material
There are three types of resources:
- **Book chapters/sections:** Contains relevant chapters or sections in LaTeX files that all start with a `book_` prefix. The entire book can be compiled from the `book.tex` in the root directory. 
- **Workflows:** Complete examples related to standard ML/GenAI workflows. The file called `main` (e.g. `main.py`) is always the principal file to run for each part/chapter/section.
- **Code snippets:** Small examples to demonstrate specialised topics. The file called `main` (e.g. `main.py`) is always the principal file to run for each part/chapter/section.



## References and Acknowledgements
It is important for me to give credit where credit is due as I am certainly not reinventing the wheel in many cases.
Therefore, all resources are attributed to their respective authors,
and I mark this clearly in the respective files and folders following best practices.

### Ways that this is achieved
- All files where I am not the original owner have a header that clearly states the original author and what modifications I have made. 
- All (external) repositories have a README.MD file, with an added Acknowledgements section that lists all relevant sources and original authors. 
- For each chapter, a `sources.md` file lists all relevant sources 

## Contributions
Contributions are welcome - submit a pull request - and remember to credit yourself!
