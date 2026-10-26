# Practical Generative AI – From Zero to Hero
This repository contains a collection of useful resources for practitioners, including learning material, workflows, code snippets.

The idea is to provide a comprehensive overview on the state-of-the-art in machine learning and GenAI, and to be self-contained.
The starting point is all fundamentals needed to efficiently work with ML/GenAI.
From thereon, the focus shifts to more advanced topics.
The philosophy is that the focus should be on the practical aspects of ML/GenAI, i.e. *learning by doing*.
As such, all necessary files are provided in this repository, with a clear structure and dependencies between parts that are clearly defined.

The repository is structured as a book, with parts, chapters and sections.
Since this is about programming and the philosophy is "hands-on", I deemed that this is the best approach.
Like this the code can be easily executed, and it is easy to follow the documentation, 
which is always found in the respective folders in the files that start with `book_` (e.g. `book_fundamentals.md`).
However, the book can also be compiled to a PDF for easy reading and reference (see below).

The topics intended to be covered in this repository are:
- Fundamentals: Technical basics, setup, mathematical foundations, best-practices
- Containerisation: Indispensable tools like Docker and Kubernetes are covered
- Data sourcing: How to collect and store data, including relevant tools for various data types
- Data processing: How to work with and process structured and unstructured data
- Visualisation: Presenting data and objects in a clear and understandable way
- Machine learning (classic): Data preprocessing, data quality, modelling techniques, evaluation, validation, deployment
- Neural networks: Various architectures, data processing, deployment
- Large language models (LLMs): Transformers and related architectures, GPT-like models, training, deployment, reinforcement learning
- Agents: Agentic AI frameworks, workflows, tool calling, human-in-the-loop, guardrails

*The repository is dynamic and constantly evolving. The structure is not yet finalised.*

## Structure
### Parts, chapters and sections
On the highest level, all topics are grouped into folders, one for each part.
Each part has an associated folder indicated by the prefix `p{nm}_` where `nm` in the placeholder is 
a 2-digit number that indicates the order of the part within the repository.
Currently, the parts and their chapters are:
- **p00_fundamentals**: best practices, package management (conda), FastAPI model-serving service
- **p01_containerisation_and_cloud_deployment**: Docker and docker compose, Kubernetes, cloud deployment and CI/CD
- **p02_data_sourcing**: data storage formats, Kaggle, OpenML, Hugging Face datasets, web scraping, scheduling, REST APIs
- **p03_data_processing**: preprocessing of structured data, data quality assurance and drift, text cleaning, tokenisation (BPE from scratch)
- **p04_visualisation_and_applications**: Plotly charts, Streamlit dashboard application
- **p05_ml_classic**: linear models, gradient-boosted trees, model explainability (SHAP, PDP), MLOps with MLflow
- **p06_neural_networks**: MLPs, CNNs, GANs, GNNs (PyTorch, PyTorch Geometric)
- **p07_llms**: GPT-like models, running LLMs, prompt engineering and structured outputs, fine-tuning (PEFT, GRPO), RAG systems, long context systems, workflows, validation
- **p08_agents**: LangChain/LangGraph agents, tools, human-in-the-loop, multi-agent supervisor and swarm, memory

The design decisions behind the examples (datasets, frameworks, tooling) are recorded in `implementation_options.md`.
Most examples outside p07/p08 run offline on a CPU; the LLM examples expect a local Ollama server
(see `p07_llms/c01_running_llms/s01_ollama/docker-compose.yml`) and some fine-tuning examples need a GPU.

The parts are furthermore divided into chapters (i.e. have the prefix `c{nm}_`) and potentially also sections (i.e. have the prefix `s{nm}_`).
This is to ensure that the structure is easily navigable and that the dependencies between parts are clear.

There are (sub)repositories found within the main repository. These are used to store the code snippets and workflows.

### Material
There are five types of resources:
- **Book chapters/sections:** Contains relevant chapters or sections in Markdown files that all start with a `book_` prefix. The entire book can be compiled as a pdf from the `build_book_pdf.sh` in the root directory. 
- **Workflows:** Complete examples related to standard ML/GenAI workflows. The file called `main` (e.g. `main.py`) is always the principal file to run for each part/chapter/section. Numbered files (`main_1_...`, `main_2_...`) are meant to be run in order. Run them from the repository root, e.g. `python -m p05_ml_classic.c00_linear_models.main_1_linear_regression`; generated files are written to an `outputs/` folder next to the script (ignored by git).
- **Utils:** Small useful helpers for specialised topics. The files with are in the `utils` subfolder.
- **conda environments:** Conda environments exported to `.yml` files for each part/chapter/section have the prefix `conda_env`.
- **README.md files:** The README.md file in each chapter folder contains a short description of how to set up the environment and how to run the code.

## How to generate the book as a PDF
The book can be generated by running the `build_book_pdf.sh` script in the root directory.
The following steps are required (on Linux):
1. install pandoc and texlive by running: `sudo apt update && sudo apt install -y pandoc texlive`
2. navigate to the repository root
3. make the script executable by running: `chmod +x build_book_pdf.sh`
4. generate the book pdf by running: `./build_book_pdf.sh`

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
