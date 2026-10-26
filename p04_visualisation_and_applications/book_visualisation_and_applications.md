# Visualisation and Applications
In this part we look at some methods to present data using visualisation, i.e. creating 
applications using interactive plots.

A chart is read by people, so it has to be designed for them: it should answer one question, make the
important comparison easy and not mislead. In data science, charts serve two different purposes:
*exploration* (quick plots for ourselves, to understand data and models) and *communication*
(polished charts and applications for others, to support a decision). Interactive charts - with
tooltips, zoom and filters - bridge both: the reader can explore the details while the default view
tells the story.

The tools used in this part are

- **Plotly** (chapter `c00_plotting`): a library for interactive charts that render in the browser,
  in notebooks and in web apps. `plotly.express` creates complete charts from a data frame with one
  call; `plotly.graph_objects` gives full control over every trace and layout element. The chapter
  starts with principles of good charts and applies them to the datasets bundled with Plotly.
- **Streamlit** (chapter `c01_streamlit_app`): a framework that turns a Python script into an
  interactive web application without any HTML or JavaScript. We build a customer analytics dashboard
  with filters, KPI tiles, Plotly charts and a churn prediction panel on top of the cleaned data of
  part 3.

The conda environment for this part is `p04_visualisation_and_applications/conda_env_gzh_viz0.yml`.
All examples run offline, from the repository root:

```bash
conda env create -f p04_visualisation_and_applications/conda_env_gzh_viz0.yml
conda activate gzh_viz0
python -m p04_visualisation_and_applications.c00_plotting.main_1_plotly_express
streamlit run p04_visualisation_and_applications/c01_streamlit_app/main_app.py
```
