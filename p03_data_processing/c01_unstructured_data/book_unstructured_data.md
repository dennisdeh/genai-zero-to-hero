## Unstructured Data
Unstructured data - text, images, audio, video - has no fixed schema of columns. Before it can be
analysed or used to train a model it has to be cleaned and converted into a numeric representation.
For text, which is the input of the large language models of part 7, the processing chain usually is

1. **extraction**: get the text out of HTML pages, PDFs, office documents or APIs,
2. **cleaning and normalisation**: remove markup and invisible characters, unify Unicode, mask
   personal data, remove duplicates and low-quality documents,
3. **tokenisation**: split the text into tokens and map them to integer ids,
4. **representation**: turn the ids into vectors (embeddings) inside the model.

This chapter covers steps 2 and 3 in section `s00_text`: `text_cleaning/main_text_cleaning.py` builds a
small cleaning pipeline with the Python standard library, and `tokenisation/` implements byte-pair
encoding from scratch, compares it with the Hugging Face `tokenizers` library and OpenAI's `tiktoken`,
and tokenises a real instruction dataset. Embeddings are covered in parts 6 and 7.
