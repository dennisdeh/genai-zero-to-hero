from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
import os


def load_documents_from_folder(folder: str):
    if not os.path.isdir(folder):
        raise NotADirectoryError(
            f"Documents folder not found: {folder}. Create it and add .pdf/.docx files."
        )
    # load all .pdf files from the folder, parsing them using PyPDFLoader
    pdf_loader = DirectoryLoader(
        path=folder,
        glob="**/*.pdf",
        recursive=True,
        loader_cls=PyPDFLoader,
        loader_kwargs={"mode": "page", "extraction_mode": "plain"},
        show_progress=True,
        use_multithreading=True,
    )
    # load all .docx files from the folder, parsing them using Docx2txtLoader
    docx_loader = DirectoryLoader(
        path=folder,
        glob="**/*.docx",
        recursive=True,
        loader_cls=Docx2txtLoader,
        loader_kwargs=None,
        show_progress=True,
        use_multithreading=True,
    )
    # combine all documents loaded into a single list
    docs = []
    docs.extend(pdf_loader.load())
    docs.extend(docx_loader.load())

    if not docs:
        raise ValueError(f"No .pdf or .docx files found under: {folder}")

    return docs
