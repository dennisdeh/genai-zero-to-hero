"""
Helpers to visualise LangGraph graphs.

A compiled LangGraph graph can describe its own structure: `graph.get_graph()` returns a drawable
representation with three useful methods:

- `draw_mermaid()` returns the graph as Mermaid text. It works offline; the text can be pasted into
  https://mermaid.live or any Markdown viewer that renders Mermaid diagrams.
- `draw_mermaid_png()` renders the Mermaid diagram to PNG bytes. By default it calls the mermaid.ink
  web service and therefore needs internet access.
- `draw_ascii()` prints the graph as ASCII art in the terminal (needs the optional `grandalf` package).

`save_graph_image` tries the PNG first and falls back to the Mermaid text (and ASCII art, if possible)
so that an example never fails just because the drawing could not be rendered.

Example:
    from p08_agents.c00_langchain_langgraph.s00_very_simple_graphs.utils.graphs import save_graph_image

    agent = graph.compile()
    save_graph_image(agent, "path/to/outputs/graph.png")
"""

import os
from typing import Any


def _get_drawable(graph: Any) -> Any:
    """
    Return the drawable graph: compiled graphs (and other runnables) expose `get_graph()`,
    an already drawable `langchain_core.runnables.graph.Graph` is returned unchanged.
    """
    return graph.get_graph() if hasattr(graph, "get_graph") else graph


def save_graph_image(graph: Any, path: str, ascii_fallback: bool = True) -> str:
    """
    Save a picture of a LangGraph graph as PNG; fall back to Mermaid text when offline.

    Args:
        graph: A compiled LangGraph graph (anything with `get_graph()`) or a drawable graph.
        path: Target path of the PNG file; missing folders are created.
        ascii_fallback: If the PNG cannot be rendered, also print the graph as ASCII art
            (requires the optional `grandalf` package; silently skipped otherwise).

    Returns:
        The path of the file that was written: the PNG, or a `.mmd` file with the Mermaid text.
    """
    # 1. Prepare the drawable graph and the output folder
    drawable = _get_drawable(graph)
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)

    # 2. Try to render the PNG (uses the mermaid.ink web service by default)
    try:
        png_bytes = drawable.draw_mermaid_png()
        with open(path, "wb") as f:
            f.write(png_bytes)
        print(f"Graph image saved to {path}")
        return path
    except Exception as e:  # no internet access, renderer unavailable, ...
        print(
            f"Could not render the graph as PNG ({type(e).__name__}); "
            "saving the Mermaid text instead."
        )

    # 3. Fallback: the Mermaid text always works offline
    mermaid_path = os.path.splitext(path)[0] + ".mmd"
    with open(mermaid_path, "w", encoding="utf-8") as f:
        f.write(drawable.draw_mermaid())
    print(f"Mermaid graph saved to {mermaid_path} (view it at https://mermaid.live)")

    # 4. Optionally show the graph as ASCII art in the terminal
    if ascii_fallback:
        try:
            print(drawable.draw_ascii())
        except Exception:  # grandalf not installed
            pass
    return mermaid_path
