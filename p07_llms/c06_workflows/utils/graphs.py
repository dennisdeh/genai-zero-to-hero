"""
Helpers to visualise LangGraph workflows.

`draw_mermaid()` works offline and returns Mermaid text (paste it into https://mermaid.live or a Markdown
viewer that renders Mermaid). `draw_mermaid_png()` calls the mermaid.ink web service by default and therefore
needs internet access - it is wrapped in try/except so that the examples never fail because of a drawing.
"""

import os

from langgraph.graph.state import CompiledStateGraph

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


def save_graph(graph: CompiledStateGraph, name: str, png: bool = True) -> None:
    """
    Save the graph structure as Mermaid text (always) and as PNG (if the renderer is reachable) in outputs/.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    drawable = graph.get_graph()
    mermaid_path = os.path.join(OUTPUT_DIR, f"{name}.mmd")
    with open(mermaid_path, "w") as f:
        f.write(drawable.draw_mermaid())
    print(f"Graph structure saved to {mermaid_path}")
    if not png:
        return
    try:
        png_bytes = drawable.draw_mermaid_png()
    except Exception as e:  # no internet access, renderer down, ...
        print(
            f"(Could not render PNG: {type(e).__name__}; the .mmd file can be viewed at https://mermaid.live)"
        )
        return
    png_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    with open(png_path, "wb") as f:
        f.write(png_bytes)
    print(f"Graph image saved to {png_path}")
