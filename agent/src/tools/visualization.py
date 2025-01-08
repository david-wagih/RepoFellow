import base64
import tempfile
from graphviz import Digraph
from typing import Dict, Any, List
from pathlib import Path
from models.schema import RepoFile

class RepoVisualizer:
    @staticmethod
    def generate_architecture_diagram(files: List[RepoFile]) -> Dict[str, Any]:
        """Generate architecture diagram"""
        try:
            if not files:
                raise ValueError("No files provided for diagram generation")

            dot = Digraph(comment="Repository Structure")
            dot.attr(rankdir="TB")

            # Add nodes and edges
            for file in files:
                parts = Path(file.path).parts
                for i in range(len(parts)):
                    node_id = f"node_{hash(''.join(parts[:i+1]))}"
                    dot.node(node_id, parts[i])
                    if i > 0:
                        parent_id = f"node_{hash(''.join(parts[:i]))}"
                        dot.edge(parent_id, node_id)

            # Save diagram
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                dot.render(tmp.name, format="png", cleanup=True)
                with open(f"{tmp.name}.png", "rb") as f:
                    image_data = f.read()
                image_base64 = base64.b64encode(image_data).decode("utf-8")

            return {
                "image_base64": image_base64,
                "image_format": "png",
            }

        except Exception as e:
            return {"error": str(e)} 