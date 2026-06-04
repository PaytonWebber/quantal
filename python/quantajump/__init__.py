"""quantajump — a two-stage cascading vector index (1-bit graph routing,
3-bit TurboQuant payloads, exact rerank), embedded and in-memory.

    from quantajump import Index
    index = Index(dim=384)

LangChain users: `from quantajump.langchain import QuantajumpVectorStore`.
"""

from .index import Index

__all__ = ["Index"]
__version__ = "0.1.0"
