from importlib.metadata import PackageNotFoundError, version

from antlr4.atn.ATNDeserializer import SERIALIZED_VERSION

try:
    antlr_version = version("antlr4-python3-runtime")
except PackageNotFoundError:
    antlr_version = ""

# importlib.metadata can report a different distribution from the antlr4 package
# that is actually imported when vendor_python and uv/conda site-packages are
# mixed on PYTHONPATH. Pick generated code by the runtime's serialized ATN
# format first; otherwise ANTLR raises "version 3 (expected 4)" at import time.
if SERIALIZED_VERSION == 4:
    if antlr_version.startswith("4.11"):
        from latex2sympy2_extended.gen.antlr4_11_0.PSParser import PSParser
        from latex2sympy2_extended.gen.antlr4_11_0.PSLexer import PSLexer
    else:
        from latex2sympy2_extended.gen.antlr4_13_2.PSParser import PSParser
        from latex2sympy2_extended.gen.antlr4_13_2.PSLexer import PSLexer
elif SERIALIZED_VERSION == 3:
    from latex2sympy2_extended.gen.antlr4_9_3.PSParser import PSParser
    from latex2sympy2_extended.gen.antlr4_9_3.PSLexer import PSLexer
else:
    raise ImportError(
        f"Unsupported ANTLR serialized ATN version {SERIALIZED_VERSION} "
        f"from runtime distribution {antlr_version!r}."
    )
