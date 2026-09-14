
from docutils import nodes
from myst_parser.parsers.sphinx_ import MystParser
from sphinx.ext.napoleon.docstring import GoogleDocstring


class NapoleonParser(MystParser):
    def parse(self, input_string: str, document: nodes.document) -> None:
        # Get the Sphinx configuration
        config = document.settings.env.config

        # Process with Google style
        google_parsed = str(GoogleDocstring(input_string, config))

        return super().parse(google_parsed, document)


Parser = NapoleonParser
