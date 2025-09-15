import os

# Optional imports
try:
    from pypdf import PdfReader
except ImportError:
    print("pypdf is not installed. Please install it using `uv add pypdf`")
    PdfReader = None

try:
    import tiktoken
except ImportError:
    print("tiktoken is not installed. Please install it using `uv add tiktoken`")
    tiktoken = None


class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".txt"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .txt file."
            )

    def load_file(self):
        with open(self.path, encoding=self.encoding) as f:
            self.documents.append(f.read())

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    with open(os.path.join(root, file), encoding=self.encoding) as f:
                        self.documents.append(f.read())

    def load_documents(self):
        self.load()
        return self.documents


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: list[str]) -> list[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks


class PDFLoader:
    def __init__(self, path: str):
        """
        Load one PDF file or a directory of PDF files
        """

        self.path = path
        self.documents = []

        if PdfReader is None:
            raise ImportError(
                "pypdf is not installed. Please install it using `uv add pypdf`"
            )

    def _load_pdf_file(self, file_path: str) -> str:
        reader = PdfReader(file_path)

        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text())
            except Exception:
                pages.append("")

        return "\n".join(pages)

    def load(self) -> list[str]:
        if os.path.isdir(self.path):
            for root, _, files in os.walk(self.path):
                for f in files:
                    if f.lower().endswith(".pdf"):
                        fp = os.path.join(root, f)
                        self.documents.append(self._load_pdf_file(fp))
        elif os.path.isfile(self.path) and self.path.lower().endswith(".pdf"):
            self.documents.append(self._load_pdf_file(self.path))
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .pdf file."
            )

        return self.documents


class TokenCounter:
    def __init__(self, model_name: str = "gpt-4.1-mini"):
        """
        Estimates token counts using tiktoken when available.
        For OpenAI 4.x/o-series models, uses 'o200k_base'; fallback to 'cl100k_base'.
        """
        self.model_name = model_name
        self._encoding = None

        if tiktoken is not None:
            # Choose encoding based on model
            if any(x in model_name.lower() for x in ["4.1", "4o", "o-", "gpt-4o"]):
                enc_name = "o200k_base"
            else:
                enc_name = "cl100k_base"

            try:
                self._encoding = tiktoken.get_encoding(enc_name)
            except Exception:
                try:
                    self._encoding = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    self._encoding = None

    def count(self, text: str) -> int:
        """Count tokens in text using tiktoken encoding."""
        if self._encoding is not None:
            try:
                return len(self._encoding.encode(text))
            except Exception:
                return 0

    def count_messages(self, messages: list[dict]) -> int:
        """Count total tokens across a list of message dictionaries."""
        total = 0
        for m in messages:
            total += self.count(m.get("content", ""))
        return total


if __name__ == "__main__":
    loader = TextFileLoader("data/KingLear.txt")
    loader.load()
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(loader.documents)
    print(len(chunks))
    print(chunks[0])
    print("--------")
    print(chunks[1])
    print("--------")
    print(chunks[-2])
    print("--------")
    print(chunks[-1])
