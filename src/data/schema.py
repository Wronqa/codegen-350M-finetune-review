from dataclasses import dataclass


@dataclass
class RawRecord:
    """Minimal raw schema we expect from any fetcher (source-specific keys allowed)."""
    source: str
    code: str
    comment: str


@dataclass
class PreparedRecord:
    """Unified schema for training."""
    prompt: str
    response: str

    @staticmethod
    def build_prompt(code: str) -> str:
        return (
            "You are a code reviewer. Give 1 short, concrete suggestion to improve the change.\n"
            f"Code:\n{code}\nSuggestion:\n"
        )
