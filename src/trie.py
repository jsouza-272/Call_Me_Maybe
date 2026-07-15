"""Simple trie structure for constrained token decoding."""


class Trie:
    """Trie node used to represent valid token paths."""

    def __init__(self, token_id: int) -> None:
        self.children: dict[int, "Trie"] = {}
        self.id = token_id
        self.is_end = False

    def add_child(self, child: "Trie") -> None:
        """Attach a child node if it does not already exist."""

        if child.id not in self.children:
            self.children[child.id] = child
