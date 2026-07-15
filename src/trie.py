class Trie:
    """Store token transitions used to constrain model decoding."""

    def __init__(self, id: int) -> None:
        """Initialize a trie node.

        Args:
            id: Token identifier represented by this node.
        """
        self.children: dict = {}
        self.id = id
        self.is_end = False

    def add_children(self, children: "Trie") -> None:
        """Attach a child node if its token id is not present yet.

        Args:
            children: Node to add as a child.
        """
        if children.id in self.children:
            return
        self.children[children.id] = children
