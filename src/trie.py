class Trie:
    def __init__(self, id: int) -> None:
        self.children: dict = {}
        self.id = id
        self.is_end = False

    def add_children(self, children: "Trie") -> None:
        if children.id in self.children:
            return
        self.children[children.id] = children
