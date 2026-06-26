class Trie:
    def __init__(self, id: int):
        self.children = {}
        self.id = id
        self.is_end = False

    def add_children(self, children: "Trie"):
        if children.id in self.children:
            return
        self.children[children.id] = children
