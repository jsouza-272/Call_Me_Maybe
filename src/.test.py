import json

test = [{"prompt": "casas", "rua": "lua"}]

with open("test.json", "x+") as file:
    json.dump(test, file, ensure_ascii=False, indent=4)
