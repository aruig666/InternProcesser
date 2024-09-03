import json

selfdile = 'self.json'
dataset = []
selfdata = 0
with open(selfdile, 'r', encoding='utf-8') as file:
    selfdata = json.loads(file.read())
n = 1000
for i in range(n):
    dataset.extend(selfdata)
savename = "self_fineturn.json"
with open(savename, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)
print(len(dataset))
