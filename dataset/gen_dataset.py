import json
import pathlib

from tqdm import tqdm

dirname = pathlib.Path(r"G:\dataset\NSMPDataset\step")
jsonfiles = list(dirname.rglob("*.json"))
promptdir = "./process_lang.txt"
processprompt = 0
with open(promptdir, 'r', encoding='utf-8') as file:
    processprompt = file.read()
    # print(processprompt)
dataset = []
for jsonfile in tqdm(jsonfiles, total=len(jsonfiles)):
    data = 0
    with open(jsonfile, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # print(data)
    dataset.append({"conversation": [
        {
            "system": processprompt,
            "input": data['input1'],
            "output": data['output1']
        }, {
            "system": "",
            "input": data['input2'] + data['system'],
            "output": data['output2']
        }
    ]})
selfdile = './self.json'
selfdata = 0
with open(selfdile, 'r', encoding='utf-8') as file:
    selfdata = json.loads(file.read())
commfile = './common.json'
commdata = 0
with open(commfile, 'r', encoding='utf-8') as file:
    commdata = json.loads(file.read())

n = 1000
for i in range(n):
    dataset.extend(selfdata)
    commdata[0]['conversation'][0]['system'] = processprompt
    dataset.extend(commdata)

savename = "./fineturn.json"
with open(savename, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)
print(len(dataset))
