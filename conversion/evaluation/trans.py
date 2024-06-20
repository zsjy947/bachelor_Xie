import json

# 读取JSON文件
with open('data.json', 'r') as file:
    data = json.load(file)

# 替换特定列中的字符串
for item in data:
    item['city'] = item['city'].replace("New York", "NY")

# 输出修改后的数据查看结果
print(data)

# 可选：将修改后的数据写回文件
with open('data_modified.json', 'w') as file:
    json.dump(data, file, indent=4)
