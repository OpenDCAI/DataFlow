import networkx as nx
from dataflow.pipeline.nodes import KeyNode
import matplotlib.pyplot as plt

# 创建节点
a = KeyNode("fuck", "fucing")
b = KeyNode("dick", "dicking")

# 创建有向图并添加节点
G = nx.DiGraph()
G.add_nodes_from([a, b])

# 添加有向边
G.add_edge(a, b)

# 设置边的属性
nx.set_edge_attributes(G, { (a, b): {'label': 'edge_label', 'content': 'edge_content'} })

# 使用 spring_layout 自动布局节点位置
pos = nx.spring_layout(G)

# 设置画布大小
plt.figure(figsize=(10, 8))

labels = {
    a: 'Node A',
    b: 'Node B'
}

# 绘制图形
nx.draw(G, pos, labels=labels, with_labels=True, node_size=3000, node_shape='s', node_color='lightblue', edge_color='gray', arrows=True)



# 绘制边的标签
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# 保存图形
plt.savefig("petersen.png")
plt.show()
