# -*- coding:utf-8 -*-
from flask import Flask, render_template, request, flash
import json

app = Flask(__name__)
app.secret_key = 'lisenzzz'


@app.route('/')
def hello_world():
    return render_template('index.html')


# 选择单个影响力最大的种子基于ic模型（每个节点模拟一次）
@app.route('/basicIc1')
def basic_ic_1():
    # file = open('static/data/test.txt', 'r')
    # graph_data = file.read()
    # file.close()
    import random
    import json
    # 读取数据
    networkTemp = []
    networkFile = open('static/data/Wiki.txt', 'r')
    # 设置节点数
    number_of_nodes = 105

    for line in networkFile.readlines():
        linePiece = line.split()
        networkTemp.append([int(linePiece[0]), int(linePiece[1])])
    # 初始化权重矩阵
    networkWeight = []
    for i in range(3):
        networkWeight.append([])
        for j in range(number_of_nodes):
            networkWeight[i].append([])
            for k in range(number_of_nodes):
                networkWeight[i][j].append(0)
    # 设置权重
    # 边的权重有三种设置方式，一种是从[0.1, 0.01, 0.001]中随机选一个，一种是都固定0.1，一种是节点的入度分之一
    probability_list = [0.1, 0.01, 0.001]
    for linePiece in networkTemp:
        networkWeight[0][linePiece[0] - 1][linePiece[1] - 1] = random.choice(probability_list)
        networkWeight[1][linePiece[0] - 1][linePiece[1] - 1] = 0.1
    for node in range(number_of_nodes):
        degree = 0
        for iteration in range(number_of_nodes):
            if iteration != node:
                if networkWeight[1][iteration][node]:
                    degree = degree + 1
        for iteration in range(number_of_nodes):
            if iteration != node:
                if networkWeight[1][iteration][node]:
                    networkWeight[2][iteration][node] = 1 / degree
    networkFile.close()
    # 设置传给前端的节点数据边数据的json串
    graph_data_json = {}
    nodes_data_json = []
    for node in range(number_of_nodes):
        nodes_data_json.append({})
        nodes_data_json[node]['attributes'] = {}
        nodes_data_json[node]['attributes']['modularity_class'] = 0
        nodes_data_json[node]['id'] = str(node)
        nodes_data_json[node]['category'] = 0
        nodes_data_json[node]['itemStyle'] = ''
        nodes_data_json[node]['label'] = {}
        nodes_data_json[node]['label']['normal'] = {}
        nodes_data_json[node]['label']['normal']['show'] = 'false'
        nodes_data_json[node]['name'] = str(node)
        nodes_data_json[node]['symbolSize'] = 35
        nodes_data_json[node]['value'] = 15
        nodes_data_json[node]['x'] = 0
        nodes_data_json[node]['y'] = 0
    links_data_json = []
    for link in networkTemp:
        links_data_json.append({})
        links_data_json[len(links_data_json) - 1]['id'] = str(len(links_data_json) - 1)
        links_data_json[len(links_data_json) - 1]['lineStyle'] = {}
        links_data_json[len(links_data_json) - 1]['lineStyle']['normal'] = {}
        links_data_json[len(links_data_json) - 1]['name'] = 'null'
        links_data_json[len(links_data_json) - 1]['source'] = str(link[0] - 1)
        links_data_json[len(links_data_json) - 1]['target'] = str(link[1] - 1)
    graph_data_json['nodes'] = nodes_data_json
    graph_data_json['links'] = links_data_json
    graph_data = json.dumps(graph_data_json)

    # 计算一个节点集合在单次模拟下的影响力
    def set_influence(node_set, method):
        active_nodes = []  # 存放所有被激活的节点
        # 把要计算的集合中的节点放入刚才定义的数组中
        for initial_node in node_set:
            active_nodes.append(initial_node)
        last_length = 0  # 这个值用来判断两次迭代之间active_nodes的数量有没有变化，如果没有变化说明没有新的节点被激活就结束迭代
        while len(active_nodes) != last_length:
            if last_length == 0:
                new_active_nodes = []  # 新激活的节点是ic模型中每次迭代中需要去激活别人的节点
                for initial_node in active_nodes:  # 如果是第一次那么输入集合的节点就都是新激活节点
                    new_active_nodes.append(initial_node)
            last_length = len(active_nodes)
            temp_active_nodes = []  # used to temporary storage for each iteration's new active nodes
            for new_active_node in new_active_nodes:
                for node in range(number_of_nodes):
                    if networkWeight[method][new_active_node][node] and node not in active_nodes:
                        # 生产随机数与边权重判断是否激活
                        if random.random() < networkWeight[method][new_active_node][node]:
                            active_nodes.append(node)
                            temp_active_nodes.append(node)
            new_active_nodes = []
            for node in temp_active_nodes:  # 把缓存中的节点放入当前步的新激活节点也就是下一步中要去激活别人的节点
                new_active_nodes.append(node)
        return active_nodes

    # 执行贪心算法找影响力最大的节点
    active_records = []  # 用来存放每个节点的模拟结果也就是最后激活的节点们
    max_node_influence = 0  # 用来存放比较过程中当前最大的影响力
    for node in range(number_of_nodes):
        active_records.append([])
        active_records[node] = set_influence([node], 1)  # 把这个节点的模拟结果存起来
        influence = len(active_records[node])
        if influence > max_node_influence:
            max_node_influence = influence
            max_influence_node = node
    active_records = json.dumps(active_records)
    # 把你需要的数据给对应的页面
    return render_template('basic_ic_1.html', graph_data=graph_data, active_records=active_records, max_node_influence=
    max_node_influence, max_influence_node=max_influence_node)


# 选择单个影响力最大的种子基于ic模型（每个节点模拟十次）
@app.route('/basicIc10')
def basic_ic_10():  # 胡莎莎
    # file = open('static/data/test.txt', 'r')
    # graph_data = file.read()
    # file.close()
    import random
    import json
    # 读取数据
    networkTemp = []
    networkFile = open('static/data/Wiki.txt', 'r')
    # 设置节点数
    number_of_nodes = 105

    for line in networkFile.readlines():
        linePiece = line.split()
        networkTemp.append([int(linePiece[0]), int(linePiece[1])])
    # 初始化权重矩阵
    networkWeight = []
    for i in range(3):
        networkWeight.append([])
        for j in range(number_of_nodes):
            networkWeight[i].append([])
            for k in range(number_of_nodes):
                networkWeight[i][j].append(0)
    # 设置权重
    # 边的权重有三种设置方式，一种是从[0.1, 0.01, 0.001]中随机选一个，一种是都固定0.1，一种是节点的入度分之一
    probability_list = [0.1, 0.01, 0.001]
    for linePiece in networkTemp:
        networkWeight[0][linePiece[0] - 1][linePiece[1] - 1] = random.choice(probability_list)
        networkWeight[1][linePiece[0] - 1][linePiece[1] - 1] = 0.1
    for node in range(number_of_nodes):
        degree = 0
        for iteration in range(number_of_nodes):
            if iteration != node:
                if networkWeight[1][iteration][node]:
                    degree = degree + 1
        for iteration in range(number_of_nodes):
            if iteration != node:
                if networkWeight[1][iteration][node]:
                    networkWeight[2][iteration][node] = 1 / degree
    networkFile.close()
    # 设置传给前端的节点数据边数据的json串
    graph_data_json = {}
    nodes_data_json = []
    for node in range(number_of_nodes):
        nodes_data_json.append({})
        nodes_data_json[node]['attributes'] = {}
        nodes_data_json[node]['attributes']['modularity_class'] = 0
        nodes_data_json[node]['id'] = str(node)
        nodes_data_json[node]['category'] = 0
        nodes_data_json[node]['itemStyle'] = ''
        nodes_data_json[node]['label'] = {}
        nodes_data_json[node]['label']['normal'] = {}
        nodes_data_json[node]['label']['normal']['show'] = 'false'
        nodes_data_json[node]['name'] = str(node)
        nodes_data_json[node]['symbolSize'] = 35
        nodes_data_json[node]['value'] = 15
        nodes_data_json[node]['x'] = 0
        nodes_data_json[node]['y'] = 0
    links_data_json = []
    for link in networkTemp:
        links_data_json.append({})
        links_data_json[len(links_data_json) - 1]['id'] = str(len(links_data_json) - 1)
        links_data_json[len(links_data_json) - 1]['lineStyle'] = {}
        links_data_json[len(links_data_json) - 1]['lineStyle']['normal'] = {}
        links_data_json[len(links_data_json) - 1]['name'] = 'null'
        links_data_json[len(links_data_json) - 1]['source'] = str(link[0] - 1)
        links_data_json[len(links_data_json) - 1]['target'] = str(link[1] - 1)
    graph_data_json['nodes'] = nodes_data_json
    graph_data_json['links'] = links_data_json
    graph_data = json.dumps(graph_data_json)

    # 计算一个节点集合在单次模拟下的影响力
    def set_influence(node_set, method):
        active_nodes = []  # 存放所有被激活的节点
        # 把要计算的集合中的节点放入刚才定义的数组中
        for initial_node in node_set:
            active_nodes.append(initial_node)
        last_length = 0  # 这个值用来判断两次迭代之间active_nodes的数量有没有变化，如果没有变化说明没有新的节点被激活就结束迭代
        while len(active_nodes) != last_length:
            if last_length == 0:
                new_active_nodes = []  # 新激活的节点是ic模型中每次迭代中需要去激活别人的节点
                for initial_node in active_nodes:  # 如果是第一次那么输入集合的节点就都是新激活节点
                    new_active_nodes.append(initial_node)
            last_length = len(active_nodes)
            temp_active_nodes = []  # used to temporary storage for each iteration's new active nodes
            for new_active_node in new_active_nodes:
                for node in range(number_of_nodes):
                    if networkWeight[method][new_active_node][node] and node not in active_nodes:
                        # 生产随机数与边权重判断是否激活
                        if random.random() < networkWeight[method][new_active_node][node]:
                            active_nodes.append(node)
                            temp_active_nodes.append(node)
            new_active_nodes = []
            for node in temp_active_nodes:  # 把缓存中的节点放入当前步的新激活节点也就是下一步中要去激活别人的节点
                new_active_nodes.append(node)
        return active_nodes

    # 执行贪心算法找影响力最大的节点
    active_records = []  # 用来存放每个节点的模拟结果也就是最后激活的节点们
    max_node_influence = 0  # 用来存放比较过程中当前最大的影响力
    for node in range(number_of_nodes):
        active_records.append([])
        influence = 0
        for simulation_count in range(0, 10):  # 模拟10次
            active_records[node].append([])
            active_records[node][simulation_count] = set_influence([node], 1)  # 把这个节点的模拟结果存起来
            influence += len(active_records[node][simulation_count])
        if influence > max_node_influence:
            max_node_influence = influence
            max_influence_node = node
    active_records = json.dumps(active_records)
    # 把你需要的数据给对应的页面
    return render_template('basic_ic_10.html', graph_data=graph_data, active_records=active_records, max_node_influence=
    max_node_influence, max_influence_node=max_influence_node)


# 选择单个影响力最大的种子基于lt模型（每个节点模拟一次）
@app.route('/basicLt1')  # 王钊
def basic_lt_1():
    # file = open('static/data/test.txt', 'r')
    # graph_data = file.read()
    # file.close()
    import random
    import json
    import copy
    # 读取数据
    networkTemp = []
    with open('static/data/Wiki.txt', 'r') as networkFile:
        # 设置节点数
        number_of_nodes = 105

        for line in networkFile.readlines():
            linePiece = line.split()
            networkTemp.append([int(linePiece[0]), int(linePiece[1])])
    # 初始化权重矩阵
    networkWeight = []
    # 设置第一，第二中权值的剩余权值，防止入度的总和超过1
    node_degree_1 = [1 for i in range(number_of_nodes)]
    node_degree_2 = [1 for i in range(number_of_nodes)]
    for i in range(3):
        networkWeight.append([])
        for j in range(number_of_nodes):
            networkWeight[i].append([])
            for k in range(number_of_nodes):
                networkWeight[i][j].append(0)
    # 设置权重
    # 边的权重有三种设置方式，一种是从[0.1, 0.01, 0.001]中随机选一个，一种是都固定0.1，一种是节点的入度分之一
    probability_list = [0.1, 0.01, 0.001]
    degree = [0 for i in range(number_of_nodes)]  # 保存每个节点的度

    for linePiece in networkTemp:
        degree[linePiece[1] - 1] += 1
        if node_degree_1[linePiece[1] - 1] >= 0.1:
            k = random.choice(probability_list)
            networkWeight[0][linePiece[0] - 1][linePiece[1] - 1] = k
            node_degree_1[linePiece[1] - 1] -= k
        elif 0.001 <= node_degree_1[linePiece[1] - 1] < 0.1:  # 去除剩余值过小的情况
            networkWeight[0][linePiece[0] - 1][linePiece[1] - 1] = node_degree_1[linePiece[1] - 1]
        if node_degree_2[linePiece[1] - 1] >= 0.1:
            networkWeight[1][linePiece[0] - 1][linePiece[1] - 1] = 0.1
            node_degree_2[linePiece[1] - 1] -= 0.1
    for node in range(number_of_nodes):
        for iteration in range(number_of_nodes):
            if iteration != node:
                if networkWeight[1][iteration][node]:
                    networkWeight[2][iteration][node] = 1 / degree[node]

    # 设置传给前端的节点数据边数据的json串
    graph_data_json = {}
    nodes_data_json = []
    for node in range(number_of_nodes):
        nodes_data_json.append({})
        nodes_data_json[node]['attributes'] = {}
        nodes_data_json[node]['attributes']['modularity_class'] = 0
        nodes_data_json[node]['id'] = str(node)
        nodes_data_json[node]['category'] = 0
        nodes_data_json[node]['itemStyle'] = ''
        nodes_data_json[node]['label'] = {}
        nodes_data_json[node]['label']['normal'] = {}
        nodes_data_json[node]['label']['normal']['show'] = 'false'
        nodes_data_json[node]['name'] = str(node)
        nodes_data_json[node]['symbolSize'] = 35
        nodes_data_json[node]['value'] = 15
        nodes_data_json[node]['x'] = 0
        nodes_data_json[node]['y'] = 0
    links_data_json = []
    for link in networkTemp:
        links_data_json.append({})
        links_data_json[len(links_data_json) - 1]['id'] = str(len(links_data_json) - 1)
        links_data_json[len(links_data_json) - 1]['lineStyle'] = {}
        links_data_json[len(links_data_json) - 1]['lineStyle']['normal'] = {}
        links_data_json[len(links_data_json) - 1]['name'] = 'null'
        links_data_json[len(links_data_json) - 1]['source'] = str(link[0] - 1)
        links_data_json[len(links_data_json) - 1]['target'] = str(link[1] - 1)
    graph_data_json['nodes'] = nodes_data_json
    graph_data_json['links'] = links_data_json
    graph_data = json.dumps(graph_data_json)

    def set_influence_LT(node_set, m):
        """
        基于LT模型计算node_set集合的影响力
        :param node_set: 节点集合
        :param m: 设置使用的权重矩阵
        :return: 返回被激活的节点集合
        """
        active_nodes = node_set  # 存放被激活的节点，初始为node_set
        last_influence = 0  # 存放最新的节点影响力，如果值为0就说明影响力传播结束
        start = last_influence
        while last_influence != len(active_nodes):
            last_influence = len(active_nodes)  # 更新影响力
            for new_active_node in active_nodes[start:last_influence]:
                for nei_node in range(number_of_nodes):
                    if networkWeight[method][new_active_node][nei_node] != 0 and nei_node not in active_nodes:
                        # 将邻居节点的阈值减去边上的权重，如果阈值小于0，那么节点被激活
                        theta[nei_node] -= networkWeight[method][new_active_node][nei_node]
                        if theta[nei_node] <= 0:
                            active_nodes.append(nei_node)
            start = last_influence
        return active_nodes

    theta = []  # 保存每个节点的阈值
    for iteration in range(number_of_nodes):  # 为每个节点随机设置阈值
        theta.append(random.random())
    # 基于LT模型，执行贪心算法找影响力最大的节点
    active_records = []  # 用来存放每个节点的模拟结果也就是最后激活的节点们
    max_node_influence = 0  # 用来存放比较过程中当前最大的影响力
    method = 1  # 选择使用哪种权重进行
    save_theta = copy.deepcopy(theta)  # 保存theta
    for node in range(number_of_nodes):  # 遍历所有的节点，判断影响力
        active_records.append([])
        active_records[node] = set_influence_LT([node], method)  # 执行贪心算法，保存被激活的节点，第一个参数为列表
        influence = len(active_records[node])
        if influence > max_node_influence:
            max_node_influence = influence
            max_influence_node = node
        theta = copy.deepcopy(save_theta)  # 恢复theta
    active_records = json.dumps(active_records)
    # 把你需要的数据给对应的页面
    return render_template('basic_ic_1.html', graph_data=graph_data, active_records=active_records, max_node_influence=
    max_node_influence, max_influence_node=max_influence_node)


# 选择单个影响力最大的种子基于page rank
@app.route('/pageRank')
def page_rank():
    return ''


# 选择单个影响力最大的种子基于节点的度
@app.route('/degree')  # 刘艳霞
def degree():
    import numpy as np
    import json
    # 读取数据
    networkTemp = []
    networkFile = open('static/data/Wiki.txt', 'r')
    # 设置节点数
    number_of_nodes = 105

    for line in networkFile.readlines():
        linePiece = line.split()
        networkTemp.append([int(linePiece[0]), int(linePiece[1])])

    # 设置传给前端的节点数据边数据的json串
    graph_data_json = {}
    nodes_data_json = []
    for node in range(number_of_nodes):
        nodes_data_json.append({})
        nodes_data_json[node]['attributes'] = {}
        nodes_data_json[node]['attributes']['modularity_class'] = 0
        nodes_data_json[node]['id'] = str(node)
        nodes_data_json[node]['category'] = 0
        nodes_data_json[node]['itemStyle'] = ''
        nodes_data_json[node]['label'] = {}
        nodes_data_json[node]['label']['normal'] = {}
        nodes_data_json[node]['label']['normal']['show'] = 'false'
        nodes_data_json[node]['name'] = str(node)
        nodes_data_json[node]['symbolSize'] = 35
        nodes_data_json[node]['value'] = 15
        nodes_data_json[node]['x'] = 0
        nodes_data_json[node]['y'] = 0
    links_data_json = []
    for link in networkTemp:
        links_data_json.append({})
        links_data_json[len(links_data_json) - 1]['id'] = str(len(links_data_json) - 1)
        links_data_json[len(links_data_json) - 1]['lineStyle'] = {}
        links_data_json[len(links_data_json) - 1]['lineStyle']['normal'] = {}
        links_data_json[len(links_data_json) - 1]['name'] = 'null'
        links_data_json[len(links_data_json) - 1]['source'] = str(link[0] - 1)
        links_data_json[len(links_data_json) - 1]['target'] = str(link[1] - 1)
    graph_data_json['nodes'] = nodes_data_json
    graph_data_json['links'] = links_data_json
    graph_data = json.dumps(graph_data_json)

    # 网络的邻接矩阵
    adjacencyMatrix = np.zeros([number_of_nodes, number_of_nodes], dtype=int)
    for i in range(len(networkTemp)):
        adjacencyMatrix[int(networkTemp[i][0] - 1)][int(networkTemp[i][1] - 1)] = 1
        adjacencyMatrix[int(networkTemp[i][1] - 1)][int(networkTemp[i][0] - 1)] = 1
    active_records = []  # 用来存放每个节点的模拟结果
    for i in range(number_of_nodes):
        active_records.append([])
        for j in range(number_of_nodes):
            if (adjacencyMatrix[i][j] == 1):
                active_records[i].append(j)
    active_records = json.dumps(active_records)

    # 存放各个节点的度
    nodeDegree = []
    for i in range(len(adjacencyMatrix)):
        nodeDegree.append(sum(adjacencyMatrix[i]))
    # 最大影响力节点
    max_influence_node = nodeDegree.index(max(nodeDegree)) + 1
    # 最大影响力节点的度
    max_node_influence = max(nodeDegree)
    return render_template('degree.html', graph_data=graph_data, active_records=active_records,
                           max_node_influence=max_node_influence, max_influence_node=max_influence_node)


if __name__ == '__main__':
    app.run()
