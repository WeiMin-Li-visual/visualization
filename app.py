# -*- coding:utf-8 -*-
from flask import Flask, render_template, request, flash
import json

app = Flask(__name__)
app.secret_key = 'lisenzzz'


@app.route('/')
def hello_world():
    return render_template('index.html')


# 计算局部区域内两两节点间的斥力所产生的单位位移
def updateReplusion(node_coordinate):
    import math
    number_of_nodes = 105
    ejectFactor = 6  # 斥力系数
    k = math.sqrt(1024 * 768 / number_of_nodes)
    for i in range(number_of_nodes):
        node_coordinate[i].append(0)
        node_coordinate[i].append(0)
        for j in range(number_of_nodes):
            if (i != j):
                dx = node_coordinate[i][0] - node_coordinate[j][0]  # 两个节点x坐标位置的距离
                dy = node_coordinate[i][1] - node_coordinate[j][1]  # 两个节点y坐标位置的距离
                dist = math.sqrt(dx * dx + dy * dy)  # 两个节点之间的距离
                if (dist < 30):
                    ejectFactor = 5
                if (dist > 0 and dist < 250):
                    node_coordinate[i][2] += dx / dist * k / dist * ejectFactor
                    node_coordinate[i][3] += dy / dist * k / dist * ejectFactor
    return node_coordinate


# 计算每条边的引力对两端节点所产生的单位位移
def updateSpring(node_coordinate, networkTemp):
    import math
    number_of_nodes = 105
    k = math.sqrt(1024 * 768 / number_of_nodes)
    condenseFactor = 5  # 引力系数
    for i in range(len(networkTemp)):
        start = networkTemp[i][0] - 1
        end = networkTemp[i][1] - 1
        dx = node_coordinate[start][0] - node_coordinate[end][0]
        dy = node_coordinate[start][1] - node_coordinate[end][1]
        dist = math.sqrt(dx * dx + dy * dy)
        node_coordinate[start][2] -= dx * dist / k * condenseFactor
        node_coordinate[start][3] -= dy * dist / k * condenseFactor
        node_coordinate[end][2] += dx * dist / k * condenseFactor
        node_coordinate[end][3] += dy * dist / k * condenseFactor
    return node_coordinate


# 更新坐标位置
def update(node_coordinate):
    import math
    number_of_nodes = 105
    maxtx = 4
    maxty = 3
    for i in range(number_of_nodes):
        dx = math.floor(node_coordinate[i][2])
        dy = math.floor(node_coordinate[i][3])
        if dx < -maxtx:
            dx = -maxtx
        if dx > maxtx:
            dx = maxtx
        if dy < -maxty:
            dy = -maxty
        if dy > maxty:
            dy = maxty
        if node_coordinate[i][0] + dx >= 1024 or node_coordinate[i][0] + dx <= 0:
            node_coordinate[i][0] -= dx
        else:
            node_coordinate[i][0] += dx
        if node_coordinate[i][1] + dy >= 768 or node_coordinate[i][1] + dy <= 0:
            node_coordinate[i][1] -= dy
        else:
            node_coordinate[i][1] += dy
        # node_coordinate[i][0]+=dx
        # node_coordinate[i][1]+=dy
    return node_coordinate


def forceDirect(networkTemp):
    import random
    number_of_nodes = 105
    node_coordinate = []  # 存放节点的坐标及节点之间的斥力，[x坐标，y坐标,force_x.force_y]
    for node in range(number_of_nodes):
        node_coordinate.append([])
        node_coordinate[node].append(500 + 40 * (random.random() - 0.5))
        node_coordinate[node].append(500 + 40 * (random.random() - 0.5))
        # node_coordinate[node].append(0)
        # node_coordinate[node].append(0)
    for i in range(500):
        node_coordinate = updateReplusion(node_coordinate)
        node_coordinate = updateSpring(node_coordinate, networkTemp)
        node_coordinate = update(node_coordinate)
    return node_coordinate


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
    node_coordinate = forceDirect(networkTemp)  # 设置节点坐标
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
        nodes_data_json[node]['x'] = node_coordinate[node][0]
        nodes_data_json[node]['y'] = node_coordinate[node][1]
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
    node_coordinate = forceDirect(networkTemp)  # 设置节点坐标
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
        nodes_data_json[node]['x'] = node_coordinate[node][0]
        nodes_data_json[node]['y'] = node_coordinate[node][1]
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


# 选择单个影响力最大的种子基于lt模型（每个节点模拟十次）
@app.route('/basicLt10')  # 王钊
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
    node_coordinate = forceDirect(networkTemp)  # 设置节点坐标

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
        nodes_data_json[node]['x'] = node_coordinate[node][0]
        nodes_data_json[node]['y'] = node_coordinate[node][1]
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

    def set_influence_LT(node_set):
        """
        基于LT模型计算node_set集合的影响力
        :param node_set: 节点集合
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

    # 基于LT模型，找影响力最大的节点
    active_records = []  # 用来存放每个节点的模拟结果也就是最后激活的节点们
    max_node_influence = 0  # 用来存放比较过程中当前最大的影响力
    method = 1  # 选择使用哪种权重进行
    for node in range(number_of_nodes):  # 遍历所有的节点，判断影响力
        active_records.append([])
        stimulate_round = 10  # 激活轮数
        count_influence = 0  # 记录当前节点10次模拟的影响力总和
        for round in range(stimulate_round):  # 重新设置每个节点的阈值
            theta = []  # 保存每个节点的阈值
            for iteration in range(number_of_nodes):  # 为每个节点随机设置阈值
                theta.append(random.random())
            l = set_influence_LT([node])
            active_records[node].append(l)  # 保存被激活的节点，第一个参数为列表
            count_influence += len(l)
        influence = count_influence / stimulate_round
        if influence > max_node_influence:
            max_node_influence = influence
            max_influence_node = node

    active_records = json.dumps(active_records)
    # 把你需要的数据给对应的页面
    return render_template('basic_lt_10.html', graph_data=graph_data, active_records=active_records, max_node_influence=
    max_node_influence, max_influence_node=max_influence_node)


# 选择单个影响力最大的种子基于page rank
# author: 张财
# date: 2019年11月25日22:19:26
@app.route('/pageRank')
def page_rank():
    import math
    import copy

    network_file = open('static/data/Wiki.txt')
    network = []
    for each_line in network_file:
        line_split = each_line.split()
        network.append([int(line_split[0]), int(line_split[1])])
    network_file.close()
    # 计算节点个数
    nodes = set({})
    for each_link in network:
        nodes.add(each_link[0])
        nodes.add(each_link[1])
    node_num = len(nodes)

    increment = 1  # 每次迭代后节点影响力的增量,迭代终止条件
    iteration = 1  # 当前迭代次数
    inf_old = [1] * node_num  # 上一次迭代后节点的影响力,初始化为1
    inf_new = [0] * node_num  # 这次迭代后影响力的更新，初始化为0
    c = 0.5
    outdegree = []  # 每个节点的出度
    node_neighbors = []  # 指向该节点的邻居节点集合
    iter_influences = [copy.deepcopy(inf_old)]  # 每次迭代后个节点的影响力

    # 求出度和邻居节点
    for node in range(node_num):
        cur_node_neighbors = []  # 节点node的邻居节点
        node_outdegree = 0
        for each_link in network:
            if node + 1 == each_link[1]:
                cur_node_neighbors.append(each_link[0] - 1)
            if node + 1 == each_link[0]:
                node_outdegree += 1
        outdegree.append(node_outdegree)
        node_neighbors.append(cur_node_neighbors)

    # 开始迭代求节点影响力
    while increment > 1 / node_num:
        increment = 0
        for node in range(node_num):
            # 求节点node的影响力
            for neighbor in node_neighbors[node]:
                inf_new[node] += c * (inf_old[neighbor] / outdegree[neighbor])
            inf_new[node] += (1 - c)
            node_increment = math.fabs(inf_new[node] - inf_old[node])  # 节点node的影响力改变值
            increment += node_increment
        # 更新inf_old
        for i in range(node_num):
            inf_old[i] = inf_new[i]
            inf_new[i] = 0
        iter_influences.append(copy.deepcopy(inf_old))
        iteration += 1
    max_influence = max(inf_old)  # 最大的影响力
    max_inf_node = inf_old.index(max_influence)  # 最大影响力的节点
    node_coordinate = forceDirect(network)  # 节点的坐标

    # 可视化部分
    # 设置传给前端的节点数据边数据的json串
    graph_data_json = {}
    nodes_data_json = []
    for node in range(node_num):
        nodes_data_json.append({
            'attributes': {'modularity_class': 0},
            'id': str(node),
            'category': 0,
            'itemStyle': '',
            'label': {'normal': {'show': 'false'}},
            'name': str(node),
            'symbolSize': 35,
            'value': 15,
            'x': node_coordinate[node][0],
            'y': node_coordinate[node][1]
        })
    links_data_json = []
    for link in network:
        link_id = len(links_data_json)
        links_data_json.append({
            'id': str(link_id),
            'lineStyle': {'normal': {}},
            'name': 'null',
            'source': str(link[0] - 1),
            'target': str(link[1] - 1)
        })
    graph_data_json['nodes'] = nodes_data_json
    graph_data_json['links'] = links_data_json
    graph_data = json.dumps(graph_data_json)
    return render_template('page_rank.html', graph_data=graph_data, influences=iter_influences,
                           max_influence=max_influence, max_inf_node=max_inf_node)


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
    node_coordinate = forceDirect(networkTemp)  # 设置节点的坐标
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
        nodes_data_json[node]['x'] = node_coordinate[node][0]
        nodes_data_json[node]['y'] = node_coordinate[node][1]
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


@app.route('/input', methods=["GET", "POST"])
def test():
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

    node_coordinate = forceDirect(networkTemp)  # 设置节点的坐标

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
        nodes_data_json[node]['x'] = node_coordinate[node][0]
        nodes_data_json[node]['y'] = node_coordinate[node][1]
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

    def influence_spread(seed, m):
        """
        使用IC模型传播影响力
        :param seed: 初始种子集合
        :param m: 使用的概率设置方法
        :return: 被激活的所有节点
        """
        active = seed
        start = 0
        end = len(seed)
        while start != end:
            for node in active[start:end]:
                for i in range(105):
                    if networkWeight[m][node][i] != 0:
                        if i not in active and random.random() < networkWeight[m][node][i]:
                            active.append(i)
            start = end
            end = len(active)
        return active

    if request.method == "GET":
        err = "true"
        active_records = json.dumps([])
        return render_template('input.html', graph_data=graph_data, active_records=active_records, err=err)
    else:
        err = "false"  # 返回错误信息
        temp = request.form.get("value")  # 获取前端传回的数据
        method = request.form.get("method")
        if method != "":
            method = int(method)
            if method > 3 or method < 1:
                err = "method error"
        else:
            err = "请输入方法"
        if len(temp) == 0:
            err = "请输入节点信息"
        if err != "false":
            active_records = json.dumps([])
            return render_template('input.html', graph_data=graph_data, active_records=active_records, err=err)
        data = []
        method -= 1
        for c in temp:
            if c.isdigit() and c != ',':
                c = int(c)
                if 0 <= c <= 104 and type(c) == int:
                    data.append(c)
                else:
                    err = "节点序号应为0-104的整数"
            elif c != ',':
                err = "节点序号应为0-104的整数"
                active_records = json.dumps([])
                return render_template('input.html', graph_data=graph_data, active_records=active_records, err=err)

        active_node = influence_spread(data, method)  # 保存激活的节点
        active_records = json.dumps(active_node)
        return render_template('input.html', graph_data=graph_data, active_records=active_records, err=err)


if __name__ == '__main__':
    app.run()
