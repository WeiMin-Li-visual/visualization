# -*- coding:utf-8 -*-
from flask import Flask, render_template, request, flash
import json
import random


# 初始化网络
# input: data_path
# output: network, node_num, graph_data
def init_network(data_path):
    # 初始化节点坐标
    def init_cordinate(network, number_of_nodes):
        # 计算局部区域内两两节点间的斥力所产生的单位位移
        def updateReplusion(node_coordinate, number_of_nodes):
            import math
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
        def updateSpring(node_coordinate, network):
            import math
            number_of_nodes = 105
            k = math.sqrt(1024 * 768 / number_of_nodes)
            condenseFactor = 5  # 引力系数
            for i in range(len(network)):
                start = network[i][0] - 1
                end = network[i][1] - 1
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

        # 力导向算法
        def forceDirect(network, number_of_nodes):
            random.seed(750)
            node_coordinate = []  # 存放节点的坐标及节点之间的斥力，[x坐标，y坐标,force_x.force_y]
            for node in range(number_of_nodes):
                node_coordinate.append([])
                node_coordinate[node].append(500 + 40 * (random.random() - 0.5))
                node_coordinate[node].append(500 + 40 * (random.random() - 0.5))
                # node_coordinate[node].append(0)
                # node_coordinate[node].append(0)
            for i in range(500):
                node_coordinate = updateReplusion(node_coordinate, number_of_nodes)
                node_coordinate = updateSpring(node_coordinate, network)
                node_coordinate = update(node_coordinate)
            return node_coordinate

        return forceDirect(network, number_of_nodes)

    # 读入数据
    network_file = open(data_path)
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

    node_coordinate = init_cordinate(network, node_num)  # 节点的坐标
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
            'value': 111,
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
    return network, node_num, graph_data


# 计算网络的权重矩阵
def init_networkWeight(networkTemp, number_of_nodes):
    networkWeight = []
    edgeNum = []  # 存储每条边在networkTemp中的序号
    for i in range(3):
        networkWeight.append([])
        for j in range(number_of_nodes):
            networkWeight[i].append([])
            for k in range(number_of_nodes):
                networkWeight[i][j].append(0)
    # 设置权重
    # 边的权重有三种设置方式，一种是从[0.1, 0.01, 0.001]中随机选一个，一种是都固定0.1，一种是节点的入度分之一
    for i in range(number_of_nodes):
        edgeNum.append([])
        for j in range(number_of_nodes):
            edgeNum[i].append(0)
    probability_list = [0.1, 0.01, 0.001]
    count = 0
    for linePiece in networkTemp:
        networkWeight[0][linePiece[0] - 1][linePiece[1] - 1] = random.choice(probability_list)
        networkWeight[1][linePiece[0] - 1][linePiece[1] - 1] = 0.1
        edgeNum[linePiece[0] - 1][linePiece[1] - 1] = count
        count += 1
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
    return networkWeight, edgeNum


def set_influence(seed, m, networkWeight, number_of_nodes, edgeNum):
    """
    基于IC模型计算一个节点集合在单次模拟下的影响力
    :param seed: 种子节点集合
    :param m: 选择哪种节点影响力进行传播
    :param networkWeight: 影响力权重矩阵
    :param number_of_nodes: 节点数
    :return: 被激活的节点集合
    """
    active = seed
    start = 0
    end = len(seed)
    while start != end:
        index = start
        while index < end:
            for i in range(number_of_nodes):
                if networkWeight[m][active[index]][i] != 0:
                    if i not in active and random.random() < networkWeight[m][active[index]][i]:
                        active.append(i)
                        active.append(edgeNum[active[index]][i])  # 存储边对应的序号
            index += 2
        start = end
        end = len(active)
    # print('active_num',active_num)
    # print(active)
    return active


# 基于IC模型计算一个节点集合在10次模拟下激活的节点
def set_influence_IC_10(seed, m, networkWeight, edgeNum):
    """
    基于IC模型计算一个节点集合在10次模拟下激活的节点
    :param seed: 种子节点集合
    :param m: 选择哪种节点影响力进行传播
    :param networkWeight: 影响力权重矩阵
    :return active_records: 被激活的节点集合
    :return: active_num  保存active中的节点在下一次传播中激活节点的个数，用于判断IC模型下active中各个节点之间的激活关系
    """
    active = seed
    start = 0
    end = len(seed)
    active_records = []  # 用来存放每个节点的模拟结果也就是最后激活的节点们

    for simulation_count in range(0, 10):  # 模拟10次
        active_records.append([])

        for node in active[start:end]:
            active_records[simulation_count].append([])
            active_nodes = set_influence([node], 1, networkWeight, number_of_nodes, edgeNum)  # 把这个节点的模拟结果存起来
            active_records[simulation_count].append(active_nodes)

    return active_records


# 基于LT模型计算一个节点集合在10次模拟下激活的节点
def set_influence_LT_10(seed, method):  # 胡莎莎
    '''
    基于LT模型计算一个节点集合在10次模拟下激活的节点
    :param seed:
    :param method:
    :return:
    '''
    active = seed
    start = 0
    end = len(seed)
    # 初始化权重矩阵
    networkWeight = init_networkWeight(networkTemp, number_of_nodes)
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

    active_records = []  # 用来存放每个节点的模拟结果也就是最后激活的节点们
    method = 1  # 选择使用哪种权重进行
    node_count = 0  # 记录当前在计算的节点个数，当列表下标用
    stimulate_round = 10  # 激活轮数
    count_influence = 0  # 记录当前节点10次模拟的影响力总和
    for round in range(stimulate_round):  # 重新设置每个节点的阈值
        active_records.append([])
        for node in active[start:end]:  # 遍历输入节点集合中的所有节点，判断影响力

            theta = []  # 保存每个节点的阈值
            for iteration in range(number_of_nodes):  # 为每个节点随机设置阈值
                theta.append(random.random())
            l = set_influence_LT([node])
            active_records[round].append(l)  # 保存被激活的节点，第一个参数为列表
            count_influence += len(l)
        node_count += 1

    # 这里返回一个三维的数组  这是节点集合为1时返回的数据
    # active_node: [[[1], [1, 3], [1], [1], [1], [1], [1], [1, 3, 11, 12, 22, 27, 17, 23, 24, 41,54], [1], [1]]]
    return active_records


# 基于pageRank算法计算一个节点集合的影响力
def set_influence_pageRank(seed, m):  # 胡莎莎
    '''
    基于pageRank算法计算一个节点集合的影响力
    :param seed:
    :param m:
    :return:
    '''

    import math
    import copy

    increment = 1  # 每次迭代后节点影响力的增量,迭代终止条件
    iteration = 1  # 当前迭代次数
    inf_old = [1] * number_of_nodes  # 上一次迭代后节点的影响力,初始化为1
    inf_new = [0] * number_of_nodes  # 这次迭代后影响力的更新，初始化为0
    c = 0.5
    outdegree = []  # 每个节点的出度
    node_neighbors = []  # 指向该节点的邻居节点集合
    iter_influences = [copy.deepcopy(inf_old)]  # 每次迭代后个节点的影响力

    # 求出度和邻居节点
    for node in range(number_of_nodes):
        cur_node_neighbors = []  # 节点node的邻居节点
        node_outdegree = 0
        for each_link in networkTemp:
            if node + 1 == each_link[1]:
                cur_node_neighbors.append(each_link[0] - 1)
            if node + 1 == each_link[0]:
                node_outdegree += 1
        outdegree.append(node_outdegree)
        node_neighbors.append(cur_node_neighbors)

    # 开始迭代求节点影响力
    while increment > 1 / number_of_nodes:
        increment = 0
        for node in range(number_of_nodes):
            # 求节点node的影响力
            for neighbor in node_neighbors[node]:
                inf_new[node] += c * (inf_old[neighbor] / outdegree[neighbor])
            inf_new[node] += (1 - c)
            node_increment = math.fabs(inf_new[node] - inf_old[node])  # 节点node的影响力改变值
            increment += node_increment
        # 更新inf_old
        for i in range(number_of_nodes):
            inf_old[i] = inf_new[i]
            inf_new[i] = 0
        iter_influences.append(copy.deepcopy(inf_old))
        iteration += 1
    active_records = seed  # 用来存放每个节点的模拟结果
    start = 0
    end = len(seed)
    influence = 0
    for node in seed[start:end]:
        if node == "":
            break
        influence += inf_old[node - 1]
    return influence


# 基于最大度算法计算一个节点集合激活的节点
def set_influence_degree(seed, m):  # 胡莎莎
    """
    基于最大度算法计算一个节点集合激活的节点
    :param seed:种子节点集合
    :param m:选择哪种节点影响力进行传播
    :return active_records:被激活的节点集合
    """
    import numpy as np

    # 网络的邻接矩阵
    adjacencyMatrix = np.zeros([number_of_nodes, number_of_nodes], dtype=int)
    for i in range(len(networkTemp)):
        adjacencyMatrix[int(networkTemp[i][0] - 1)][int(networkTemp[i][1] - 1)] = 1
        # adjacencyMatrix[int(networkTemp[i][1] - 1)][int(networkTemp[i][0] - 1)] = 1
    active_records = seed  # 用来存放每个节点的模拟结果
    start = 0
    end = len(seed)
    for node in seed[start:end]:
        if node == "":
            break
        for j in range(number_of_nodes):
            if adjacencyMatrix[node - 1][j] == 1 and j + 1 not in active_records:
                active_records.append(j + 1)
    return active_records


app = Flask(__name__)
app.secret_key = 'lisenzzz'
networkTemp, number_of_nodes, graph_data = init_network('static/data/Wiki.txt')


@app.route('/')
def hello_world():
    return render_template('index.html')


# 选择单个影响力最大的种子基于ic模型（每个节点模拟一次）
@app.route('/basicIc1')
def basic_ic_1():
    # 初始化权重矩阵
    networkWeight, edgeNum = init_networkWeight(networkTemp, number_of_nodes)
    # 执行贪心算法找影响力最大的节点
    active_records = []  # 用来存放每个节点的模拟结果也就是最后激活的节点们
    max_node_influence = 0  # 用来存放比较过程中当前最大的影响力
    active_nums = []
    # graph_data1 = json.loads(graph_data)
    for node in range(number_of_nodes):
        active_records.append([])
        active_nums.append([])
        active_records[node] = set_influence([node, -1], 1, networkWeight, number_of_nodes, edgeNum)  # 把这个节点的模拟结果存起来
        influence = len(active_records[node]) / 2
        # graph_data1['nodes'][node]['value']=influence
        if influence > max_node_influence:
            max_node_influence = influence
            max_influence_node = node
    active_records = json.dumps(active_records)
    # graph_data1=json.dumps(graph_data1)
    # 把你需要的数据给对应的页面

    return render_template('common_template.html', graph_data=graph_data, active_records=active_records,
                           max_node_influence=
                           max_node_influence, max_influence_node=max_influence_node, method_type=1)


# 选择单个影响力最大的种子基于ic模型（每个节点模拟十次）
@app.route('/basicIc10')
def basic_ic_10():  # 胡莎莎
    # 初始化权重矩阵
    networkWeight, edgeNum = init_networkWeight(networkTemp, number_of_nodes)

    # 执行贪心算法找影响力最大的节点
    active_records = []  # 用来存放每个节点的模拟结果也就是最后激活的节点们
    max_node_influence = 0  # 用来存放比较过程中当前最大的影响力
    active_nums = []  # 每个节点每次模拟激活的节点数
    for node in range(number_of_nodes):
        active_records.append([])
        active_nums.append([])
        influence = 0
        for simulation_count in range(0, 10):  # 模拟10次
            active_records[node].append([])
            active_nums[node].append([])
            active_records[node][simulation_count] = set_influence([node, -1], 1, networkWeight, number_of_nodes,
                                                                   edgeNum)  # 把这个节点的模拟结果存起来
            influence += len(active_records[node][simulation_count])
        if influence > max_node_influence:
            max_node_influence = influence
            max_influence_node = node
    active_records = json.dumps(active_records)
    # 把你需要的数据给对应的页面
    max_node_influence /= 20  # 求平均值
    return render_template('common_template.html', graph_data=graph_data, active_records=active_records,
                           max_node_influence=
                           max_node_influence, max_influence_node=max_influence_node, method_type=2)


# 选择单个影响力最大的种子基于lt模型（每个节点模拟十次）
@app.route('/basicLt10')  # 王钊
def basic_lt_1():
    # 初始化权重矩阵
    networkWeight, edgeNum = init_networkWeight(networkTemp, number_of_nodes)
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

    def set_influence_LT(node_set):
        """
        基于LT模型计算node_set集合的影响力
        :param node_set: 节点集合
        :return: 返回被激活的节点集合
        """
        active_nodes = node_set  # 存放被激活的节点，初始为node_set
        start = 0
        end = len(active_nodes)
        while start != end:
            index = start
            while index < end:
                for nei_node in range(number_of_nodes):
                    if networkWeight[method][active_nodes[index]][nei_node] != 0 and nei_node not in active_nodes:
                        # 将邻居节点的阈值减去边上的权重，如果阈值小于0，那么节点被激活
                        theta[nei_node] -= networkWeight[method][active_nodes[index]][nei_node]
                        if theta[nei_node] <= 0:
                            active_nodes.append(nei_node)
                            active_nodes.append(edgeNum[active_nodes[index]][nei_node])
                index += 2
            start = end
            end = len(active_nodes)
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
            l = set_influence_LT([node, -1])
            active_records[node].append(l)  # 保存被激活的节点，第一个参数为列表
            count_influence += len(l)
        influence = count_influence / (stimulate_round * 2)
        if influence > max_node_influence:
            max_node_influence = influence
            max_influence_node = node

    active_records = json.dumps(active_records)
    # 把你需要的数据给对应的页面
    return render_template('common_template.html', graph_data=graph_data, active_records=active_records,
                           max_node_influence=
                           max_node_influence, max_influence_node=max_influence_node, method_type=2)


# 选择单个影响力最大的种子基于page rank
# author: 张财
# date: 2019年11月25日22:19:26
@app.route('/pageRank')
def page_rank():
    import math
    import copy

    increment = 1  # 每次迭代后节点影响力的增量,迭代终止条件
    iteration = 1  # 当前迭代次数
    inf_old = [1] * number_of_nodes  # 上一次迭代后节点的影响力,初始化为1
    inf_new = [0] * number_of_nodes  # 这次迭代后影响力的更新，初始化为0
    c = 0.5
    outdegree = []  # 每个节点的出度
    node_neighbors = []  # 指向该节点的邻居节点集合
    iter_influences = [copy.deepcopy(inf_old)]  # 每次迭代后个节点的影响力

    # 求出度和邻居节点
    for node in range(number_of_nodes):
        cur_node_neighbors = []  # 节点node的邻居节点
        node_outdegree = 0
        for each_link in networkTemp:
            if node + 1 == each_link[1]:
                cur_node_neighbors.append(each_link[0] - 1)
            if node + 1 == each_link[0]:
                node_outdegree += 1
        outdegree.append(node_outdegree)
        node_neighbors.append(cur_node_neighbors)

    # 开始迭代求节点影响力
    while increment > 1 / number_of_nodes:
        increment = 0
        for node in range(number_of_nodes):
            # 求节点node的影响力
            for neighbor in node_neighbors[node]:
                inf_new[node] += c * (inf_old[neighbor] / outdegree[neighbor])
            inf_new[node] += (1 - c)
            node_increment = math.fabs(inf_new[node] - inf_old[node])  # 节点node的影响力改变值
            increment += node_increment
        # 更新inf_old
        for i in range(number_of_nodes):
            inf_old[i] = inf_new[i]
            inf_new[i] = 0
        iter_influences.append(copy.deepcopy(inf_old))
        iteration += 1
    max_influence = max(inf_old)  # 最大的影响力
    max_inf_node = inf_old.index(max_influence)  # 最大影响力的节点

    return render_template('page_rank.html', graph_data=graph_data, influences=iter_influences,
                           max_influence=max_influence, max_inf_node=max_inf_node)


# 选择单个影响力最大的种子基于节点的度
@app.route('/degree')  # 刘艳霞
def degree():
    import numpy as np
    networkWeight, edgeNum = init_networkWeight(networkTemp, number_of_nodes)
    # 网络的邻接矩阵
    adjacencyMatrix = np.zeros([number_of_nodes, number_of_nodes], dtype=int)
    for i in range(len(networkTemp)):
        adjacencyMatrix[int(networkTemp[i][0] - 1)][int(networkTemp[i][1] - 1)] = 1
        ##adjacencyMatrix[int(networkTemp[i][1] - 1)][int(networkTemp[i][0] - 1)] = 1
    active_records = []  # 用来存放每个节点的模拟结果
    for i in range(number_of_nodes):
        active_records.append([])
        active_records[i].append(i)  # 将当前节点放入激活列表中
        active_records[i].append(-1)
        for j in range(number_of_nodes):
            if (adjacencyMatrix[i][j] == 1):
                active_records[i].append(j)
                active_records[i].append(edgeNum[i][j])
    active_records = json.dumps(active_records)

    graph_data1 = json.loads(graph_data)

    # 存放各个节点的度
    nodeDegree = []
    influence = 0
    for i in range(len(adjacencyMatrix)):
        influence = sum(adjacencyMatrix[i])
        nodeDegree.append(influence)
        graph_data1['nodes'][i]['value'] = str(influence)

    # 最大影响力节点
    max_influence_node = nodeDegree.index(max(nodeDegree)) + 1
    # 最大影响力节点的度
    max_node_influence = max(nodeDegree)
    graph_data1 = json.dumps(graph_data1)
    return render_template('common_template.html', graph_data=graph_data1, active_records=active_records,
                           max_node_influence=max_node_influence, max_influence_node=max_influence_node, method_type=1)


@app.route('/input', methods=["GET", "POST"])
def input():
    global networkWeight
    global edgeNum
    if request.method == "GET":
        # 初始化权重矩阵
        networkWeight, edgeNum = init_networkWeight(networkTemp, number_of_nodes)

        err = "true"
        active_records = json.dumps([])
        return render_template('input.html', graph_data=graph_data, active_records=active_records, err=err)
    else:
        err = "false"  # 返回错误信息
        temp = request.form.get('value')
        method = request.form.get('method')
        if method != "":
            if len(method) != 1:
                err = "请输入一个大小为1-3的整数值"
            elif method < '0' or method > '9' or int(method) > 3 or int(method) < 1:
                err = "请输入一个大小为1-3的整数值"
        else:
            err = "请输入方法"
        if temp == "":
            err = "请输入节点信息"
        elif err == "false":
            data = []
            method = int(method)
            method -= 1
            s = ""
            for c in temp:
                if c == "，":
                    s += ","
                else:
                    s += c
            s = set(s.split(","))
            for c in s:
                if c.isdigit() and c != ',' and c != '，':
                    c = int(c)
                    if 0 <= c <= 104 and type(c) == int:
                        data.append(c)
                    else:
                        err = "节点序号应为0-104的整数"
                        break
                else:
                    err = "节点序号应为0-104的整数"

        if err != "false":
            active_records = json.dumps([])
            return render_template('input.html', graph_data=graph_data, active_records=active_records, err=err)
        influence = len(data)
        data.append(-1)
        active_node = set_influence(data, 1, networkWeight, number_of_nodes, edgeNum)  # 保存激活的节点
        influence += (len(active_node) - influence - 1) / 2
        active_records = json.dumps(active_node)
        return render_template('input.html', graph_data=graph_data, active_records=active_records, influence=influence,
                               err=err)


# 计算在选定算法下单个集合的影响力
def calculateSingleSet(method, temp):
    """
    计算在选定算法下单个集合的影响力
    :param method:选择的算法
    :param temp:输入的种子集合
    :return active_records:激活节点
    :return err:错误信息
    """
    err = "false"  # 返回错误信息

    if temp == "":
        err = "请输入节点集合"
    elif err == "false":
        data = []
        method = int(method)
        # method -= 1
        s = ""
        for c in temp:
            if c == "，":
                s += ","
            else:
                s += c
        s = set(s.split(","))
        for c in s:
            if c.isdigit() and c != ',' and c != '，':
                c = int(c)
                if 0 <= c <= 104 and type(c) == int:
                    data.append(c)
                else:
                    err = "节点序号应为0-104的整数"
                    break
            else:
                err = "节点序号应为0-104的整数"
    if err != "false":
        active_records = json.dumps([])
        return active_records, err
    if method == 1:  # IC模型模拟一次
        active_node = set_influence(data, 1, networkWeight, number_of_nodes, edgeNum)  # 保存激活的节点
    elif method == 2:  # IC模型模拟十次
        active_node = set_influence_IC_10(data, 1, networkWeight, edgeNum)  # 保存激活的节点
    elif method == 3:  # LT模型模拟十次
        active_node = set_influence_LT_10(data, 1)  # 保存激活的节点
        active_num = 0
    elif method == 4:  # pageRank算法
        active_node = set_influence_pageRank(data, 1)  # 保存激活的节点
        active_num = 0
    elif method == 5:  # 最大度算法
        active_node = set_influence_degree(data, 1)  # 保存激活的节点
        active_num = 0
    active_records = json.dumps(active_node)
    return active_records, err


# 集合影响力对比
@app.route('/collectiveInfluenceComparison', methods=["GET", "POST"])  # 胡莎莎
def collectiveInfluenceComparison():
    global networkWeight
    global edgeNum

    if request.method == "GET":
        # 初始化边的权重
        networkWeight, edgeNum = init_networkWeight(networkTemp, number_of_nodes)
        err1 = "true"
        err2 = "true"
        active_records1 = json.dumps([])
        active_records2 = json.dumps([])
        seed1 = ""
        seed2 = ""
        method1 = ""
        method2 = ""
        return render_template('collectiveInfluenceComparison.html', graph_data=graph_data,
                               active_records1=active_records1, active_records2=active_records2, err1=err1, err2=err2,
                               seed1=seed1, seed2=seed2, method1=method1, method2=method2)
    else:
        temp1 = request.form.get('value1')
        method1 = request.form.get('method1')
        active_records1, err1 = calculateSingleSet(method1, temp1)
        seed1 = []
        s = ""
        for c in temp1:
            if c == "，":
                s += ","
            else:
                s += c
        s = set(s.split(","))
        for c in s:
            if c.isdigit() and c != ',' and c != '，':
                c = int(c)
                if 0 <= c <= 104 and type(c) == int:
                    seed1.append(c)

        temp2 = request.form.get('value2')
        method2 = request.form.get('method2')
        active_records2, err2 = calculateSingleSet(method2, temp2)

        seed2 = []
        s = ""
        for c in temp2:
            if c == "，":
                s += ","
            else:
                s += c
        s = set(s.split(","))
        for c in s:
            if c.isdigit() and c != ',' and c != '，':
                c = int(c)
                if 0 <= c <= 104 and type(c) == int:
                    seed2.append(c)

        if err2 != "false":
            err1 = err2
        return render_template('collectiveInfluenceComparison.html', graph_data=graph_data,
                               active_records1=active_records1, active_records2=active_records2, err1=err1, err2=err2,
                               seed1=seed1, seed2=seed2, method1=method1, method2=method2)


if __name__ == '__main__':
    app.run(debug=True)
