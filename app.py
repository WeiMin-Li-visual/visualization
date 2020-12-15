# -*- coding:utf-8 -*-
from flask import Flask, render_template, request, flash, jsonify
import json
import random
import copy
from flask_socketio import SocketIO
from threading import Lock
from flask_mail import Mail, Message
import pymysql

thread_lock = Lock()


# 初始化网络
# input: data_path
# output: network, node_num, graph_data
# 节点布局改进
# author: 邓志斌
# date: 2020年6月2日19:55:52
def init_network(data_path, opacity=1):
    """
    :param data_path: 数据所在路径
    :param opacity: 透明度，主要控制是否显示数据
    :return: 所有边的集合，节点数， 图数据
    """
    import networkx as nx

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

    G = nx.Graph(network)

    # # 让节点名称和索引相对应
    # nodes_list = list(nx.nodes(G))
    # nodes_list.sort()
    # node_dict = dict()
    # for i in range(node_num):
    #     node_dict[nodes_list[i]] = i

    # 记录每个节点的位置信息
    pos = nx.drawing.spring_layout(G, iterations=100, k=0.5)
    node_coordinate = []
    for i in range(node_num):
        node_coordinate.append([])
    for i, j in pos.items():
        # node_coordinate[node_dict[i]].append(float(j[0]))
        # node_coordinate[node_dict[i]].append(float(j[1]))
        node_coordinate[i - 1].append(float(j[0]))
        node_coordinate[i - 1].append(float(j[1]))

    # 设置传给前端的节点数据边数据的json串
    graph_data_json = {}
    nodes_data_json = []
    # for node in nodes_list:
    #     nodes_data_json.append({
    #         'attributes': {'modularity_class': 0},
    #         'id': str(node),
    #         'category': 0,
    #         'itemStyle': '',
    #         'label': {'normal': {'show': 'false'}},
    #         'name': str(node),
    #         'symbolSize': 35,
    #         'value': 111,
    #         'x': node_coordinate[node_dict[node]][0],
    #         'y': node_coordinate[node_dict[node]][1]
    #     })
    for node in range(node_num):
        nodes_data_json.append({
            'attributes': {'modularity_class': 0},
            'id': str(node),
            'category': 0,
            'itemStyle': {'opacity': opacity},
            'label': {'normal': {'show': 'false'}},
            'name': str(node),
            'symbolSize': 35,
            'value': 111,
            'x': node_coordinate[node][0],
            'y': node_coordinate[node][1]
        })
    links_data_json = []
    cur_edges = []
    for link in network:
        edge = [link[1], link[0]]
        if edge not in cur_edges:
            link_id = len(links_data_json)
            links_data_json.append({
                'id': str(link_id),
                'lineStyle': {'normal': {}},
                'name': 'null',
                'source': str(link[0] - 1),
                'target': str(link[1] - 1)
            })
            cur_edges.append(link)
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

    active = copy.deepcopy(seed)
    start = 0
    end = len(seed)
    edge_location = [-1 for i in seed]  # 记录激活边的序号
    while start != end:
        index = start
        while index < end:
            for i in range(number_of_nodes):
                if networkWeight[m][active[index]][i] != 0:
                    if i not in active and random.random() < networkWeight[m][active[index]][i]:
                        active.append(i)
                        edge_location.append(edgeNum[active[index]][i])  # 存储边对应的序号
            index += 1
        start = end
        end = len(active)
    # print('active_num',active_num)
    # print(active)
    return active, edge_location


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
    active_records = []  # 用来存放每个节点的模拟结果也就是最后激活的节点们
    edge_location = [-1 for i in seed]
    for simulation_count in range(0, 10):  # 模拟10次
        active_records.append([])
        edge_location.append([])
        active_records[simulation_count], edge_location[simulation_count] = set_influence(active, m, networkWeight,
                                                                                          number_of_nodes,
                                                                                          edgeNum)  # 把这个节点的模拟结果存起来

    return active_records, edge_location


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
        active_nodes = copy.deepcopy(node_set)  # 存放被激活的节点，初始为node_set
        start = 0
        end = len(active_nodes)
        edge_location = [-1 for i in node_set]  # 记录被激活边的位置
        while start != end:
            index = start
            while index < end:
                for nei_node in range(number_of_nodes):
                    if networkWeight[method][active_nodes[index]][nei_node] != 0 and nei_node not in active_nodes:
                        # 将邻居节点的阈值减去边上的权重，如果阈值小于0，那么节点被激活
                        theta[nei_node] -= networkWeight[method][active_nodes[index]][nei_node]
                        if theta[nei_node] <= 0:
                            active_nodes.append(nei_node)
                            edge_location.append(edgeNum[active_nodes[index]][nei_node])
                index += 1
            start = end
            end = len(active_nodes)
        return active_nodes, edge_location

    active_records = []  # 用来存放每个节点的模拟结果也就是最后激活的节点们
    method = 1  # 选择使用哪种权重进行
    # node_count = 0 记录当前在计算的节点个数，当列表下标用
    stimulate_round = 10  # 激活轮数
    count_influence = 0  # 记录当前节点10次模拟的影响力总和
    edge_records = []
    for round in range(stimulate_round):  # 重新设置每个节点的阈值
        theta = []  # 保存每个节点的阈值
        for iteration in range(number_of_nodes):  # 为每个节点随机设置阈值
            theta.append(random.random())
        l, k = set_influence_LT(active)
        active_records.append(l)  # 保存被激活的节点，第一个参数为列表
        edge_records.append(k)
        count_influence += len(l)

    # 这里返回一个三维的数组  这是节点集合为1时返回的数据
    # active_node: [[[1], [1, 3], [1], [1], [1], [1], [1], [1, 3, 11, 12, 22, 27, 17, 23, 24, 41,54], [1], [1]]]
    return active_records, edge_records


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
def set_influence_degree(seed, m, edgeNum):  # 胡莎莎
    """
    基于最大度算法计算一个节点集合激活的节点
    :param seed:种子节点集合
    :param m:选择哪种节点影响力进行传播
    :return active_records:被激活的节点集合
    """
    import numpy as np
    edge_records = [-1 for i in seed]
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
            if adjacencyMatrix[node][j] == 1:
                active_records.append(j)
                edge_records.append(edgeNum[node][j])
    return active_records, edge_records


def dyanmicMOACD_thread():
    """
    基于社交网络属性的多目标优化动态社区划分
    author:张财、邓志斌
    date:2020年6月6日
    """

    import math
    import networkx as nx
    import numpy as np
    from numpy import mat
    from numpy import trace

    # 求出度矩阵和B矩阵
    def deg_and_B(G):
        node_len = len(G.nodes)
        m2 = 2 * G.number_of_edges()
        # 构造邻接矩阵A
        A = np.zeros((node_len, node_len), int)
        for node in G.nodes:
            for nh in G.neighbors(node):
                A[node - 1][nh - 1] = 1
        # 度矩阵
        Degree = np.zeros(node_len, int)
        for i in list(G.nodes):
            k = sum(A[i - 1])
            Degree[i - 1] = k
        # 构造B矩阵
        B = np.zeros((node_len, node_len))
        for i in list(G.nodes):
            ki = Degree[i - 1]
            for j in list(G.nodes):
                kj = Degree[j - 1]
                B[i - 1][j - 1] = A[i - 1][j - 1] - ki * kj / m2
        # print(B)
        return Degree, B

    def cal_Q(partition, node_len, m2, B):
        s_cor = len(partition)
        S_labels = np.zeros([node_len, s_cor], int)
        for i in range(s_cor):
            for j in partition[i]:
                S_labels[j - 1][i] = 1
        S_labels = mat(S_labels)
        Q = 1 / m2 * (trace(S_labels.T * B * S_labels))
        return Q

    # 论文中3.3.2初始化：公式(3.7)
    def select_probability(G):
        node_set = list(G.nodes)  # 节点集合
        len_node_set = len(node_set)
        # 保存每个节点的邻居的个数
        # 前提是网络节点连续排好，中间没有缺失
        nodenh_set = [0] * len_node_set  # 每个节点的邻居集合
        nodenh_lenset = [0] * len_node_set  # 每个节点的邻居集合的长度
        for node in node_set:
            nh = list(G.neighbors(node))
            nodenh_lenset[node - 1] = len(nh)
            nodenh_set[node - 1] = list(G.neighbors(node))
        # i和j有边的共同邻居相似度
        select_pro = []
        for node in node_set:
            # 字典{邻居节点:共同邻居数,邻居节点:共同邻居数,...}
            sim_sum = 0  # 每个节点所有邻居相似度之和
            sim_set = []
            neis_set = nodenh_set[node - 1]
            for nei in neis_set:
                nei_neis_set = nodenh_set[nei - 1]
                # 交集
                d = [n for n in neis_set if n in nei_neis_set]
                # 计算相似度
                s = float(len(d) + 1) / math.sqrt(nodenh_lenset[node - 1] * nodenh_lenset[nei - 1])
                sim_set.append(s)
                sim_sum += s
            nei_pro = {}
            for i in range(nodenh_lenset[node - 1]):
                pro = sim_set[i] / sim_sum
                nei_pro.update({neis_set[i]: pro})
            select_pro.append([node, nei_pro])
        select_pro = sorted(select_pro, key=(lambda x: x[0]))
        return select_pro

    # 计算论文公式(3.10)
    def silhouette(G, partition):
        GS = 0.0
        for index, community in enumerate(partition):
            result = 0.0
            community_len = len(community)
            for i in range(community_len):
                # 求ai---每个点与社区内其他点平均相似度
                degree = sum([1 for j in range(community_len) if G.has_edge(community[i], community[j])])
                ai = degree / community_len
                # 求bi --节点i与其他社区节点最大平均相似度
                other_sim = [sum([1 for j in range(len(other_community)) if
                                  G.has_edge(community[i], other_community[j])]) / len(other_community)
                             for index1, other_community in enumerate(partition) if index1 != index]
                if other_sim:
                    bi = max(other_sim)
                else:
                    bi = 0  # 有问题
                if max(ai, bi) == 0:
                    result += 0
                else:
                    result += (ai - bi) / max(ai, bi) / community_len
            GS += result / len(partition)
        return GS

    # 帕累托最优解集合
    def pareto(pop_value):
        '''
        # 非支配规则:a所有目标值不小于b,并且至少有一个目标值比b大，则a支配b
        :param pop_value:
        :return:
        '''

        non = []
        # 将解集复制给非支配解集
        for i in range(len(pop_value)):
            non.append(pop_value[i][:])
        for i in range(-len(non), -1):
            for j in range(i + 1, 0):
                # print non[i], non[j]
                if non[i][1] == non[j][1] and non[i][2] == non[j][2]:
                    # print i,"deng"
                    non.remove(non[i])
                    # print non
                    break
                elif ((non[i][1] >= non[j][1]) and (non[i][2] >= non[j][2])) and (
                        (non[i][1] > non[j][1]) or (non[i][2] > non[j][2])):
                    # a[i][0]支配a[j][0]
                    # print "ai支配aj"
                    non.remove(non[j])
                    break
                elif ((non[j][1] >= non[i][1]) and (non[j][2] >= non[i][2])) and (
                        (non[j][1] > non[i][1]) or (non[j][2] > non[i][2])):
                    # print "aj支配ai"
                    non.remove(non[i])
                    break
        return non

    # 基因漂流，就是论文算法1邻居从众策略
    def turbulence(G, indi):
        G3 = nx.Graph()  # 解码后（稀疏）的图
        for node_cp in indi:
            G3.add_edge(node_cp[0], node_cp[1])
        components = [list(c) for c in list(nx.connected_components(G3))]
        components_len = len(components)
        for j in range(len(indi)):
            node = indi[j][0]
            nei = [n for n in G.neighbors(node)]
            # 记录邻居的集群标签
            nei_com = []  # 邻居集群的位置
            for x in range(len(components)):
                nei_com.append([n for n in nei if n in components[x]])
            # 选出最多邻居的集群
            max_nei_com = nei_com[int(np.argmax([len(nei_com[m]) for m in range(len(nei_com))]))]
            # 从最多数量的邻居分区中随机选一个值
            if indi[j][1] not in max_nei_com:
                indi[j] = [node, random.sample(max_nei_com, 1)[0]]
        return indi

    # 基因变异，就是论文算法2邻居多样策略
    def mutation(G, indi):
        for j in range(len(indi)):
            node = indi[j][0]
            t = [n for n in G.neighbors(node)]
            t.remove(indi[j][1])
            if t:
                indi[j] = [node, random.sample(t, 1)[0]]

        return indi

    # 初始社区划分方案，方案个数为100
    def init_community(G, B, N=100):
        node_num = G.number_of_nodes()
        m2 = 2 * G.number_of_edges()
        pop_ns = []  # 社区的节点邻接表示集合，键值对
        pop_partition = []  # 社区的分区方案集合
        pop_value = []  # 所有方案的目标结果集合
        # 计算节点之间的选择概率
        select_pro = select_probability(G)
        # 生成N中划分方案
        for i in range(N):
            G2 = nx.Graph()  # 邻接表示解码后的图
            indi = []  # 每个节点的基于轨迹的邻接表示
            # 生成划分方案
            for m in range(node_num):
                nei_pro = select_pro[m][1]  # 节点m+1的所有邻居节点的选择概率
                # 按照概率随机选择一个节点
                ran = random.random()  # 生成一个0-1的随机数
                rate_sum = 0
                select_node = 1  # 被节点m+1选中的邻居节点
                for neighbor, rate in nei_pro.items():
                    rate_sum += rate
                    if ran < rate_sum:
                        select_node = neighbor
                        break
                indi.append([m + 1, select_node])
                G2.add_edge(m + 1, neighbor)
            pop_ns.append(indi)
            components = [list(c) for c in list(nx.connected_components(G2))]  # G2的分区
            Q = cal_Q(components, node_num, m2, B)  # 计算该划分方案的模块度
            GS = silhouette(G, components)  # 计算GS值
            pop_partition.append(components)
            pop_value.append([i, Q, GS])
        return pop_ns, pop_partition, pop_value

    # 动态划分的时候需要调用
    def staticMoacd(path_e, N=100):
        '''
        静态MOACD算法
        :param path_e: 文件路径
        :param N: 种群数量，默认100
        :return:
        '''

        pop_N = N  # 群体个数
        network_synfix, num_nodes_synfix, graph_data_synfix = init_network(path_e)
        socketio.emit('server_response',
                      {'data': [graph_data_synfix, 0], 'count': 0},
                      namespace='/dyanmicMOACD')

        G = nx.Graph()  # 图数据
        for edge in network_synfix:
            G.add_edge(edge[0], edge[1])
        node_num = G.number_of_nodes()
        edge_num = G.number_of_edges()
        m2 = edge_num * 2

        Degree, B = deg_and_B(G)  # 度矩阵和B矩阵（B矩阵用于模块度计算）

        # 初始社区划分方案
        pop_ns, pop_partition, pop_value = init_community(G, B, pop_N)

        # 求解帕累托最优解
        rep_value = pareto(pop_value)

        rep_ns = []  # 帕累托最优解的节点邻接表示集合
        rep_partition = []  # 帕累托最优解的分区方案集合
        for i in range(len(rep_value)):
            j = rep_value[i][0]
            rep_ns.append(pop_ns[j])
            rep_partition.append(pop_partition[j])

        # 迭代更新解决方案
        genmax = 20  # 最大迭代次数
        gen = 0  # 当前迭代次数
        best_gen = []  # 迭代过程中最好的值
        gen_equal = 0  # 多目标值相等的次数

        while gen < genmax and gen_equal < 10:
            npop_ns = copy.deepcopy(pop_ns)  # 深拷贝
            npop_good_value = []  # 更新后较好的多目标值
            npop_good_partition = []  # 更新后较好的划分方案
            nrep_value = []  # 新的帕累托最优多目标值
            nrep_ns = []  # 新的帕累托节点邻接表示
            nrep_partition = []  # 新的帕累托社区划分方案
            # update_num = 10  # 每次迭代的更新次数

            for i in range(pop_N):
                npop_ns[i] = random.sample(rep_ns, 1)[0]  # 随机从帕累托前沿中选择一个方案
                t_ns = copy.deepcopy(npop_ns[i])  # 暂存当前选中的方案

                # 随机选择一种算法更新
                if random.random() < 0.2:
                    npop_ns[i] = mutation(G, npop_ns[i])  # 邻居多样策略
                else:
                    npop_ns[i] = turbulence(G, npop_ns[i])  # 邻居从众策略

                # 判断是否新值旧值支配情况
                G_new = nx.Graph()
                for node_cp in npop_ns[i]:
                    G_new.add_edge(node_cp[0], node_cp[1])

                components = [list(c) for c in list(nx.connected_components(G_new))]
                Q = cal_Q(components, node_num, m2, B)
                GS = silhouette(G, components)

                # 判断是否新值旧值支配情况  新值不被旧值支配, 且不等于旧址
                if (Q <= pop_value[i][1] and GS <= pop_value[i][2]) and (Q < pop_value[i][1] or GS < pop_value[i][2]):
                    no_dominate = False
                else:
                    no_dominate = True
                # 支配，新值直接输出，否则，改为旧值
                if no_dominate:
                    npop_good_value.append([i + (1 + gen) * 100, Q, GS])  # 新值加编号100处理
                    npop_good_partition.append(components)
                else:
                    npop_ns[i] = pop_ns[i]

            npop_gv_i = []
            for i in range(len(npop_good_value)):
                npop_gv_i.append(npop_good_value[i][0])

            # 帕累托最优解集合
            all_value = rep_value + npop_good_value
            # nrep_value = pareto(all_value)
            nrep_value = pareto(all_value)

            # rep_value 位置
            rep_gv_i = []
            for i in range(len(rep_value)):
                rep_gv_i.append(rep_value[i][0])
            for i in range(len(nrep_value)):
                j = nrep_value[i][0]
                if j < 100 * (gen + 1):
                    nrep_ns.append(rep_ns[rep_gv_i.index(j)])
                    nrep_partition.append(rep_partition[rep_gv_i.index(j)])
                else:
                    nrep_ns.append(npop_ns[j % 100])
                    nrep_partition.append(npop_good_partition[npop_gv_i.index(j)])

            rep_ns = nrep_ns
            rep_value = nrep_value
            rep_partition = nrep_partition

            # 迭代终止条件
            best_gen.append(sorted(rep_value, key=lambda x: x[1], reverse=True)[0])
            if gen >= 1 and best_gen[gen][1:] == best_gen[gen - 1][1:]:
                gen_equal += 1
            gen += 1

        best = sorted(rep_value, key=lambda x: x[1], reverse=True)[0]
        best_location = rep_value.index(best)
        best_partition = rep_partition[best_location]

        return best_partition, G, graph_data_synfix

    # 计算两图的节点交集数量
    def cal_com_node(GA, GB):
        node_set_a = GA.nodes()
        node_set_b = GB.nodes()
        com_num = 0
        for i in node_set_a:
            if i in node_set_b:
                com_num += 1
        return com_num

    # 计算公式(3.11),DCEC值
    def dcec(components_pre, components, G_pre_nn, G_nn, com_nn):
        '''
        :param components_pre:
        :param components:
        :param G_pre_nn:
        :param G_nn:
        :param com_nn:
        :return: 返回公式(3.11),DCEC值
        '''
        A = components_pre
        B = components
        NA = G_pre_nn
        NB = G_nn
        NC = com_nn
        Nall = G_pre_nn + G_nn - com_nn
        C = []
        fenzi = 0
        fenmu1 = 0
        fenmu2 = 0
        nmi = 0
        len_A = len(A)
        len_B = len(B)

        # 计算C矩阵/列表
        C = [[sum([1 for node in A[i] if node in B[j]]) for j in range(len_B)] for i in range(len_A)]

        # 求分子，分母
        for i in range(len_A):
            ci = sum([C[i][j] for j in range(len_B)])
            if ci != 0:
                fenmu1 += float(ci) / NA * math.log(float(ci) / NA)  #
            else:
                fenmu1 += 0
            for j in range(len_B):
                cj = sum([C[i][j] for i in range(len_A)])
                if cj != 0:
                    fenmu2 += float(cj) / NB * math.log(float(cj) / NB)  #
                else:
                    fenmu2 += 0

                if C[i][j] != 0 and ci != 0 and cj != 0:
                    fenzi += float(C[i][j]) / Nall * math.log(float(C[i][j] * NC) / (ci * cj))
                else:
                    fenzi += 0

        nmi = -2.0 * fenzi / (fenmu1 + fenmu2)
        return nmi

    def iteration(G, pop_ns, pop_value, rep_ns, rep_value, rep_partition, components_pre, G_pre_nn, G_nn, com_nn, m2, B,
                  pop_N):

        N = pop_N  # 个体个数
        genmax = 20  # 最大迭代次数
        gen = 0
        best_gen = []  # 迭代过程中，最好的值
        gen_equal = 0  # 值相等次数

        while gen < genmax and gen_equal < 10:  # 迭代次数，蝙蝠时间t
            npop_ns = copy.deepcopy(pop_ns)  # 深拷贝，py3
            npop_good_value = []  # 新的npop比较好的值的集合
            npop_good_partition = []  # 新的比较好的值的分区集合
            nrep_value = []  # 较优值集合
            nrep_ns = []  # 节点集
            nrep_partition = []  # 较优值分区集合

            for i in range(N):  # 群体个数-
                # 随机从rep里选取一个bat作为最佳值
                best_x = random.sample(rep_ns, 1)[0]
                npop_ns[i] = best_x

                # 根据生成的概率进行变异or湍流
                if random.random() < 0.2:
                    npop_ns[i] = mutation(G, npop_ns[i])
                else:
                    npop_ns[i] = turbulence(G, npop_ns[i])

                # 判断是否新值旧值支配情况
                G_new = nx.Graph()
                for node_cp in npop_ns[i]:
                    G_new.add_edge(node_cp[0], node_cp[1])

                components = [list(c) for c in list(nx.connected_components(G_new))]

                Q = cal_Q(components, G_nn, m2, B)
                a = dcec(components_pre, components, G_pre_nn, G_nn, com_nn)
                nmi = round(a, 10)

                # 判断是否新值旧值支配情况  新值不被旧值支配, 且不等于旧址
                if (Q <= pop_value[i][1] and nmi <= pop_value[i][2]) and (Q < pop_value[i][1] or nmi < pop_value[i][2]):
                    no_dominate = False
                else:
                    no_dominate = True

                # 支配，新值直接输出，否则，改为旧值
                if no_dominate:
                    npop_good_value.append([i + (1 + gen) * 100, Q, nmi])  # 新值加编号100处理
                    npop_good_partition.append(components)
                else:
                    npop_ns[i] = pop_ns[i]

            npop_gv_i = []
            for i in range(len(npop_good_value)):
                npop_gv_i.append(npop_good_value[i][0])

            # 帕累托最优解集合
            all_value = rep_value + npop_good_value
            nrep_value = pareto(all_value)

            rep_gv_i = []
            for i in range(len(rep_value)):
                rep_gv_i.append(rep_value[i][0])
            for i in range(len(nrep_value)):
                j = nrep_value[i][0]
                if j < 100 * (gen + 1):
                    nrep_ns.append(rep_ns[rep_gv_i.index(j)])
                    nrep_partition.append(rep_partition[rep_gv_i.index(j)])
                else:
                    nrep_ns.append(npop_ns[j % 100])
                    nrep_partition.append(npop_good_partition[npop_gv_i.index(j)])

            # 更新rep
            rep_ns = nrep_ns
            rep_value = nrep_value
            rep_partition = nrep_partition
            # 迭代终止条件
            best_gen.append(sorted(rep_value, key=lambda x: x[1], reverse=True)[0])
            if gen >= 1 and best_gen[gen][1:] == best_gen[gen - 1][1:]:
                gen_equal += 1
            gen += 1

        # 选择模块度最大的一个作为最终结果
        best = sorted(rep_value, key=lambda x: x[1], reverse=True)[0]
        best_location = rep_value.index(best)

        # 某一最好的结果的分区
        best_partition = rep_partition[best_location]
        return best_partition

    pop_N = 50
    T = 11  # 总时间步 一共T-1个时间步

    file_qian = "static/data/synfix/z_3/synfix_3.t"  # 文件名前缀
    # file_qian = "static/data/Cell/real.t"  # 文件名前缀
    file_bian = ".edges"  # 后缀--边

    # 保存每一时间片的社区划分结果
    all_best_components = []

    # 保存所有时间片的网络结构
    all_graph_data = []

    # 第一个时间步
    # result_pre:最优社区划分
    # G_pre：前一个网络图
    # graph_data：网络图数据
    components_pre, G_pre, graph_data = staticMoacd(file_qian + "01" + file_bian, pop_N)
    G_pre_nn = G_pre.number_of_nodes()  # 前一个时间片图的节点个数

    socketio.emit('server_response',
                  {'data': [graph_data, components_pre], 'count': 1},
                  namespace='/dyanmicMOACD')

    all_graph_data.append(graph_data)
    all_best_components.append(components_pre)

    # 之后的时间步
    for t in range(2, T):
        count = t
        if t < 10:
            t = "0" + str(t)

        network_synfix, num_nodes_synfix, graph_data_synfix = init_network(file_qian + str(t) + file_bian)
        all_graph_data.append(graph_data_synfix)

        G = nx.Graph()  # 图数据

        for edge in network_synfix:
            G.add_edge(edge[0], edge[1])

        G_nn = G.number_of_nodes()
        com_nn = cal_com_node(G_pre, G)  # 计算两个图相同节点的数量
        edge_num = G.number_of_edges()
        m2 = edge_num * 2

        Degree, B = deg_and_B(G)  # 度矩阵和B矩阵（B矩阵用于模块度计算）

        # 初始社区划分方案
        pop_ns, pop_partition, pop_value = init_community(G, B, pop_N)

        # 求解帕累托最优解
        rep_value = pareto(pop_value)
        rep_ns = []  # 帕累托最优解的节点邻接表示集合
        rep_partition = []  # 帕累托最优解的分区方案集合
        for i in range(len(rep_value)):
            j = rep_value[i][0]
            rep_ns.append(pop_ns[j])
            rep_partition.append(pop_partition[j])

        # 最好的结果的分区
        best_partition = iteration(G, pop_ns, pop_value, rep_ns, rep_value, rep_partition, components_pre, G_pre_nn,
                                   G_nn, com_nn,
                                   m2, B, pop_N)

        # 这一时间步的图为下一时间步的图比较的对象，这一时间步的分区结果设置为下一个前一个分区结果
        G_pre = G
        G_pre_nn = G_pre.number_of_nodes()  # 前一个时间片图的节点个数
        components_pre = best_partition
        all_best_components.append(components_pre)

        socketio.emit('server_response',
                      {'data': [graph_data_synfix, components_pre], 'count': count},
                      namespace='/dyanmicMOACD')

    # 最后接收一次空数据，前端显示“演化结束”
    socketio.sleep(1)  # 休眠1秒
    socketio.emit('server_response',
                  {'data': [0, 0], 'count': 0},
                  namespace='/dyanmicMOACD')


async_mode = None
thread = None
app = Flask(__name__)
app.secret_key = 'lisenzzz'
socketio = SocketIO(app, async_mode=async_mode)
path = 'static/data/synfix/z_3/synfix_3.t01.edges'
path1 = 'static/data/Wiki.txt'
networkTemp, number_of_nodes, graph_data = init_network(path1)
network_synfix, num_nodes_synfix, graph_data_synfix = init_network(path)
connection = pymysql.connect(host='localhost',  # host属性
                             user='root',  # 用户名
                             password='mysql',  # 此处填登录数据库的密码
                             db='mysql'  # 数据库名
                             )
cur = connection.cursor()
cur.execute('use logindata')
app.config['MAIL_SERVER'] = 'smtp.qq.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = '719723236@qq.com'
app.config['MAIL_PASSWORD'] = 'tlwiuoueauapbefb'
mail = Mail(app)


@app.route('/checkUser', methods=["POST"])
def checkUser():
    if request.method == 'POST':
        cur.execute('use logindata')
        requestArgs = request.values
        user = requestArgs.get('user')
        cur.execute("select * from udata where user = " + "'" + user + "'")
        result = cur.fetchone()
        if result is None:
            return jsonify({'isExist': False})
        elif result is not None:
            return jsonify({'isExist': True})


@app.route('/forget', methods=["GET", "POST"])
def forget():
    if request.method == 'GET':
        return render_template('forget.html')
    elif request.method == 'POST':
        requestArgs = request.values
        new = requestArgs.get('password')
        user = requestArgs.get('user')
        cur.execute("update udata set password=MD5('" + new + "') where user='" + user + "'")
        connection.commit()
        return jsonify({'isSuccess': 1})


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    elif request.method == 'POST':
        requestArgs = request.values
        user = requestArgs.get('user')
        password = requestArgs.get('password')
        number = requestArgs.get('number')
        unit = requestArgs.get('unit')
        mail = requestArgs.get('mail')
        str = "'" + user + "'" + ",'" + number + "'," + "'" + mail + "'" + "," \
              + "'" + unit + "'" + "," + "MD5('" + password + "')"
        cur.execute('insert into udata (user,number,mail,unit,password) values (' + str + ")")
        connection.commit()
        return jsonify({'isSuccess': 1})


@app.route('/send', methods=["POST"])
def send():
    requestArgs = request.values
    dirMail = requestArgs.get('mail')
    user = requestArgs.get('user')
    if user is not None:
        cur.execute("select * from udata where user = " + "'" + user + "'")
        result = cur.fetchone()
        if result[3] != dirMail:
            return jsonify({'ischecked': 0})
    verificationList = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                        'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 's', 't', 'x', 'y', 'z', 'A', 'B', 'C', 'D',
                        'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'S', 'T', 'X', 'Y', 'Z']
    veriCode = ''
    for i in range(4):
        veriCode += verificationList[random.randint(0, len(verificationList) - 1)]
    msg = Message("可视化平台验证码", sender="719723236@qq.com", recipients=[dirMail])
    msg.body = veriCode
    mail.send(msg)
    return jsonify({'code': veriCode, 'ischecked': 1})


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == 'GET':
        cur.execute('use logindata')
        return render_template('login.html')
    elif request.method == 'POST':
        requestArgs = request.values
        user = requestArgs.get('user')
        password = requestArgs.get('password')
        cur.execute("select * from udata where user = " + "'" + user + "'")
        result = cur.fetchone()  # 没找到为None, 否则返回对应的元组
        cur.execute("select md5('" + password + "')")
        p = cur.fetchone()  # 返回的是三元组，p[0]是需要的值
        check = {'userInfo': -1, 'passwordInfo': -1}
        if result is None:
            check['userInfo'] = -1
        elif result is not None:
            check['userInfo'] = 0
            if p[0] == result[1]:
                check['passwordInfo'] = 1
            elif p[0] != result[1]:
                check['passwordInfo'] = 0
        check = json.dumps(check)
        return jsonify({'check': check})


@app.route('/fun', methods=["POST"])
def fun():
    requestArgs = request.values
    user = requestArgs.get('userName')
    cur.execute("select * from udata where user = " + "'" + user + "'")
    result = cur.fetchone()
    return jsonify({'user': result})


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
    edge_records = []  # 记录激活边在networkTemp中的序号
    graph_data1 = json.loads(graph_data)  # 将json数据转化为字典的形式
    for node in range(number_of_nodes):
        active_records.append([])
        active_nums.append([])
        edge_records.append([])
        active_records[node], edge_records[node] = set_influence([node], 1, networkWeight, number_of_nodes,
                                                                 edgeNum)  # 把这个节点的模拟结果存起来
        influence = len(active_records[node])
        graph_data1['nodes'][node]['value'] = influence  # 使图中各个节点右下角显示节点的影响力大小
        if influence > max_node_influence:
            max_node_influence = influence
            max_influence_node = node
    active_records = json.dumps(active_records)
    graph_data1 = json.dumps(graph_data1)  # 将数据转化为json格式
    # 把需要的数据给对应的页面
    return render_template('common_template.html', graph_data=graph_data1, active_records=active_records,
                           max_node_influence=max_node_influence, edge_records=edge_records,
                           max_influence_node=max_influence_node, method_type=1)


# 选择单个影响力最大的种子基于ic模型（每个节点模拟十次）
@app.route('/basicIc10')
def basic_ic_10():  # 胡莎莎
    # 初始化权重矩阵
    networkWeight, edgeNum = init_networkWeight(networkTemp, number_of_nodes)

    # 执行贪心算法找影响力最大的节点
    active_records = []  # 用来存放每个节点的模拟结果也就是最后激活的节点们
    max_node_influence = 0  # 用来存放比较过程中当前最大的影响力
    active_nums = []  # 每个节点每次模拟激活的节点数
    graph_data1 = json.loads(graph_data)
    edge_records = []
    for node in range(number_of_nodes):
        active_records.append([])
        active_nums.append([])
        edge_records.append([])
        influence = 0
        for simulation_count in range(0, 10):  # 模拟10次
            active_records[node].append([])
            active_nums[node].append([])
            edge_records[node].append([])
            active_records[node][simulation_count], edge_records[node][simulation_count] = \
                set_influence([node], 1, networkWeight, number_of_nodes, edgeNum)  # 把这个节点的模拟结果存起来
            influence += len(active_records[node][simulation_count])
        graph_data1['nodes'][node]['value'] = influence / 10  # 模拟十次的平均影响力
        if influence > max_node_influence:
            max_node_influence = influence
            max_influence_node = node
    active_records = json.dumps(active_records)
    # 把你需要的数据给对应的页面
    max_node_influence /= 20  # 求平均值
    graph_data1 = json.dumps(graph_data1)
    return render_template('common_template.html', graph_data=graph_data1, active_records=active_records,
                           edge_records=edge_records, max_node_influence=max_node_influence,
                           max_influence_node=max_influence_node, method_type=2)


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
        edge_location = [-1 for i in node_set]  # 记录被激活边的位置
        while start != end:
            index = start
            while index < end:
                for nei_node in range(number_of_nodes):
                    if networkWeight[method][active_nodes[index]][nei_node] != 0 and nei_node not in active_nodes:
                        # 将邻居节点的阈值减去边上的权重，如果阈值小于0，那么节点被激活
                        theta[nei_node] -= networkWeight[method][active_nodes[index]][nei_node]
                        if theta[nei_node] <= 0:
                            active_nodes.append(nei_node)
                            edge_location.append(edgeNum[active_nodes[index]][nei_node])
                index += 1
            start = end
            end = len(active_nodes)
        return active_nodes, edge_location

    # 基于LT模型，找影响力最大的节点
    active_records = []  # 用来存放每个节点的模拟结果也就是最后激活的节点们
    max_node_influence = 0  # 用来存放比较过程中当前最大的影响力
    method = 1  # 选择使用哪种权重进行
    edge_records = []
    graph_data1 = json.loads(graph_data)
    for node in range(number_of_nodes):  # 遍历所有的节点，判断影响力
        active_records.append([])
        edge_records.append([])
        stimulate_round = 10  # 激活轮数
        count_influence = 0  # 记录当前节点10次模拟的影响力总和
        for round in range(stimulate_round):  # 重新设置每个节点的阈值
            theta = []  # 保存每个节点的阈值
            for iteration in range(number_of_nodes):  # 为每个节点随机设置阈值
                theta.append(random.random())
            l1, l2 = set_influence_LT([node])
            active_records[node].append(l1)  # 保存被激活的节点，第一个参数为列表
            edge_records[node].append(l2)
            count_influence += len(l1)
        influence = count_influence / (stimulate_round * 2)
        graph_data1['nodes'][node]['value'] = influence
        if influence > max_node_influence:
            max_node_influence = influence
            max_influence_node = node

    active_records = json.dumps(active_records)
    graph_data1 = json.dumps(graph_data1)
    # 把你需要的数据给对应的页面
    return render_template('common_template.html', graph_data=graph_data1, active_records=active_records,
                           edge_records=edge_records, max_node_influence=max_node_influence,
                           max_influence_node=max_influence_node, method_type=2)


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
    edge_records = []  # 存放激活边的位置
    for i in range(number_of_nodes):
        active_records.append([])
        edge_records.append([-1])
        active_records[i].append(i)  # 将当前节点放入激活列表中
        for j in range(number_of_nodes):
            if (adjacencyMatrix[i][j] == 1):
                active_records[i].append(j)
                edge_records[i].append(edgeNum[i][j])
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
                           edge_records=edge_records, max_node_influence=max_node_influence,
                           max_influence_node=max_influence_node, method_type=1)


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
        active_node, edge_records = set_influence(data, 1, networkWeight, number_of_nodes, edgeNum)  # 保存激活的节点
        active_records = json.dumps(active_node)
        return render_template('input.html', graph_data=graph_data, active_records=active_records,
                               edge_records=edge_records, err=err)


# 计算在选定算法下单个集合的影响力
def calculateSingleSet(method, temp, edgeNum):
    """
    计算在选定算法下单个集合的影响力
    :param method:选择的算法
    :param temp:输入的种子集合
    :param edgeNum:每条边所在位置
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
        edge_records = []
        active_records = json.dumps([])
        return active_records, err, edge_records
    if method == 1:  # IC模型模拟一次
        active_node, edge_records = set_influence(data, 1, networkWeight, number_of_nodes, edgeNum)  # 保存激活的节点
    elif method == 2:  # IC模型模拟十次
        active_node, edge_records = set_influence_IC_10(data, 1, networkWeight, edgeNum)  # 保存激活的节点
    elif method == 3:  # LT模型模拟十次
        active_node, edge_records = set_influence_LT_10(data, 1)  # 保存激活的节点
        # active_num = 0
    elif method == 4:  # pageRank算法
        edge_records = []
        active_node = set_influence_pageRank(data, 1)  # 保存激活的节点
        # active_num = 0
    elif method == 5:  # 最大度算法
        active_node, edge_records = set_influence_degree(data, 1, edgeNum)  # 保存激活的节点
        # active_num = 0
    active_records = json.dumps(active_node)
    return active_records, err, edge_records


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
        edge_records1 = json.dumps([])
        edge_records2 = json.dumps([])
        method1 = ""
        method2 = ""
        return render_template('collectiveInfluenceComparison.html', graph_data=graph_data,
                               active_records1=active_records1, active_records2=active_records2, err1=err1, err2=err2,
                               edge_records1=edge_records1, edge_records2=edge_records2, seed1=seed1, seed2=seed2,
                               method1=method1,
                               method2=method2)
    else:
        temp1 = request.form.get('value1')
        method1 = request.form.get('method1')
        active_records1, err1, edge_records1 = calculateSingleSet(method1, temp1, edgeNum)
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
        active_records2, err2, edge_records2 = calculateSingleSet(method2, temp2, edgeNum)

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
                               seed1=seed1, seed2=seed2, edge_records1=edge_records1, edge_records2=edge_records2,
                               method1=method1, method2=method2)


@app.route('/communityECDR')
def ECDR():
    from scipy.linalg import qr, svd, pinv
    import pandas as pd
    import numpy as np

    node_num = num_nodes_synfix  # 网络中节点的数量
    graph_edge = json.loads(graph_data_synfix)['links']
    edgeNum = [[0 for i in range(node_num)] for j in range(node_num)]
    for i in range(len(graph_edge)):
        source = int(graph_edge[i]['source'])
        target = int(graph_edge[i]['target'])
        edgeNum[source][target] = i
        edgeNum[target][source] = i
    # 返回所有的边，以列表的形式[[],[]...]
    edges = network_synfix

    A = np.zeros([node_num, node_num], dtype=int)  # 网络G的邻接矩阵A
    for i in range(len(edges)):
        A[int(edges[i][0]) - 1][int(edges[i][1]) - 1] = 1  # 数据中的节点是从1开始的，而矩阵是从0开始的
        A[int(edges[i][1]) - 1][int(edges[i][0] - 1)] = 1

    # 度的节点矩阵,返回各个节点的度
    D1 = []  # 存放各个节点的度D1
    D = np.zeros([node_num, node_num], dtype=int)  # 度的节点矩阵D
    for i in range(len(A)):
        D1.append(sum(A[i]))
    for i in range(len(D1)):
        D[i][i] = D1[i]

    # 电阻矩阵
    def ResistorMatrix():
        L = D - A  # 图状网络G的拉普拉斯矩阵L
        L1 = pinv(L)  # L的摩尔-彭若斯逆
        Q = np.zeros([node_num, node_num])  # 电阻矩阵Q
        for i in range(len(Q)):
            for j in range(len(Q[0])):
                Q[i][j] = L1[i][i] + L1[j][j] - 2 * L1[i][j]
        return Q

    # 两个节点有共同连接节点的矩阵
    def CommonNode():
        CommonNode = np.zeros([node_num, node_num], dtype=int)
        for i in range(len(CommonNode) - 1):
            for j in range(i + 1, len(CommonNode[0])):
                count = 0
                for t in range(i + 1, len(CommonNode[0])):
                    if A[i][t] == A[j][t] == 1:
                        count = count + 1
                CommonNode[i][j] = count
                CommonNode[j][i] = count
        return CommonNode

    # 邻居节点间的距离
    def NodeDistance():
        ComNode = CommonNode()  # 公共节点矩阵
        Q = ResistorMatrix()  # 电阻矩阵
        Dis = np.zeros([node_num, node_num])  # 邻居节点的距离
        for i in range(len(Dis)):
            for j in range(len(Dis[0])):
                dd = min((1 - ComNode[i][j] / D[i][i]), (1 - ComNode[i][j] / D[j][j]))
                Dis[i][j] = Q[i][j] * dd
        return Dis

    # 计算局部亲密邻居阈值矩阵
    def Threshold():
        Dis = NodeDistance()  # 邻居节点间的距离
        thres = np.zeros([node_num, node_num])  # 各个节点的阈值
        for i in range(len(thres)):
            s = 1
            for j in range(len(thres[0])):
                if A[i][j] == 1:
                    s = s * Dis[i][j]
            thres[i][i] = s ** (1 / D[i][i])
        return thres

    # 节点在t时刻的局部紧密邻居
    thres = Threshold()  # 局部亲密邻居阈值
    LCN_Matrix = copy.deepcopy(A)  # 邻接矩阵
    Dis = NodeDistance()  # 邻居节点间的距离
    for i in range(len(LCN_Matrix)):
        for j in range(len(LCN_Matrix[0])):
            if LCN_Matrix[i][j] == 1 and (Dis[i][j] <= thres[i][i] or Dis[i][j] <= thres[j][j]):
                LCN_Matrix[i][j] = LCN_Matrix[j][i] = 2  # 亲密邻居？

    # 每个节点局部最小聚类阈值
    def LocalMinimumClusteringThreshold(a):
        LMCT = []
        for i in range(len(D1)):
            LMCT.append(D1[i] * a)
        return LMCT

    def CoreNode():
        LMCT = LocalMinimumClusteringThreshold(0.7)  # 每个节点局部最小聚类阈值
        LCN = []  # 存放每个节点的局部邻居数量
        # LCN_Matrix=LocalCloseNeighbors()#节点在t时刻的局部紧密邻居矩阵
        for i in range(len(LCN_Matrix)):
            count = 0
            for j in range(len(LCN_Matrix[0])):
                if LCN_Matrix[i][j] == 2:
                    count = count + 1
            LCN.append(count)
        core_node = []
        for i in range(len(LCN)):
            if LCN[i] >= LMCT[i]:
                core_node.append(i)
        return core_node

    def CommunityDivision():
        communityEdge = []  # 保存核心节点与局部亲密邻居的边
        codenode = CoreNode()  # 获得所有的核心节点
        # 计算所有核心节点的局部亲密邻居
        LocalClosedNeighbors = {}
        for node in codenode:
            LocalClosedNeighbors[node] = []
            for i in range(len(LCN_Matrix)):
                if LCN_Matrix[node][i] == 2:
                    LocalClosedNeighbors[node].append(i)
        community = []
        cNum = -1
        unvisited = [0 for i in range(node_num)]
        for node in codenode:
            if unvisited[node] == 0:  # 核心节点未被访问，说明存在新社区
                community.append([])
                communityEdge.append([])
                cNum += 1
            else:
                continue
            coreList = [node]
            community[cNum].append(-1 * node)
            communityEdge[cNum].append([node, node])
            unvisited[node] = 1
            index = 0
            while index < len(coreList):
                corenode = coreList[index]
                closedNei = LocalClosedNeighbors[corenode]
                for n in closedNei:  # 遍历所有的局部亲密邻居，将核心节点放入coreList中，将不在社区中的节点加入相应社区
                    if n in codenode:
                        if n not in coreList:
                            coreList.append(n)
                        n *= -1
                    if unvisited[abs(n)] == 0:
                        community[cNum].append(n)
                        communityEdge[cNum].append([corenode, abs(n)])
                        unvisited[abs(n)] = 1
                index += 1
        return community, communityEdge

    c, CommunityEdge = CommunityDivision()
    C = []
    communityEdge = []
    for cNum in range(len(c)):
        if len(c[cNum]) >= 3:
            C.append(c[cNum])
            communityEdge.append(CommunityEdge[cNum])
    graph_data1 = json.loads(graph_data_synfix)
    graph_data1 = json.dumps(graph_data1)
    return render_template('EDCR.html', graph_data=graph_data1, community=C,
                           communityEdge=communityEdge, edgeNum=edgeNum)


@app.route('/communityEvolutionECDR')
def ECDR_Evolution():
    from scipy.linalg import qr, svd, pinv
    import pandas as pd
    import numpy as np
    import random
    import copy
    # 电阻矩阵
    def ResistorMatrix():
        L = D - A  # 图状网络G的拉普拉斯矩阵L
        L1 = pinv(L)  # L的摩尔-彭若斯逆
        Q = np.zeros([node_num, node_num])  # 电阻矩阵Q
        for i in range(len(Q)):
            for j in range(len(Q[0])):
                Q[i][j] = L1[i][i] + L1[j][j] - 2 * L1[i][j]
        return Q

    # 两个节点有共同连接节点的矩阵
    def CommonNode():
        CommonNode = np.zeros([node_num, node_num], dtype=int)
        for i in range(len(CommonNode) - 1):
            for j in range(i + 1, len(CommonNode[0])):
                count = 0
                for t in range(i + 1, len(CommonNode[0])):
                    if A[i][t] == A[j][t] == 1:
                        count = count + 1
                CommonNode[i][j] = count
                CommonNode[j][i] = count
        return CommonNode

    # 邻居节点间的距离
    def NodeDistance():
        ComNode = CommonNode()  # 公共节点矩阵
        Q = ResistorMatrix()  # 电阻矩阵
        Dis = np.zeros([node_num, node_num])  # 邻居节点的距离
        for i in range(len(Dis)):
            for j in range(len(Dis[0])):
                dd = min((1 - ComNode[i][j] / D[i][i]), (1 - ComNode[i][j] / D[j][j]))
                Dis[i][j] = Q[i][j] * dd
        return Dis

    # 计算局部亲密邻居阈值矩阵
    def Threshold():
        Dis = NodeDistance()  # 邻居节点间的距离
        thres = np.zeros([node_num, node_num])  # 各个节点的阈值
        for i in range(len(thres)):
            s = 1
            for j in range(len(thres[0])):
                if A[i][j] == 1:
                    s = s * Dis[i][j]
            thres[i][i] = s ** (1 / D[i][i])
        return thres

    # 每个节点局部最小聚类阈值
    def LocalMinimumClusteringThreshold(a):
        LMCT = []
        for i in range(len(D1)):
            LMCT.append(D1[i] * a)
        return LMCT

    def CoreNode():
        LMCT = LocalMinimumClusteringThreshold(0.7)  # 每个节点局部最小聚类阈值
        LCN = []  # 存放每个节点的局部邻居数量
        # LCN_Matrix=LocalCloseNeighbors()#节点在t时刻的局部紧密邻居矩阵
        for i in range(len(LCN_Matrix)):
            count = 0
            for j in range(len(LCN_Matrix[0])):
                if LCN_Matrix[i][j] == 2:
                    count = count + 1
            LCN.append(count)
        core_node = []
        for i in range(len(LCN)):
            if LCN[i] >= LMCT[i]:
                core_node.append(i)
        return core_node

    def CommunityDivision():
        communityEdge = []  # 保存核心节点与局部亲密邻居的边
        codenode = CoreNode()  # 获得所有的核心节点
        # 计算所有核心节点的局部亲密邻居
        LocalClosedNeighbors = {}
        for node in codenode:
            LocalClosedNeighbors[node] = []
            for i in range(len(LCN_Matrix)):
                if LCN_Matrix[node][i] == 2:
                    LocalClosedNeighbors[node].append(i)
        community = []
        cNum = -1
        unvisited = [0 for i in range(node_num)]
        for node in codenode:
            if unvisited[node] == 0:  # 核心节点未被访问，说明存在新社区
                community.append([])
                communityEdge.append([])
                cNum += 1
            else:
                continue
            coreList = [node]
            community[cNum].append(1 * node)
            communityEdge[cNum].append([node, node])
            unvisited[node] = 1
            index = 0
            while index < len(coreList):
                corenode = coreList[index]
                closedNei = LocalClosedNeighbors[corenode]
                for n in closedNei:  # 遍历所有的局部亲密邻居，将核心节点放入coreList中，将不在社区中的节点加入相应社区
                    if n in codenode:
                        if n not in coreList:
                            coreList.append(n)
                        n *= 1
                    if unvisited[abs(n)] == 0:
                        community[cNum].append(n)
                        communityEdge[cNum].append([corenode, abs(n)])
                        unvisited[abs(n)] = 1
                index += 1
        return community, communityEdge

    def nodeContribute(community, Q):
        """
        :param community: 社区
        :param Q: 电阻矩阵
        :return: 每个节点的节点贡献度
        """
        contribute = {}
        for c in community:
            sum = 0
            temp = {}
            for node1 in c:
                nodeSum = 0
                for node2 in c:
                    nodeSum += Q[node1 - 1][node2 - 1]
                sum += nodeSum
                temp[node1] = nodeSum
            for key, value in temp.items():
                contribute[key] = 1 - temp[key] / sum
        """
        num = -1
        for c in community:
            contribute.append({})
            num += 1
            sum = 0
            for node1 in c:
                nodeSum = 0
                for node2 in c:
                    nodeSum += Q[node1 - 1][node2 - 1]
                sum += nodeSum
                contribute[num][node1] = nodeSum
            totalContribute.append(sum)
        for j in range(len(contribute)):
            for key, value in contribute[j].items():
                contribute[j][key] = 1 - contribute[j][key] / totalContribute[j]
        """
        return contribute

    def dynamicContribute(CT):
        """
        :param CT: 不同时间片的节点贡献度
        :return: 不同时间片的节点动态贡献度
        """
        IM = [CT[0]]
        for i in range(1, len(CT)):
            IM.append({})
            for key, value in CT[i].items():
                if (key not in CT[i - 1].keys()) or (IM[i - 1][key] == 0):
                    IM[i][key] = CT[i][key] * 0.5
                elif IM[i - 1][key] != 0:
                    IM[i][key] = CT[i][key] - CT[i][key] * (CT[i][key] - IM[i - 1][key])
        return IM

    def communitySimilar(IM, C, delta=0.5):
        """
        :param delta: 阈值，大于该阈值则社区匹配成功
        :param IM: 每个时间片的节点的动态贡献度
        :param C: 每个时间片社区
        :return: 相似的社区，用字典表示社区关系
        """
        similar = []
        for stamp in range(len(C) - 1):  # 最后一个社区不参与运算
            similar.append({})
            for current in range(len(C[stamp])):
                for n in range(len(C[stamp + 1])):
                    intersectionNode = list(set(C[stamp][current]) & set(C[stamp + 1][n]))
                    sumCurrentCommunity = 0  # 当前时刻社区动态贡献度总和
                    sumNextCommunity = 0  # 下一时刻时刻社区动态贡献度总和
                    sumIntersection1 = 0  # t时刻社区交集的动态贡献度总和
                    sumIntersection2 = 0  # t + 1时刻社区交集的动态贡献度总和
                    for node in C[stamp][current]:
                        sumCurrentCommunity += IM[stamp][node]
                    for node in C[stamp + 1][n]:
                        sumNextCommunity += IM[stamp + 1][node]
                    for node in intersectionNode:
                        sumIntersection1 += IM[stamp][node]
                    for node in intersectionNode:
                        sumIntersection2 += IM[stamp + 1][node]
                    sim = (sumIntersection1 + sumIntersection2) / (sumCurrentCommunity + sumNextCommunity)
                    if sim > delta:
                        if current not in similar[stamp].keys():
                            similar[stamp][current] = [n]
                        elif current in similar[stamp].keys():
                            similar[stamp][current].append(n)
        return similar

    totalQ = []
    totalCommunity = []
    timeEdgeNum = []
    timeCommunityEdge = []
    graph_data = []
    path = [r'static/data/synfix/z_3/synfix_3.t01.edges',
            r'static/data/synfix/z_3/synfix_3.t02.edges',
            r'static/data/synfix/z_3/synfix_3.t03.edges',
            r'static/data/synfix/z_3/synfix_3.t04.edges',
            r'static/data/synfix/z_3/synfix_3.t05.edges',
            r'static/data/synfix/z_3/synfix_3.t06.edges',
            r'static/data/synfix/z_3/synfix_3.t07.edges',
            r'static/data/synfix/z_3/synfix_3.t08.edges',
            r'static/data/synfix/z_3/synfix_3.t09.edges',
            r'static/data/synfix/z_3/synfix_3.t10.edges',
            ]
    for j in range(len(path)):
        network_synfix, num_nodes_synfix, g = init_network(path[j])
        graph_data.append(g)
        node_num = num_nodes_synfix  # 网络中节点的数量
        # 返回所有的边，以列表的形式[[],[]...]
        edges = network_synfix
        graph_edge = json.loads(g)['links']
        edgeNum = [[0 for i in range(node_num)] for j in range(node_num)]
        for i in range(len(graph_edge)):
            source = int(graph_edge[i]['source'])
            target = int(graph_edge[i]['target'])
            edgeNum[source][target] = i
            edgeNum[target][source] = i
        timeEdgeNum.append(edgeNum)
        A = np.zeros([node_num, node_num], dtype=int)  # 网络G的邻接矩阵A
        for i in range(len(edges)):
            A[int(edges[i][0]) - 1][int(edges[i][1]) - 1] = 1  # 数据中的节点是从1开始的，而矩阵是从0开始的
            A[int(edges[i][1]) - 1][int(edges[i][0] - 1)] = 1

        # 度的节点矩阵,返回各个节点的度
        D1 = []  # 存放各个节点的度D1
        D = np.zeros([node_num, node_num], dtype=int)  # 度的节点矩阵D
        for i in range(len(A)):
            D1.append(sum(A[i]))
        for i in range(len(D1)):
            D[i][i] = D1[i]

        totalQ.append(ResistorMatrix())
        # 节点在t时刻的局部紧密邻居
        thres = Threshold()  # 局部亲密邻居阈值
        LCN_Matrix = copy.deepcopy(A)  # 邻接矩阵
        Dis = NodeDistance()  # 邻居节点间的距离
        for i in range(len(LCN_Matrix)):
            for j in range(len(LCN_Matrix[0])):
                if LCN_Matrix[i][j] == 1 and (Dis[i][j] <= thres[i][i] or Dis[i][j] <= thres[j][j]):
                    LCN_Matrix[i][j] = LCN_Matrix[j][i] = 2

        C, communityEdge = CommunityDivision()
        tempC = []
        for i in C:
            if len(i) >= 3:
                tempC.append(i)
        tempEdge = []
        n = 0
        for community in tempC:
            tempEdge.append([])
            for i in range(len(community)):
                for j in range(len(community)):
                    if i != j and A[community[i]][community[j]] != 0:
                        tempEdge[n].append([community[i], community[j]])
            n += 1
        timeCommunityEdge.append(tempEdge)
        totalCommunity.append(tempC)
    CT = []  # 不同时间片的节点贡献度
    for cNum in range(len(totalCommunity)):
        CT.append(nodeContribute(totalCommunity[cNum], totalQ[cNum]))  # 计算每个时间片中节点的贡献度
    IM = dynamicContribute(CT)  # 不同时间片的节点动态贡献度
    S = communitySimilar(IM, totalCommunity)
    return render_template('ECDR_Evolution.html', graph_data=graph_data, timeCommunity=totalCommunity,
                           S=S, timeCommunityEdge=timeCommunityEdge, timeEdgeNum=timeEdgeNum)


@app.route('/communityEvolution', methods=["GET", "POST"])
def Evolution():
    from tiles import TILES
    import time

    def check(info):
        # 检测前端传来的数据是否出错
        str = ""
        count = 0
        for c in info:
            # 将数据中所有的中文逗号转换成英文逗号，如果逗号超过两个或是出现非数字字符，返回错误
            if c.isdigit():
                str += c
            elif c == ',' or c == '，':
                str += ','
                count += 1
                if count > 1:
                    return [], True
            else:
                return [], True
        e = str.strip().split(',')
        if len(e) != 2:
            return [], True
        for i in e:
            if int(i) not in eNode:
                return [], True
        return e, False

    if request.method == 'GET':
        # 在开始的时候设置相关的属性
        global eNode
        global t
        global isContinue
        global s
        global eNum
        global networkData
        global graph_edge
        path = r"static/data/reptilia-tortoise-network-cs.edges"  # 以' '分隔
        networkData, node_num, graph_data = init_network(path, opacity=0)
        graph_data = json.loads(graph_data)
        graph_edge = graph_data['links']
        graph_data['links'] = []
        graph_data = json.dumps(graph_data)
        isContinue = 1
        s = 0
        t = TILES()
        eNode = []
        eNum = [[0 for _ in range(node_num)] for _ in range(node_num)]
        for i in range(len(graph_edge)):
            source = int(graph_edge[i]['source'])
            target = int(graph_edge[i]['target'])
            eNum[source][target] = i
            eNum[target][source] = i
        edges = json.dumps(networkData)
        return render_template('Evolution.html', graph_data=graph_data, edges=edges)
    elif request.method == 'POST':
        deleteCommunity = []  # 记录被删除的社区
        requestArgs = request.values
        step = requestArgs.get('step')
        speed = requestArgs.get('speed')
        if speed is not None and speed != "":
            s = float(speed)
        if step is not None:
            step = int(step)
        addEdge = requestArgs.get('addEdge')
        removeEdge = requestArgs.get('removeEdge')
        removeNode = requestArgs.get('removeNode')
        removeAllEdge = []
        typ = 1  # 设置处理的数据类型，为2使处理边删除的情况，为1时处理边增加的情况
        isRemoveEdge = 0  # 是否是边删除的情况
        isRemoveNode = 0
        if removeEdge is None and removeEdge != '' and addEdge is None and addEdge != '' and removeNode is None and removeNode != '':
            if step < len(networkData):
                currentEdge = [networkData[step][0] - 1, networkData[step][1] - 1]  # 获得边
            else:
                return jsonify({'error': 'lengthOver'})
        elif addEdge != '' and addEdge is not None:
            temp, error = check(addEdge)  # 检测数据是否正确
            if error:
                return jsonify({'error': error})
            currentEdge = [int(temp[0]), int(temp[1])]
            temp = [int(temp[0]) + 1, int(temp[1]) + 1]
            temp_ = [temp[1], temp[0]]
            if temp in networkData:
                # 如果出现重复的情况，则删除这些边
                networkData.remove(temp)
            if temp_ in networkData:
                # 如果出现重复的情况，则删除这些边
                networkData.remove(temp_)
            if edgeNum[currentEdge[0]][currentEdge[1]] == 0:
                link_id = len(graph_edge)
                graph_edge.append({
                    'id': str(link_id),
                    'lineStyle': {'normal': {}},
                    'name': 'null',
                    'source': str(currentEdge[0]),
                    'target': str(currentEdge[1])
                })
                edgeNum[currentEdge[0]][currentEdge[1]] = link_id
        elif removeEdge != '' and removeEdge is not None:
            temp, error = check(removeEdge)
            if error:
                return jsonify({'error': error})
            isRemoveEdge = 1
            typ = 2
            currentEdge = [int(temp[0]), int(temp[1])]
            temp = [int(temp[0]) + 1, int(temp[1]) + 1]
            temp_ = [temp[1], temp[0]]
            while temp in networkData:
                networkData.remove(temp)
            while temp_ in networkData:
                networkData.remove(temp_)
        elif removeNode != '' and removeNode is not None:
            for i in removeNode:
                if not i.isdigit():
                    return jsonify({'error': True})
            removeNode = int(removeNode)
            if removeNode not in eNode:
                return jsonify({'error': True})
            typ = 2
            for node in t.g.neighbors(removeNode):
                removeAllEdge.append([removeNode, node])
        if removeNode != '' and removeNode is not None:
            currentEdge = removeAllEdge
            isRemoveNode = 1
            for e in removeAllEdge:
                t.execute(e, t=typ)
            timeChangedNodeCommunity = t.change
            deleteCommunity = t.deleteCommunity
            t.change = {}
            t.deleteCommunity = []
        elif currentEdge[0] in eNode and currentEdge[1] in eNode:
            t.execute(currentEdge, t=typ)
            timeChangedNodeCommunity = t.change
            deleteCommunity = t.deleteCommunity
            t.change = {}
            t.deleteCommunity = []
        elif currentEdge[0] not in eNode:
            eNode.append(currentEdge[0])
            currentEdge = [currentEdge[0], currentEdge[0]]
            timeChangedNodeCommunity = {}
        elif currentEdge[1] not in eNode:
            eNode.append(currentEdge[1])
            currentEdge = [currentEdge[1], currentEdge[1]]
            timeChangedNodeCommunity = {}
        if step == len(networkData) - 1:
            isContinue = 0
        time.sleep(s)
        return jsonify({'changedCommunity': timeChangedNodeCommunity, 'delCommunity': deleteCommunity
                           , 'edgeNum': eNum, 'graph_edge': graph_edge, 'isContinue': isContinue
                           , 'currentEdge': currentEdge, 'isRemoveEdge': isRemoveEdge, 'isRemoveNode': isRemoveNode})


@app.route('/StaticMOACD')
def StaticMOACD():
    """
    基于社交网络属性的多目标优化静态社区划分
    author:张财、邓志斌
    date:2020年6月1日
    """
    import networkx as nx
    import random
    import math
    import numpy as np
    from numpy import mat
    from numpy import trace

    # 求出度矩阵和B矩阵
    def deg_and_B(G):
        node_len = len(G.nodes)
        m2 = 2 * G.number_of_edges()
        # 构造邻接矩阵A
        A = np.zeros((node_len, node_len), int)
        for node in G.nodes:
            for nh in G.neighbors(node):
                A[node - 1][nh - 1] = 1
        # 度矩阵
        Degree = np.zeros(node_len, int)
        for i in list(G.nodes):
            k = sum(A[i - 1])
            Degree[i - 1] = k
        # 构造B矩阵
        B = np.zeros((node_len, node_len))
        for i in list(G.nodes):
            ki = Degree[i - 1]
            for j in list(G.nodes):
                kj = Degree[j - 1]
                B[i - 1][j - 1] = A[i - 1][j - 1] - ki * kj / m2
        # print(B)
        return Degree, B

    def cal_Q(partition, node_len, m2, B):
        s_cor = len(partition)
        S_labels = np.zeros([node_len, s_cor], int)
        for i in range(s_cor):
            for j in partition[i]:
                S_labels[j - 1][i] = 1
        S_labels = mat(S_labels)
        Q = 1 / m2 * (trace(S_labels.T * B * S_labels))
        return Q

    # 论文中3.3.2初始化：公式(3.7)
    def select_probability(G):
        node_set = list(G.nodes)  # 节点集合
        len_node_set = len(node_set)
        # 保存每个节点的邻居的个数
        # 前提是网络节点连续排好，中间没有缺失
        nodenh_set = [0] * len_node_set  # 每个节点的邻居集合
        nodenh_lenset = [0] * len_node_set  # 每个节点的邻居集合的长度
        for node in node_set:
            nh = list(G.neighbors(node))
            nodenh_lenset[node - 1] = len(nh)
            nodenh_set[node - 1] = list(G.neighbors(node))
        # i和j有边的共同邻居相似度
        select_pro = []
        for node in node_set:
            # 字典{邻居节点:共同邻居数,邻居节点:共同邻居数,...}
            sim_sum = 0  # 每个节点所有邻居相似度之和
            sim_set = []
            neis_set = nodenh_set[node - 1]
            for nei in neis_set:
                nei_neis_set = nodenh_set[nei - 1]
                # 交集
                d = [n for n in neis_set if n in nei_neis_set]
                # 计算相似度
                s = float(len(d) + 1) / math.sqrt(nodenh_lenset[node - 1] * nodenh_lenset[nei - 1])
                sim_set.append(s)
                sim_sum += s
            nei_pro = {}
            for i in range(nodenh_lenset[node - 1]):
                pro = sim_set[i] / sim_sum
                nei_pro.update({neis_set[i]: pro})
            select_pro.append([node, nei_pro])
        select_pro = sorted(select_pro, key=(lambda x: x[0]))
        return select_pro

    # 计算论文公式(3.10)
    def silhouette(G, partition):
        GS = 0.0
        for index, community in enumerate(partition):
            result = 0.0
            community_len = len(community)
            for i in range(community_len):
                # 求ai---每个点与社区内其他点平均相似度
                degree = sum([1 for j in range(community_len) if G.has_edge(community[i], community[j])])
                ai = degree / community_len
                # 求bi --节点i与其他社区节点最大平均相似度
                other_sim = [sum([1 for j in range(len(other_community)) if
                                  G.has_edge(community[i], other_community[j])]) / len(other_community)
                             for index1, other_community in enumerate(partition) if index1 != index]
                if other_sim:
                    bi = max(other_sim)
                else:
                    bi = 0  # 有问题
                if max(ai, bi) == 0:
                    result += 0
                else:
                    result += (ai - bi) / max(ai, bi) / community_len
            GS += result / len(partition)
        return GS

    # 帕累托最优解集合
    def pareto(pop_ns, pop_partition, pop_value):
        '''
        # 非支配规则:a所有目标值不小于b,并且至少有一个目标值比b大，则a支配b
        :param pop_value:
        :return:
        '''

        non = []
        # 将解集复制给非支配解集
        for i in range(len(pop_value)):
            non.append(pop_value[i][:])
        for i in range(-len(non), -1):
            for j in range(i + 1, 0):
                # print non[i], non[j]
                if non[i][1] == non[j][1] and non[i][2] == non[j][2]:
                    # print i,"deng"
                    non.remove(non[i])
                    # print non
                    break
                elif ((non[i][1] >= non[j][1]) and (non[i][2] >= non[j][2])) and (
                        (non[i][1] > non[j][1]) or (non[i][2] > non[j][2])):
                    # a[i][0]支配a[j][0]
                    # print "ai支配aj"
                    non.remove(non[j])
                    break
                elif ((non[j][1] >= non[i][1]) and (non[j][2] >= non[i][2])) and (
                        (non[j][1] > non[i][1]) or (non[j][2] > non[i][2])):
                    # print "aj支配ai"
                    non.remove(non[i])
                    break
        rep_ns = []  # 帕累托最优解的节点邻接表示集合
        rep_partition = []  # 帕累托最优解的分区方案集合
        rep_num = len(non)  # 帕累托最优解的个数
        if len(pop_ns) == len(pop_value):  # 只针对初始化方案
            for i in range(rep_num):
                j = non[i][0]
                rep_ns.append(pop_ns[j])
                rep_partition.append(pop_partition[j])
        return rep_ns, rep_partition, non

    # 基因漂流，就是论文算法1邻居从众策略
    def turbulence(G, indi):
        G3 = nx.Graph()  # 解码后（稀疏）的图
        for node_cp in indi:
            G3.add_edge(node_cp[0], node_cp[1])
        components = [list(c) for c in list(nx.connected_components(G3))]
        components_len = len(components)
        for j in range(len(indi)):
            node = indi[j][0]
            nei = [n for n in G.neighbors(node)]
            # 记录邻居的集群标签
            nei_com = []  # 邻居集群的位置
            for x in range(len(components)):
                nei_com.append([n for n in nei if n in components[x]])
            # 选出最多邻居的集群
            max_nei_com = nei_com[int(np.argmax([len(nei_com[m]) for m in range(len(nei_com))]))]
            # 从最多数量的邻居分区中随机选一个值
            if indi[j][1] not in max_nei_com:
                indi[j] = [node, random.sample(max_nei_com, 1)[0]]
        return indi

    # 基因变异，就是论文算法2邻居多样策略
    def mutation(G, indi):
        for j in range(len(indi)):
            node = indi[j][0]
            t = [n for n in G.neighbors(node)]
            t.remove(indi[j][1])
            if t:
                indi[j] = [node, random.sample(t, 1)[0]]

        return indi

    # 初始社区划分方案，方案个数为100
    def init_community(G, B, N=100):
        node_num = G.number_of_nodes()
        m2 = 2 * G.number_of_edges()
        pop_ns = []  # 社区的节点邻接表示集合，键值对
        pop_partition = []  # 社区的分区方案集合
        pop_value = []  # 所有方案的目标结果集合
        # 计算节点之间的选择概率
        select_pro = select_probability(G)
        # 生成N中划分方案
        for i in range(N):
            G2 = nx.Graph()  # 邻接表示解码后的图
            indi = []  # 每个节点的基于轨迹的邻接表示
            # 生成划分方案
            for m in range(node_num):
                nei_pro = select_pro[m][1]  # 节点m+1的所有邻居节点的选择概率
                # 按照概率随机选择一个节点
                ran = random.random()  # 生成一个0-1的随机数
                rate_sum = 0
                select_node = 1  # 被节点m+1选中的邻居节点
                for neighbor, rate in nei_pro.items():
                    rate_sum += rate
                    if ran < rate_sum:
                        select_node = neighbor
                        break
                indi.append([m + 1, select_node])
                G2.add_edge(m + 1, neighbor)
            pop_ns.append(indi)
            components = [list(c) for c in list(nx.connected_components(G2))]  # G2的分区
            Q = cal_Q(components, node_num, m2, B)  # 计算该划分方案的模块度
            GS = silhouette(G, components)  # 计算GS值
            pop_partition.append(components)
            pop_value.append([i, Q, GS])
        return pop_ns, pop_partition, pop_value

    G = nx.Graph()  # 图数据
    for edge in network_synfix:
        G.add_edge(edge[0], edge[1])
    node_num = G.number_of_nodes()
    edge_num = G.number_of_edges()
    Degree, B = deg_and_B(G)  # 度矩阵和B矩阵，B矩阵用于模块度计算
    # 初始社区划分方案
    pop_ns, pop_partition, pop_value = init_community(G, B)
    # 求解帕累托最优解
    rep_ns, rep_partition, rep_value = pareto(pop_ns, pop_partition, pop_value)
    rep_par = json.dumps(rep_partition)  # 传给前端的帕累托最优划分方案

    # 迭代更新解决方案
    genmax = 20  # 最大迭代次数
    gen = 0  # 当前迭代次数
    best_gen = []  # 迭代过程中最好的值
    gen_equal = 0  # 多目标值相等的次数
    updated_par = []  # 记录每次迭代中被选中更新的分区方案
    node_update_rec = []  # 每次迭代对应方案的每个节点的更新记录
    while gen < genmax and gen_equal < 10:
        npop_ns = copy.deepcopy(pop_ns)  # 深拷贝
        npop_good_value = []  # 更新后较好的多目标值
        npop_good_partition = []  # 更新后较好的划分方案
        npop_good_ns = []  # 更新后较好的节点邻接表示
        nrep_value = []  # 新的帕累托最优多目标值
        nrep_ns = []  # 新的帕累托节点邻接表示
        nrep_partition = []  # 新的帕累托社区划分方案
        method = 1  # 更新策略
        update_num = 10  # 每次迭代的更新次数
        record_gen = []  # 第gen次迭代的所有节点更新记录
        up_par_gen = []  # 第gen次迭代中被选中更新的划分方案
        for i in range(update_num):
            npop_ns[i] = copy.deepcopy(random.sample(rep_ns, 1)[0])  # 随机从帕累托前沿中选择一个方案
            t_ns = copy.deepcopy(npop_ns[i])  # 暂存当前选中的方案
            # 随机选择一种算法更新
            if random.random() < 0.2:
                npop_ns[i] = mutation(G, npop_ns[i])  # 邻居多样策略
                method = 2
            else:
                npop_ns[i] = turbulence(G, npop_ns[i])  # 邻居从众策略
                method = 1

            # 记录第1次更新
            if i == 0:
                # 记录此次选中的方案
                G_de = nx.Graph()
                for edge in t_ns:
                    G_de.add_edge(edge[0], edge[1])
                t_components = [list(c) for c in list(nx.connected_components(G_de))]
                up_par_gen.append(t_components)

                # 记录每个节点更新后的社区所属
                t_record = [0] * node_num
                record = [0] * node_num
                com_id = 0  # 社区编号
                for community in t_components:
                    for node in community:
                        t_record[node - 1] = com_id
                    com_id += 1
                for edge in npop_ns[i]:
                    record[edge[0] - 1] = t_record[edge[1] - 1]
                record_gen.append([method, record])

            # 判断是否新值旧值支配情况
            G_new = nx.Graph()
            for node_cp in npop_ns[i]:
                G_new.add_edge(node_cp[0], node_cp[1])
            components = [list(c) for c in list(nx.connected_components(G_new))]
            Q = cal_Q(components, node_num, 2 * edge_num, B)
            GS = silhouette(G, components)
            # 判断是否新值旧值支配情况  新值不被旧值支配, 且不等于旧值
            if (Q == 0.0) or (
                    (Q <= pop_value[i][1] and GS <= pop_value[i][2]) and (Q < pop_value[i][1] or GS < pop_value[i][2])):
                no_dominate = False
            else:
                no_dominate = True
            # 支配，新值直接输出，否则，改为旧值
            if no_dominate:
                npop_good_value.append([i + (1 + gen) * 100, Q, GS])  # 新值加编号100处理
                npop_good_partition.append(copy.deepcopy(components))
                npop_good_ns.append(copy.deepcopy(npop_ns[i]))
            else:
                npop_ns[i] = pop_ns[i]

        # 帕累托最优解集合
        all_value = rep_value + npop_good_value
        ns, par, nrep_value = pareto(pop_ns, pop_partition, all_value)
        # rep_value 位置
        rep_gv_i = []
        for i in range(len(rep_value)):
            rep_gv_i.append(rep_value[i][0])
        npop_gv_i = []
        for i in range(len(npop_good_value)):
            npop_gv_i.append(npop_good_value[i][0])
        for i in range(len(nrep_value)):
            j = nrep_value[i][0]
            if j < 100 * (gen + 1):
                nrep_ns.append(copy.deepcopy(rep_ns[rep_gv_i.index(j)]))
                nrep_partition.append(copy.deepcopy(rep_partition[rep_gv_i.index(j)]))
            else:
                nrep_ns.append(copy.deepcopy(npop_good_ns[npop_gv_i.index(j)]))
                nrep_partition.append(copy.deepcopy(npop_good_partition[npop_gv_i.index(j)]))

        rep_ns = copy.deepcopy(nrep_ns)
        rep_value = copy.deepcopy(nrep_value)
        rep_partition = copy.deepcopy(nrep_partition)

        # 保存当前迭代的更新记录
        updated_par.append(up_par_gen)
        node_update_rec.append(record_gen)

        # 迭代终止条件
        best_gen.append(sorted(rep_value, key=lambda x: x[1], reverse=True)[0])
        if gen >= 1 and best_gen[gen][1:] == best_gen[gen - 1][1:]:
            gen_equal += 1
        gen += 1

    best = sorted(rep_value, key=lambda x: x[1], reverse=True)[0]
    best_location = rep_value.index(best)
    best_partition = rep_partition[best_location]

    updated_par = json.dumps(updated_par)
    node_update_rec = json.dumps(node_update_rec)
    best_partition = json.dumps(best_partition)

    return render_template('StaticMOACD.html', graph_data=graph_data_synfix, rep_par=rep_par,
                           updated_par=updated_par, node_update_rec=node_update_rec, best_partition=best_partition)


@app.route('/dyanmicMOACD')
def dyanmicMOACD():
    return render_template('DyanmicMOACD.html', async_mode=socketio.async_mode)


# 前后端通信
@socketio.on('connect', namespace='/dyanmicMOACD')
def test_connect():
    thread = socketio.start_background_task(target=dyanmicMOACD_thread)

# 基于lps算法的静态社区划分
@app.route('/lpa')
def lpa(): # 缪赏
    # 获取数据
    # G = nx.karate_club_graph()
    networkTemp = []
    G = init_network('static/data/Wiki.txt')
    # 初始化前端json数据：根据Echarts设计初始化值。参数参考https://echarts.apache.org/zh/option.html#series-graph

    #获取边信息
    networkTemp = G[0]
    # 初始化前端json数据：根据Echarts设计初始化值。参数参考https://echarts.apache.org/zh/option.html#series-graph

    # 获取节点数
    number_of_nodes = G[1]

    # 设置传给前端的节点数据边数据的json串
    graph_data_json = json.loads(G[2])
    #点类别设置每个都不同
    for i in range(number_of_nodes):
        graph_data_json['nodes'][i]['attributes']['modularity_class'] = i
    nodes_data_json = graph_data_json['nodes']
    links_data_json  = graph_data_json['links']
    graph_data = json.dumps(graph_data_json)

    # lpa算法,返回每次迭代更新的节点
    def get_neighbors(node):
        neighbors=[]
        for i in links_data_json:
            if (i.get('source')==str(node)):
                target=int(i.get('target'))
                neighbors.append([target,nodes_data_json[target].get('attributes').get('modularity_class')])
            if (i.get('target')==str(node)):
                source=int(i.get('source'))
                neighbors.append([source,nodes_data_json[source].get('attributes').get('modularity_class')])
        return neighbors;

    active_records = []  # 用来存放每个节点的模拟结果
    max_iter_num = 0  # 迭代次数
    iter_num=10 #迭代总次数
    neighbors_and_time=[] #存放循环的次数和邻居节点
    while max_iter_num < iter_num:
        active_records.append([])
        neighbors_and_time.append({})
        neighbors_and_time[max_iter_num]['time']=max_iter_num+1
        max_iter_num += 1
        # 分类的社区数
        com = len(set([nodes_data_json[node]['attributes']['modularity_class'] for node in range(number_of_nodes)]))
        #print('迭代次数', max_iter_num)
        for node in range(number_of_nodes):
            count = {}  # 记录邻居节点及其标签
            neighbors=get_neighbors(node)
            neighbortemp=[]

            for nbr in neighbors:  # node的邻居节点
                neighbortemp.append(nbr[0])
                label = nbr[1]
                count[label] = count.setdefault(label, 0) + 1
            # 找到出现次数最多的标签
            count_items = sorted(count.items(), key=lambda x: -x[-1])
            best_labels = [k for k, v in count_items if v == count_items[0][1]]
            # 当多个标签最大技术值相同时随机选取一个标签
            if(best_labels != []):
                label = random.sample(best_labels, 1)[0]  # 返回的是列表，所以需要[0]
                nodes_data_json[node]['attributes']['modularity_class'] = label  # 更新标签
            active_records[max_iter_num-1].append(label)
            #存放该节点邻居节点
            neighbors_and_time[max_iter_num - 1][str(node)] = neighbortemp
        #存放循环开始前的社区数
        neighbors_and_time[max_iter_num-1]['com'] = com

    #存放最后的社区数
    last_com = len(set([nodes_data_json[node]['attributes']['modularity_class'] for node in range(number_of_nodes)]))
    active_records = json.dumps(active_records)
    # neighbors_and_time = json.dumps(neighbors_and_time)

    #验证用
    print('社区数{}'.format(com))
    #储存分类结果 分类类型--数量
    classlist=[]
    sort_list = list()
    for node in range(number_of_nodes):
        classlist.append(nodes_data_json[node]['attributes']['modularity_class'])
    sort_set = set([nodes_data_json[node]['attributes']['modularity_class'] for node in range(number_of_nodes)])
    for item in sort_set:
        sort_list.append((item,classlist.count(item)))
    print(sort_list)

    return render_template('lpa.html',graph_data = graph_data,active_records = active_records,last_com=last_com,neighbors_and_time=neighbors_and_time)

# 基于GN算法的静态社区划分
@app.route('/GN')   # 郭易兴
def GN():
    import json
    import networkx as nx
    from networkx.algorithms import community

    def cal_Q(partition, G):
        a = []
        e = []
        m = len(G.edges(None, False))


        for community in partition:
            t = 0.0
            for node in community:
                t += len([x for x in G.neighbors(node)])
            a.append(t / (2 * m))

        for community in partition:
            t = 0.0
            for i in range(len(community)):
                for j in range(len(community)):
                    if (G.has_edge(community[i], community[j])):
                        t += 1.0
            e.append(t / (2 * m))

        q = 0.0
        for ei, ai in zip(e, a):
            q += (ei - ai ** 2)
        return q

    # 读取数据
    networkTemp = []
    networkFile = nx.read_gml('static/data/karate.gml', label='id')
    networkFile_copy = nx.read_gml('static/data/karate.gml', label='id')
    # 设置节点数
    number_of_nodes = 34
    networkTemp = list(networkFile.edges)

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

    remove_temp=[]
    remove_edge=[]
    all_Q=[]
    community_num=[]
    best_record=[]
    community_record=[]
    while list(networkFile.edges)!=[]:
        # 记录本次要删掉的边
        edge = max(nx.edge_betweenness_centrality(networkFile).items(), \
                   key=lambda item: item[1])[0]
        networkFile.remove_edge(edge[0],edge[1])
        components = [list(c) for c in list(nx.connected_components(networkFile))]
        community_record.append(components)
        remove_temp.append(edge)
        cur_Q = cal_Q(components, networkFile_copy)
        all_Q.append(cur_Q)

    best_index = all_Q.index(max(all_Q))
    best_record = community_record[best_index]
    best_num = len(best_record)
    best_Q = max(all_Q)
    community_num = [len(i) for i in community_record]
    remove_temp = [list(i) for i in remove_temp]
    # 删除边以什么顺序
    for i in remove_temp:
        for j in links_data_json:
            if i[0]-1==int(j['source']) and i[1]-1==int(j['target']):
                remove_edge.append(int(j['id']))

    remove_edge=json.dumps(remove_edge)
    return render_template('GN.html', graph_data=graph_data,
            remove_edge=remove_edge,all_Q=all_Q,community_num=community_num,
            best_record=best_record,community_record=community_record,
            best_num=best_num,best_Q=best_Q)

if __name__ == '__main__':
    socketio.run(app, debug=True)
