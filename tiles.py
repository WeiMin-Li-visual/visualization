"""
    Created on 11/feb/2015
    @author: Giulio Rossetti
"""
import networkx as nx
import gzip
import datetime
import time
from future.utils import iteritems
import os

import sys

if sys.version_info > (2, 7):
    from queue import PriorityQueue
else:
    from Queue import PriorityQueue


class TILES(object):
    """
        TILES
        Algorithm for evolutionary community discovery
    """

    def __init__(self, filename=None, g=None, ttl=float('inf'), obs=7, path="", start=None, end=None):
        """
            Constructor
            :param g: networkx graph
            :param ttl: edge time to live (days)
            :param obs: observation window (days)
            :param path: Path where generate the results and find the edge file
            :param start: starting date
            :param end: ending date
        """
        self.path = path
        self.ttl = ttl
        self.cid = 0  # 社区的编号
        self.actual_slice = 0
        if g is None:
            self.g = nx.Graph()
        else:
            self.g = g
        self.splits = None
        self.base = os.getcwd()
        self.status = open("%s/%s/extraction_status.txt" % (self.base, path), "w")
        self.removed = 0
        self.added = 0
        self.filename = filename
        self.start = start
        self.end = end
        self.obs = obs
        self.communities = {}
        self.nodeCommunity = {}
        self.change = {}
        self.deleteCommunity = []
        self.newCommunity = 1

    def execute(self, edge, t=1):
        """
        :param edge: 发生变化的边
        :param t: 变化类型，1：边增加，2：边删除
        :return: 无返回值
        """
        u = edge[0]
        v = edge[1]
        count = 0
        if not self.g.has_node(u):
            self.g.add_node(u)
            self.g.nodes[u]['c_coms'] = {}
            count += 1

        if not self.g.has_node(v):
            self.g.add_node(v)
            self.g.nodes[v]['c_coms'] = {}
            count += 1

        if not self.g.has_edge(u, v):
            self.g.add_edge(u, v)
        if t == 2:
            self.g.remove_edge(u, v)
        if count == 2:
            # count = 2时说明两个节点都是新加入的节点，此时对网络没有影响
            return
        u_n = list(self.g.neighbors(u))
        v_n = list(self.g.neighbors(v))
        common_nei = list(set(u_n) & set(v_n))
        if t == 2:  # 边删除
            if u not in self.nodeCommunity.keys() or v not in self.nodeCommunity.keys():
                # u和v都不属于任何社区，此时边删除没有任何影响
                return
            if self.nodeCommunity[u] != self.nodeCommunity[v]:
                # u和v属于不同社区
                return
            if len(list(self.g.nodes[u]['c_coms'].keys())) == 0 and len(list(self.g.nodes[v]['c_coms'].keys())) == 0:
                return
            cid = self.nodeCommunity[u]  # 此时u和v属于相同社区，获得社区号
            if len(list(self.g.nodes[u]['c_coms'].keys())) == 1 and len(list(self.g.nodes[v]['c_coms'].keys())) == 0:
                # u是核心节点而v是边缘节点，遍历v的邻居，如果没有核心节点，此时v不属于任何社区
                nCid = -1
                for nei in v_n:
                    # 遍历v的邻居，如果有核心节点则将v加入到该核心节点所属的社区中
                    if len(list(self.g.nodes[nei]['c_coms'].keys())) == 1:
                        nCid = self.nodeCommunity[nei]
                del self.nodeCommunity[v]
                del self.communities[cid][v]
                if nCid != -1:
                    self.nodeCommunity[v] = nCid
                    self.communities[nCid][v] = None
                self.change[v] = nCid
            elif len(list(self.g.nodes[u]['c_coms'].keys())) == 0 and len(list(self.g.nodes[v]['c_coms'].keys())) == 1:
                # u是核心节点而v是边缘节点,同上
                nCid = -1
                for nei in u_n:
                    if len(list(self.g.nodes[nei]['c_coms'].keys())) == 1:
                        nCid = self.nodeCommunity[nei]
                del self.nodeCommunity[u]
                del self.communities[cid][u]
                if nCid != -1:
                    self.nodeCommunity[u] = nCid
                    self.communities[nCid][u] = None
                self.change[u] = nCid
            else:
                # u,v均属于核心节点
                for nei in common_nei:
                    # 遍历u,v的共同邻居，如果同一个邻居和u,v形成三角形，此时社区没有分裂
                    nei_n = list(self.g.neighbors(nei))
                    if len(list(set(nei_n) & set(u_n))) > 0 and len(list(set(nei_n) & set(v_n))) > 0:
                        return

                # 社区分裂事件发生
                for i in self.communities[cid]:
                    # 删除分裂社区中所有的节点
                    del self.nodeCommunity[i]
                    self.change[i] = -1
                    self.g.nodes[i]['c_coms'] = {}
                deleteCid = cid  # 记录社区号，后面最新生成的社区的社区号
                del self.communities[cid]  # 删除社区本身
                self.deleteCommunity.append(cid)
                q = [u, v]
                q.extend(common_nei)  # 记录下核心节点，从该节点进行传播会形成新的社区
                for node in q:
                    if node in self.nodeCommunity.keys():
                        q.remove(node)
                for node in q:
                    if node not in self.nodeCommunity.keys():
                        self.fun(node, deleteCid)
            return
        if len(common_nei) == 0:
            if len(list(self.g.nodes[u]['c_coms'].keys())) == 1 and len(list(self.g.nodes[v]['c_coms'].keys())) == 1:
                return
            if len(list(self.g.nodes[u]['c_coms'].keys())) == 1 and len(list(self.g.nodes[v]['c_coms'].keys())) == 0:
                cid = list(self.g.nodes[u]['c_coms'].keys())[0]
                self.nodeCommunity[v] = cid
                self.change[v] = cid
                self.communities[cid][v] = None
                return
            if len(list(self.g.nodes[u]['c_coms'].keys())) == 0 and len(list(self.g.nodes[v]['c_coms'].keys())) == 1:
                cid = list(self.g.nodes[v]['c_coms'].keys())[0]
                self.nodeCommunity[u] = cid
                self.change[u] = cid
                self.communities[cid][u] = None
                return
        else:
            qr = [u, v]
            cid = -1
            cIndxCollection = []
            # 获得社区号
            if len(list(self.g.nodes[u]['c_coms'].keys())) == 1:
                cid = list(self.g.nodes[u]['c_coms'].keys())[0]
                cIndxCollection.append(cid)
            if len(list(self.g.nodes[v]['c_coms'].keys())) == 1:
                cid = list(self.g.nodes[v]['c_coms'].keys())[0]
                cIndxCollection.append(cid)
            for nei in common_nei:
                if len(list(self.g.nodes[nei]['c_coms'].keys())) != 0:
                    cid = list(self.g.nodes[nei]['c_coms'].keys())[0]
                    cIndxCollection.append(cid)

            if cid == -1:
                # 如果u和v及其所有的共同邻居都不属于任何社区那么新社区生成
                newCid = self.newCommunity
                self.newCommunity += 1
                cid = newCid
                self.communities[newCid] = {}
                self.g.nodes[u]['c_coms'][newCid] = None
                self.communities[newCid][u] = None
                self.nodeCommunity[u] = newCid
                self.change[u] = newCid
                self.g.nodes[v]['c_coms'][newCid] = None
                self.communities[newCid][v] = None
                self.nodeCommunity[v] = newCid
                self.change[v] = newCid
                for nei in common_nei:
                    if len(list(self.g.nodes[nei]['c_coms'].keys())) == 0:
                        self.g.nodes[nei]['c_coms'][newCid] = None
                        self.communities[newCid][nei] = None
                        self.nodeCommunity[nei] = newCid
                        self.change[nei] = newCid
            else:
                self.g.nodes[u]['c_coms'][cid] = None
                self.g.nodes[v]['c_coms'][cid] = None
                for nei in common_nei:
                    self.g.nodes[nei]['c_coms'][cid] = None
            qr.extend(common_nei)
            i = 0
            while i != len(qr):
                n = qr[i]
                if self.isCore(n):
                    for nei in self.g.neighbors(n):
                        if nei not in qr:
                            nei_n = list(self.g.neighbors(nei))
                            n_n = list(self.g.neighbors(n))
                            c_n = list(set(nei_n) & set(n_n))
                            if len(c_n) != 0:
                                qr.append(nei)
                            if nei not in self.nodeCommunity.keys() or self.nodeCommunity[nei] in cIndxCollection:
                                qr.append(nei)
                if n not in self.communities[cid]:
                    if n in self.nodeCommunity.keys():
                        lastCid = self.nodeCommunity[n]
                        del self.communities[lastCid][n]
                        self.g.nodes[n]['c_coms'] = {}
                        if len(self.communities[lastCid]) == 0:
                            del self.communities[lastCid]
                            self.deleteCommunity.append(lastCid)
                        else:
                            count = 0
                            for j in self.communities[lastCid].keys():
                                if len(list(self.g.nodes[j]['c_coms'])) != 0:
                                    count += 1
                            if count == 0:
                                for j in self.communities[lastCid].keys():
                                    del self.nodeCommunity[j]
                                    self.change[j] = -1
                                del self.communities[lastCid]
                                self.deleteCommunity.append(lastCid)
                    self.nodeCommunity[n] = cid
                    self.change[n] = cid
                    self.communities[cid][n] = None
                    if self.isDeleteCore(n):
                        self.g.nodes[n]['c_coms'][cid] = None  # 给予核心节点身份
                i += 1

    def isCore(self, node):
        if len(list(self.g.nodes[node]['c_coms'])) != 0:
            return True
        else:
            return False

    def isDeleteCore(self, node):
        nei = list(self.g.neighbors(node))
        for n in nei:
            if len(list(set(self.g.neighbors(n)) & set(nei))) != 0:
                return True
        return False

    def fun(self, node, dCid):
        nei = list(self.g.neighbors(node))
        sign = True
        for n in nei:  # 遍历node 的邻居节点
            nei_n = list(self.g.neighbors(n))
            common_nei = list(set(nei) & set(nei_n))
            if len(common_nei) > 0:  # 新社区生成
                u = node
                v = n
                if dCid not in self.communities.keys():
                    newCid = dCid
                else:
                    newCid = self.newCommunity
                    self.newCommunity += 1
                sign = False
                break
        if sign:
            sign_2 = True
            for n in nei:
                nei_n = list(self.g.neighbors(n))
                if len(list(self.g.nodes[n]['c_coms'].keys())) == 0:
                    if n in self.nodeCommunity.keys():
                        oldCid = self.nodeCommunity[n]
                        del self.nodeCommunity[n]
                        self.change[n] = -1
                        del self.communities[oldCid][n]
                    for j in nei_n:
                        if len(list(self.g.nodes[j]['c_coms'].keys())) != 0 and j in self.nodeCommunity.keys():
                            c = self.nodeCommunity[j]
                            self.nodeCommunity[n] = c
                            self.change[n] = c
                            self.communities[c][n] = None
                            break
                elif sign_2:
                    cid = self.nodeCommunity[n]
                    self.nodeCommunity[node] = cid
                    self.change[node] = cid
                    self.communities[cid][node] = None
                    sign_2 = False
        else:
            # 新社区诞生，进行传播
            qr = [u, v]
            cid = newCid
            self.g.nodes[u]['c_coms'][cid] = None
            self.g.nodes[v]['c_coms'][cid] = None
            self.nodeCommunity[u] = cid
            self.change[u] = cid
            self.nodeCommunity[v] = cid
            self.change[v] = cid
            self.communities[cid] = {}
            self.communities[cid][u] = None
            self.communities[cid][v] = None
            for nei in common_nei:
                self.g.nodes[nei]['c_coms'][cid] = None
                self.nodeCommunity[nei] = cid
                self.change[nei] = cid
                self.communities[cid][nei] = None
            qr.extend(common_nei)
            i = 0
            while i != len(qr):
                n = qr[i]
                if n not in self.communities[cid]:
                    if n in self.nodeCommunity.keys():
                        # n如果已有社区，则将它从原社区移除并加入新社区
                        lastCid = self.nodeCommunity[n]
                        del self.communities[lastCid][n]
                        self.g.nodes[n]['c_coms'] = {}
                        if len(self.communities[lastCid]) == 0:
                            # 如果移除的节点是社区中的最后一个节点
                            del self.communities[lastCid]
                            self.deleteCommunity.append(lastCid)
                        else:
                            count = 0
                            # 判断社区中是否还有核心节点
                            for j in self.communities[lastCid].keys():
                                if len(list(self.g.nodes[j]['c_coms'])) != 0:
                                    count += 1
                            if count == 0:
                                # 如果没有核心节点的话，则删除该社区
                                for j in self.communities[lastCid].keys():
                                    del self.nodeCommunity[j]
                                    self.change[j] = -1
                                del self.communities[lastCid]
                                self.deleteCommunity.append(lastCid)
                    self.nodeCommunity[n] = cid
                    self.change[n] = cid
                    self.communities[cid][n] = None
                if self.isDeleteCore(n):  # 判断是否为核心节点, 如果是的话，其邻居可能会加入该社区
                    self.g.nodes[n]['c_coms'][cid] = None
                    for nei in self.g.neighbors(n):
                        if nei not in qr:
                            nei_n = list(self.g.neighbors(nei))
                            n_n = list(self.g.neighbors(n))
                            c_n = list(set(nei_n) & set(n_n))  # 获得n与nei的共同邻居
                            if len(c_n) == 0:
                                if not self.isDeleteCore(nei):
                                    qr.append(nei)
                            else:
                                qr.append(nei)
                i += 1
