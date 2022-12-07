import networkx as nx
import os
import yaml

class RandomGraph(object):
    def __init__(self, node_num, p, k=4, m=5, graph_mode="WS"):
        self.node_num = node_num
        self.p = p
        self.k = k
        self.m = m
        self.graph_mode = graph_mode

    def make_graph(self):

        if self.graph_mode is "ER":
            graph = nx.random_graphs.erdos_renyi_graph(self.node_num, self.p)
        elif self.graph_mode is "WS":
            graph = nx.random_graphs.connected_watts_strogatz_graph(self.node_num, self.k, self.p)
        elif self.graph_mode is "BA":
            graph = nx.random_graphs.barabasi_albert_graph(self.node_num, self.m)

        return graph

    def get_graph_info(self, graph):
        in_edges = {}
        in_edges[0] = []
        nodes = [0]
        end = []
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            neighbors.sort()

            edges = []
            check = []
            for neighbor in neighbors:
                if node > neighbor:
                    edges.append(neighbor + 1)
                    check.append(neighbor)
            if not edges:
                edges.append(0)
            in_edges[node + 1] = edges
            if check == neighbors:
                end.append(node + 1)
            nodes.append(node + 1)
        in_edges[self.node_num + 1] = end
        nodes.append(self.node_num + 1)
       # mid_16 = in_edges[16]
       # mid_15 = in_edges[15]
        '''
        for item in in_edges[21][:]:
            if item==5:
               in_edges[21].remove(item)
        in_edges[21].append(5)
        for item in in_edges[23][:]:
            if item==3:
               in_edges[23].remove(item)
        in_edges[23].append(3)
        '''
        repeat_1 = 0
        repeat_2 = 0
        for i in range(self.node_num-2):

            for item in in_edges[i][:]:
                if item==1:
                   repeat_1 = repeat_1+1
                if item==2:
                    repeat_2 = repeat_2+1
        if repeat_1 >=2 :
            for item in in_edges[self.node_num-1][:]:
                if item==1:
                   in_edges[self.node_num-1].remove(item)
            for item in in_edges[self.node_num][:]:
                if item==1:
                    in_edges[self.node_num].remove(item)
        if repeat_2 >=2 :
            for item in in_edges[self.node_num][:]:
                if item==2:
                    in_edges[self.node_num].remove(item)
        '''
        for i in range(1,self.node_num-1):
            if len(in_edges[i])==1 and in_edges[i][0] ==0:
                for j in range(i+1, self.node_num-1):
                    if len(in_edges[j])>1:
                        for item in in_edges[j][:]:
                            if item ==i:
                                in_edges[j].remove(item)
        '''
        #in_edges[16].remove(2)
        #in_edges[15].remove(0)
        #in_edges[15].remove(1)

        return nodes, in_edges

    def save_random_graph(self, graph, path):
        if not os.path.isdir("saved_graph"):
            os.mkdir("saved_graph")
        nx.write_yaml(graph, "./saved_graph/" + path)

    def load_random_graph(self, path):
        return nx.read_yaml("./saved_graph/" + path)
