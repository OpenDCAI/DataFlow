import copy
import numpy as np


class treeNode(object):
    def __init__(self, pcd, parent=None, depth=0):
        self.pcd = pcd  # str
        self.y = ''  # str
        self.parent = parent  # treeNode
        self.numVisits = 0  # int
        self.V = 0  # float
        self.children = {}  # dict{str:treeNode}
        self.depth = depth  # int
        self.isFullyExpanded = False  # expanded
        self.visit_sequence = 0
        self.final_ans_flag = 0
        self.reflection = ''
        self.isTerminal = False  # value acceptable
        self.on_final_route = False
        self.min_steps_to_correct = 1024
        self.summary = ''
        self.he = 0  # hard estimation
        self.se = 0  # soft estimation

    def __str__(self):
        s = ["numVisits: %d" % self.numVisits, f'V:{self.V}', "possibleActions: %s" % (self.children.keys())]
        return "%s: {%s}" % (self.__class__.__name__, ', '.join(s))

    def append_children(self, new_pcd: str):
        node = treeNode(new_pcd, self, self.depth + 1)
        node.update_y_from_parent()
        self.children.update({new_pcd: node})
        return self

    def update_y_from_parent(self):
        if self.parent is None:
            self.y = self.pcd
        else:
            self.y = self.parent.y + self.pcd

    def update_value(self, value):
        self.V = value

    def update_reflection(self, reflection):
        self.reflection = reflection

    def getBestV(self):  # Gets the subtree maximum value node
        if not self.isFullyExpanded:
            return self, self.V
        max_V = self.V
        max_node = self
        for child in self.children.values():
            subNode, subValue = child.getBestV()
            if subValue >= max_V:
                max_V = subValue
                max_node = subNode
        return max_node, max_V

    def trace_route(self):  # trace route from terminal node to root
        cur_node = self
        while cur_node is not None:
            cur_node.on_final_route = True
            cur_node = cur_node.parent

    def get_new_value_samples(self):  # get value samples from search tree (start from terminal node)
        if self.depth == 0:
            return []
        step_value = 1.0 / self.depth
        new_samples = []
        cur_node = self.parent
        while cur_node is not None:
            for child in cur_node.children.values():
                if child.on_final_route:
                    child_value = step_value * child.depth
                    new_item = {'steps': child.y, 'value': child_value}
                    new_samples.append(new_item)
                else:
                    child_value = max(step_value * (cur_node.depth - 1), 0)
                    new_item = {'steps': child.y, 'value': child_value}
                    new_samples.append(new_item)
            cur_node = cur_node.parent
        return new_samples

    def get_all_end_root_nodes_vm(self, end_gate):
        end_nodes = []
        if self.isFullyExpanded:
            for child in self.children.values():
                end_nodes.extend(child.get_all_end_root_nodes_vm(end_gate))
            return end_nodes
        else:
            if self.V >= end_gate or self.reflection == '<end>':
                return [self]
            else:
                return []

    def get_all_end_root_nodes_prm(self):
        end_nodes = []
        if self.isFullyExpanded:
            for child in self.children.values():
                end_nodes.extend(child.get_all_end_root_nodes_prm())
            return end_nodes
        else:
            if self.reflection == '<end>':
                return [self]
            else:
                return []

    def get_all_value_samples_vm(self):
        full_value_samples = []
        if self.depth == 0:
            self.V = 0
        else:
            if self.he == 0:
                r = -1
            else:
                r = 1
            self.V = max(0, (1 - self.parent.V) * r / self.min_steps_to_correct + self.parent.V)
            full_value_samples.append({'steps': self.y, 'value': self.V})
        if self.isFullyExpanded:
            for child in self.children.values():
                if child.min_steps_to_correct < 1024:
                    sub_samples = child.get_all_value_samples_vm()
                    full_value_samples.extend(sub_samples)
        return full_value_samples

    def get_full_value_samples_vm(self, end_leaf_nodes):
        for leaf in end_leaf_nodes:
            if leaf.min_steps_to_correct > 1:
                continue
            else:
                leaf.he = 1
                cur_node = leaf.parent
                while cur_node is not None:
                    cur_node.min_steps_to_correct = min(
                        [n.min_steps_to_correct for n in cur_node.children.values()]) + 1
                    cur_node.he = 1
                    cur_node = cur_node.parent
        for leaf in end_leaf_nodes:
            if leaf.min_steps_to_correct > 1:
                cur_node = leaf.parent
                while cur_node is not None and cur_node.min_steps_to_correct == 1024:
                    cur_node = cur_node.parent
                if cur_node is None:
                    continue
                else:
                    m = cur_node.min_steps_to_correct
                    cur_node = leaf
                    while cur_node.min_steps_to_correct == 1024:
                        cur_node.min_steps_to_correct = m
                        cur_node = cur_node.parent
            else:
                continue
        value_samples = self.get_all_value_samples_vm()
        return value_samples

    def get_all_value_samples_prm(self):
        full_value_samples = []
        if self.on_final_route:
            full_value_samples.append({'steps': self.y, 'value': self.he})
            if self.isFullyExpanded:
                for child in self.children.values():
                    if child.on_final_route:
                        sub_samples = child.get_all_value_samples_prm()
                        full_value_samples.extend(sub_samples)
            return full_value_samples
        else:
            return []

    def get_full_value_samples_prm(self, end_leaf_nodes):
        for leaf in end_leaf_nodes:
            cur_node = leaf.parent
            while cur_node is not None:
                cur_node.on_final_route = True
                cur_node = cur_node.parent
        for leaf in end_leaf_nodes:
            cur_node = leaf.parent
            while cur_node is not None:
                cur_node.he = max([n.he for n in cur_node.children.values() if n.on_final_route])
                cur_node = cur_node.parent
        value_samples = self.get_all_value_samples_prm()
        return value_samples
