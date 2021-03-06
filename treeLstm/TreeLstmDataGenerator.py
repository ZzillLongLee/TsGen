import torch
from treelstm import calculate_evaluation_orders
import numpy

def buildTree(astNode2Vec, taskElement, astNodeDict):
    taskTree = taskElement['tree']
    nodeType = taskTree['nodeType']
    children = taskTree['children']
    root = dict()
    vector = astNode2Vec.getVectorValue(nodeType)
    label = astNodeDict.index(nodeType)
    root['features'] = vector
    root['labels'] = vector
    root['children'] = list()
    if (len(children) != 0):
        buildSubTree(root, children, astNode2Vec, astNodeDict)
    return root

def buildSubTree(root, children, astNode2Vec, astNodeDict):
    for child in children:
        nodeType = child['nodeType']
        children = child['children']
        subRoot = dict()
        vector = astNode2Vec.getVectorValue(nodeType)
        label = astNodeDict.index(nodeType)
        subRoot['features'] = vector
        subRoot['labels'] = vector
        subRoot['children'] = list()
        root['children'].append(subRoot)
        if (len(children) != 0):
            buildSubTree(subRoot, children, astNode2Vec, astNodeDict)


def _label_node_index(node, n=0):
    node['index'] = n
    for child in node['children']:
        n += 1
        _label_node_index(child, n)


def _gather_node_attributes(node, key):
    features = [node[key]]
    for child in node['children']:
        features.extend(_gather_node_attributes(child, key))
    return features


def _gather_adjacency_list(node):
    adjacency_list = []
    for child in node['children']:
        adjacency_list.append([node['index'], child['index']])
        adjacency_list.extend(_gather_adjacency_list(child))

    return adjacency_list


def convert_tree_to_tensors(tree, device=torch.device('cpu')):
    # Label each node with its walk order to match nodes to feature tensor indexes
    # This modifies the original tree as a side effect
    _label_node_index(tree)

    features = _gather_node_attributes(tree, 'features')
    labels = _gather_node_attributes(tree, 'labels')
    adjacency_list = _gather_adjacency_list(tree)

    node_order, edge_order = calculate_evaluation_orders(adjacency_list, len(features))

    return {
        'features': torch.tensor(numpy.array(features), device=device, dtype=torch.float32),
        'labels': torch.tensor(numpy.array(labels), device=device, dtype=torch.float32),
        'node_order': torch.tensor(node_order, device=device, dtype=torch.int64),
        'adjacency_list': torch.tensor(numpy.array(adjacency_list), device=device, dtype=torch.int64),
        'edge_order': torch.tensor(edge_order, device=device, dtype=torch.int64),
    }
