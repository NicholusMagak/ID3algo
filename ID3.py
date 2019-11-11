import math
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import operator
import joblib


class Node:
    # constructor
    def __init__(self, value):
        self.value = value
        self.childs = []
        self.arcs = []


class ID3:
    # constructor
    def __init__(self, data, target):
        self.data = data
        self.target_attribute = target
        self.decision_tree = nx.DiGraph()

    # calculate entropy using the probability
    def entropy(self, probabilities: any) -> float:
        entropy_val = 0
        if type(probabilities) is list:
            entropy_val = sum(
                map(lambda prob: -1 * (prob * math.log2(prob)), probabilities))
        elif type(probabilities) is int or type(probabilities) is float:
            entropy_val = -1 * (probabilities * math.log2(probabilities))
        return entropy_val

    def __getTargetEntropy(self, data):
        target_probs = self.probabilities(self.target_attribute, data)[
            "probs"].values()
        return self.entropy(list(target_probs))

    def probabilities(self, attribute, data=False):
        attribute_class_count = dict(data[attribute].value_counts())
        probabilities = dict(map(lambda class_name, count: (
            class_name, count / data[attribute].count()), attribute_class_count.keys(), attribute_class_count.values()))
        return {
            'probs': probabilities,
            'class_counts': attribute_class_count
        }

    def isSingleLabeled(self, attribute: str, data: pd.DataFrame) -> bool:
        return data[attribute].nunique() == 1

    def getDominantLabel(self, attribute: str, data: pd.DataFrame) -> str:
        labels = dict(data[attribute].value_counts())
        return max(labels.items(), key=operator.itemgetter(1))[0]

    def getMaxInfoGain(self, inputs: list, data: pd.DataFrame) -> str:
        target_entropy = self.__getTargetEntropy(data)

        # print("Target entropy => {}".format(target_entropy))

        # ?Local function to calculate attribute entropy
        def attribute_info_gain(attribute) -> float:
            # get attribute probabilities
            attribute_weights_dict = self.probabilities(attribute, data)["probs"]
            attribute_entropy = 0

            for value in attribute_weights_dict.keys():
                # ? subset for each value of attribute
                subset = data[data[attribute] == value]
                subset_target_probs = self.probabilities(self.target_attribute, subset)['probs'].values()
                # print("Subset for {} = {}: \n {} \n".format(attribute,value,subset))
                # print("Probabilities of {} : {}".format(self.target_attribute, subset_target_probs))
                # ? add weighted entropies
                attribute_entropy += attribute_weights_dict[value] * self.entropy(list(subset_target_probs))

            return target_entropy - attribute_entropy

        # ?end

        info_gains_dict = dict(map(lambda attribute: (attribute, attribute_info_gain(attribute)), inputs))
        max_info_gain = max(info_gains_dict.items(), key=operator.itemgetter(1))[0]
        # print("Info gains: {} \n Max info gain: {}".format(info_gains_dict, max_info_gain))
        return max_info_gain

    def id3Run(self, inputs: list, output: str, data: pd.DataFrame) -> Node:
        """
        returns a decision tree
        inputs: list of attributes
        output: target attribute, the decision
        """
        if data.empty:
            return Node("Failure")

        if self.isSingleLabeled(output, data):
            return Node(data[output].unique()[0])

        if len(inputs) == 0:
            return Node(self.getDominantLabel(output, data))

        best_attribute = self.getMaxInfoGain(inputs, data)
        root = Node(best_attribute)
        best_attribute_values = data[best_attribute].unique()
        inputs.remove(best_attribute)
        for value in best_attribute_values:
            value_subset = data[data[best_attribute] == value]
            # print("Subset: \n ",value_subset)
            root.arcs.append(value)
            root.childs.append(self.id3Run(inputs, self.target_attribute, value_subset))

        return root

    def print_word_tree(self, root: Node):
        if len(root.childs) == 0:
            print("{} = {}".format(self.target_attribute, root.value))
        else:
            for arc in root.arcs:
                print("if {} = {}:".format(root.value, arc))
                self.print_word_tree(root.childs[root.arcs.index(arc)])

    def print_graph_tree(self, node: Node):
        # ?Add root node
        self.decision_tree.add_node(node.value)

        # ?Local function to add child nodes and edges
        leaf_node_count = 0;

        def add_children(parent: Node):
            for arc in parent.arcs:
                child_node: Node = parent.childs[parent.arcs.index(arc)]

                if len(child_node.childs) == 0:
                    nonlocal leaf_node_count
                    leaf_node_count += 1
                    custom_value = "({}) {}={}".format(leaf_node_count, self.target_attribute, child_node.value)
                    self.decision_tree.add_node(custom_value)
                    self.decision_tree.add_edge(parent.value, custom_value, label=arc)
                else:
                    self.decision_tree.add_node(child_node.value)
                    self.decision_tree.add_edge(parent.value, child_node.value, label=arc)

                if len(child_node.childs) > 0:
                    add_children(child_node)

        # ?end
        add_children(node)

        arc_weights = nx.get_edge_attributes(self.decision_tree, 'label')
        pos = nx.spring_layout(self.decision_tree)
        nx.draw_networkx(self.decision_tree, pos=pos)
        nx.draw_networkx_edge_labels(self.decision_tree, pos=pos, edge_labels=arc_weights)
        plt.savefig("id3_fast_tree")
        plt.show()

    def train(self):
        # ?train and save model
        inputs = list(self.data.loc[:, self.data.columns != self.target_attribute].columns)
        root = self.id3Run(inputs, self.target_attribute, self.data)
        self.print_word_tree(root)
        self.print_graph_tree(root)

        # ?Save model
        joblib.dump(root, "id3_fast_model.pkl")

    def predict(self, data: pd.DataFrame) -> list:
        # fetch stored model
        model = joblib.load("id3_fast_model.pkl")

        # ?output prediction
        def output_prediction(node: Node, record) -> str:
            if len(node.childs) == 0:
                return node.value

            child_index = node.arcs.index(record[node.value])
            return output_prediction(node.childs[child_index], record)

        # ?end

        return list(map(lambda record: output_prediction(model, record), data.to_dict("records")))

    def confusion_matrix(self, y_true: list, y_pred: list, labels: list):
        return confusion_matrix(y_true, y_pred, labels)


if __name__ == "__main__":
    data_headers = ['engine', 'turbo', 'weight', 'fueleco', 'fast']
    data = pd.read_csv("id3_data.csv", names=data_headers, header=None)
    id3 = ID3(data, "fast")
    # id3.train()
    test_data = pd.read_csv("id3_test_data.csv")
    y_true = test_data.loc[:, "fast"].values
    print("Gold Standard:\n", y_true)
    y_predictions = id3.predict(test_data)
    print("Prediction:\n", y_predictions)
    labels = id3.data[id3.target_attribute].unique()
    confusion_matrix = id3.confusion_matrix(y_true, y_predictions, labels)
    print("Confusion matrix:\n", confusion_matrix, "\n")
    print("The tree and graph print outs: \n", id3.train())