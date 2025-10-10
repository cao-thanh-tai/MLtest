from graphviz import Digraph


class VeCay:
    def __init__(self, tree, col):
        self.tree=tree
        self.feature = tree.feature
        self.threshold = tree.threshold
        self.left = tree.left
        self.right = tree.right
        self.value = tree.value
        self.ti_le = tree.ti_le
        self.dot = Digraph(comment="my tree")
        self.col=col
        pass
    
    def build_tree(self):
        
        pass
    
    def draw_tree(self, fea_i = 0, bra_j = -1, node = 0):
        if (self.left[node]==None): return 
        self.dot.node(str(node),f"{self.col[self.feature[fea_i]]}\n{self.ti_le[node]}")
        node_left=self.left[node]
        node_right=self.right[node]
        self.dot.node(str(node_left),f"{self.value[node_left]}")
        self.dot.node(str(node_right),f"{self.value[node_right]}")
        
        lb = self.threshold[fea_i]
        print(lb, bra_j)
        if bra_j != -1:
            lb = lb[bra_j]
        else:
            lb = lb[0]
        if isinstance(lb, float):
            lb = f"> {round(lb, 2)}"
        self.dot.edge(str(node),str(self.left[node]),lb)
        new_bra_j = -1
        if fea_i+1!=len(self.threshold):
            new_bra_j = -1 if len(self.threshold[fea_i+1]) == 1 else 0
        
        self.draw_tree(fea_i+1,new_bra_j,self.left[node])
        
        self.dot.edge(str(node),str(self.right[node]),"con lai")
        
        if bra_j == -1 or bra_j + 1 == len(self.threshold[fea_i]):
            bra_j=new_bra_j
            fea_i+=1
        else:
            bra_j += 1
        self.draw_tree(fea_i, bra_j, self.right[node])
        pass
    
    def show_tree(self):
        bra_j = -1 if len(self.threshold[0]) == 1 else 0
        self.dot.node('0',f"0\n{self.col[self.feature[0]]}")
        self.draw_tree(0,bra_j,0)
        self.dot.render('tree.gv', view=True)
        pass
    



