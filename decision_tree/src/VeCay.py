from graphviz import Digraph

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature     # Chỉ số thuộc tính để chia
        self.threshold = threshold # Ngưỡng chia (hoặc giá trị phân loại)
        self.left = left           # Node con trái
        self.right = right         # Node con phải
        self.value = value         # Giá trị dự đoán (nếu là lá)
    
    def is_leaf(self):
        """Kiểm tra node có phải là lá không"""
        return self.value is not None


class VeCay:
    def __init__(self, node, col = None):
        self.root = node
        self.dot = Digraph()
        self.node_count = 0
        self.col = col
    def is_leaf(self, node):
        return node.value is not None
    
    def _add_node(self, node, parent_id=None, edge_label=""):
        """Hàm đệ quy để thêm node vào đồ thị"""
        if node is None:
            return
        
        # Tạo ID duy nhất cho node
        current_id = str(self.node_count)
        self.node_count += 1
        
        # Tạo label cho node
        if self.is_leaf(node):
            # Node lá: hiển thị giá trị dự đoán
            label = f"Giá trị: {node.value}"
            node_style = {'shape': 'ellipse', 'style': 'filled', 'fillcolor': 'lightgreen'}
        else:
            # Node quyết định: hiển thị điều kiện
            try:
                # Thử định dạng số
                threshold_str = f"{float(node.threshold):.2f}"
            except (ValueError, TypeError):
                # Nếu không phải số, dùng trực tiếp
                threshold_str = str(node.threshold)
            str_fea = self.col[node.feature] if self.col is not None else  f"X[{node.feature}]"
            label = f"{str_fea} <= {threshold_str}"
            node_style = {'shape': 'box', 'style': 'filled', 'fillcolor': 'lightblue'}
        
        # Thêm node vào đồ thị
        self.dot.node(current_id, label, **node_style)
        
        # Kết nối với node cha (nếu có)
        if parent_id is not None:
            self.dot.edge(parent_id, current_id, label=edge_label)
        
        # Đệ quy thêm node con
        if not self.is_leaf(node):
            self._add_node(node.left, current_id, "Có")
            self._add_node(node.right, current_id, "Không")
    
    def ve_cay(self, filename="decision_tree", view=True):
        """Vẽ và hiển thị cây quyết định"""
        # Reset đồ thị
        self.dot = Digraph()
        self.node_count = 0
        
        # Cấu hình đồ thị
        self.dot.attr(rankdir='TB', size='12,8')  # Top to Bottom
        self.dot.attr('node', fontname='Arial', fontsize='10')
        self.dot.attr('edge', fontname='Arial', fontsize='9')
        
        # Bắt đầu thêm nodes từ root
        self._add_node(self.root)
        
        # Render và hiển thị
        self.dot.render(filename, view=view, cleanup=True)
        return self.dot
    
    def hien_thi_source(self):
        """Hiển thị mã nguồn DOT"""
        self.ve_cay(view=False)
        return self.dot.source
    
    def luu_file(self, filename, format='png'):
        """Lưu cây dưới dạng file ảnh"""
        self.ve_cay(view=False)
        return self.dot.render(filename, format=format, cleanup=True)


# Ví dụ sử dụng
if __name__ == "__main__":
    # Tạo một cây ví dụ
    """
    Cây ví dụ:
            [X[0] <= 0.5]
            /           \
    [X[1] <= 1.2]    Giá trị: 0
        /       \
    Giá trị: 1  Giá trị: 2
    """
    
    # Tạo các node lá
    leaf1 = Node(value=1)
    leaf2 = Node(value=2)
    leaf3 = Node(value=0)
    
    # Tạo node quyết định
    decision1 = Node(feature=1, threshold=1.2, left=leaf1, right=leaf2)
    root = Node(feature=0, threshold=0.5, left=decision1, right=leaf3)
    
    # Vẽ cây
    ve_cay = VeCay(root)
    ve_cay.ve_cay("vi_du_cay")
    
    # Hiển thị mã nguồn DOT
    print("Mã nguồn DOT:")
    print(ve_cay.hien_thi_source())