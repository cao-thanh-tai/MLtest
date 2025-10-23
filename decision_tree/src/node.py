class node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature     # Chỉ số thuộc tính để chia
        self.threshold = threshold # Ngưỡng chia (hoặc giá trị phân loại)
        self.left = left           # Node con trái
        self.right = right         # Node con phải
        self.value = value         # Giá trị dự đoán (nếu là lá)
        
