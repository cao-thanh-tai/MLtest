import math







class DecisionTree:
    def __init__(self):
        self.feature = []
        self.threshold = []
        self.left = []
        self.right = []
        self.value = []
        pass
    
    def E(self, y):
        freq = {}
        n=len(y)
        for i in range(n):
            val = y[i]
            if val in freq  : freq[val] +=1
            else : freq[val] = 1
        sum=0.0
        for key in freq:
            pi=freq[key]/n
            sum+=pi*math.log(pi,2)
        return -sum
    def EF(self, s, y):
        freq = {}
        ds = {}
        n = len(s)
        for i in range(n):
            if s[i] in freq :
                freq[s[i]] +=1
                ds[s[i]].append(y[i])
            else : 
                freq[s[i]] = 1
                ds[s[i]] = [y[i]]
        sum=0.0
        for key in freq:
            sum += (freq[key]/n)*self.E(ds[key])
        self.threshold.append(list(freq.keys())[:-1])
        return sum

    def IG(self, y, s):
        return self.E(y)-self.EF(s,y)
    
    def update_fea(self, S, y):
        for i in range(len(S[0])):
            s=S[:,i]
            self.feature.append([self.IG(y,s),i,self.threshold[i]])
        self.feature=sorted(self.feature, key=lambda x: x[0], reverse=True)
        self.threshold=[hang[2] for hang in self.feature]
        self.feature=[hang[1] for hang in self.feature]
        pass
    
    def build(self, S, y):
        self.update_fea(S,y)
        bra_j = -1 if len(self.threshold[0]) == 1 else 0
        self.build_tree(S, y, 0, bra_j)
        pass
    def split_tree(self, S, y, threshold, fea_i):
        S_left= []
        S_right= []
        y_left= []
        y_right = []
        for i in range (len(S)):
            if S[i][fea_i] == threshold :
                S_left.append(S[i])
                y_left.append(y[i])
            else :
                S_right.append(S[i])
                y_right.append(y[i])
        return S_left, y_left, S_right, y_right
    
    def is_pure(self, y):
        if len(y)==0 : return True
        val = y[0]
        for i in range(len(y)):
            if val != y[i] : return False
        return True
    
    def tinh_phan_tram(self, y):
        d={}
        for val in y:
            if val in d:
                d[val]+=1
            else :
                d[val]=1
        n = len(y)
        s="| "
        for key,value in d.items():
            pt=round((value/n)*100,2)
            s +=f"{pt} {key} | "
        return s
    def build_tree(self, S, y, fea_i = 0, bra_j = -1):
        if self.is_pure(y) : 
            self.left.append(None)
            self.right.append(None)
            self.value.append("ko bt" if len(y) == 0 else y[0])
            return
        if fea_i ==len(self.threshold) : 
            print('92',fea_i)
            self.left.append(None)
            self.right.append(None)
            self.value.append("ko bt")
            return
        self.value.append(self.tinh_phan_tram(y))
        count_node=len(self.left)
        self.left.append(count_node+1)
        self.right.append(None)
        if bra_j==-1:
            thre = self.threshold[fea_i][0]
        else: 
            thre = self.threshold[fea_i][bra_j]
        S_l,y_l,S_r,y_r=self.split_tree(S,y,thre,self.feature[fea_i])
        
        if fea_i + 1 == len(self.threshold):
            j = -1
        else :
            j = -1 if len(self.threshold[fea_i+1]) == 1 else 0
        self.build_tree(S_l, y_l, fea_i+1, j)
        
        new_count_node=len(self.left)
        self.right[count_node]=new_count_node
        
        if len(self.threshold[fea_i])!= bra_j and bra_j != -1:
            bra_j+=1
        else :
            bra_j = j
            fea_i += 1
        self.build_tree(S_r, y_r, fea_i, bra_j)
        pass
    
    def predict(self, s):
        n=0
        for i in range(len(self.feature)):
            for j in range(len(self.threshold[i])):
                if not self.left[n] : return self.value[n]
                if s[self.feature[i]]==self.threshold[i][j]:
                    n=self.left[n]
                    break
                else:
                    n=self.right[n]
        if n != None : return self.value[n]
        return "loi"
    