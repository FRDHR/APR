'''
处理数据集
'''
import scipy.sparse as sp   # 稀疏矩阵库scipy.sparse
import numpy as np
from time import time

class Dataset(object):
    '''
    加载数据文件:
        trainMatrix(训练模型构建的矩阵): 记录训练数据的评分为一个稀疏矩阵
        trianList: 记录训练数据的评分为一个列表来加快用户的特征检索
        testRatings: load leave-one-out rating test for class Evaluate
        testNegatives: 对未经用户评定的物品进行抽样
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix = self.load_training_file_as_matrix(path + ".train.rating") #返回训练集构成的稀疏矩阵记录相关情况
        self.trainList = self.load_training_file_as_list(path + ".train.rating") #返回训练数据构建的列表形式
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating") #返回测试集构建的列表，形式为[[user, item],[user, item],...[user, item]]
        self.testNegatives = self.load_negative_file(path + ".test.negative")   #返回未经用户评定的物品构成的列表
        assert len(self.testRatings) == len(self.testNegatives)
        self.num_users, self.num_items = self.trainMatrix.shape #返回训练集构成的稀疏矩阵的大小，即num_users：用户数 num_items：项目数

    def load_rating_file_as_list(self, filename):
        #读取Data/yelp.test.rating文件
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename):
        # 读取Data/yelp.test.negative文件
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:   #将negative文件中的数据全部转换为int型存储为列表
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_training_file_as_matrix(self, filename):
        '''
        读取Data/yelp.train.rating文件然后返回一个dok矩阵
        The first line of .rating file is: num_users（用户）\t num_items（项目）
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)   #用户最大编号
                num_items = max(num_items, i)   #最大项目号
                line = f.readline()
        # 矩阵构建
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32) #dok_matrix创建稀疏矩阵类似于字典，keys：位置，values：值
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2]) #rating：评分
                if (rating > 0): #记录与用户相关的项目
                    mat[user, item] = 1.0
                line = f.readline()
        print "already load the trainMatrix..."
        return mat

    def load_training_file_as_list(self, filename):
        #读取Data/yelp.train.rating文件
        # Get number of users and items
        u_ = 0
        lists, items = [], []
        with open(filename, "r") as f:
            line = f.readline()
            index = 0
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                if u_ < u:
                    index = 0
                    lists.append(items) #在lists末尾添加items
                    items = []
                    u_ += 1
                index += 1
                #if index<300:
                items.append(i)
                line = f.readline()
        lists.append(items)
        print "already load the trainList..."
        return lists #每个用户对应一个items列表，最后将这些列表全都放在lists列表中
    #list对应的格式为：[[item01,item02,...,item0x],...,[itemy0,itemy1,...,itemyx]]