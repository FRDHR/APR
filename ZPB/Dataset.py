'''
Created on Aug 8, 2016
Processing datasets. 
@author: Xiangnan He (xiangnanhe@gmail.com)
Update on 5 17, 2019 -- Laugh
'''
import scipy.sparse as sp
import numpy as np
from time import time

class Dataset(object):
    '''
    加载数据文件
        trainMatrix: 将等级记录作为类数据的稀疏矩阵加载
        trainList：将评级记录加载为列表以加速用户的特征检索
        testRatings：为课程评估加载一次性评分测试
        testNegatives：对未被评分的项目进行抽样
    '''

    def __init__(self, path):
        '''
        构造函数
        '''
        self.trainMatrix = self.load_training_file_as_matrix(path + ".train.rating")
        self.trainList = self.load_training_file_as_list(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        assert len(self.testRatings) == len(self.testNegatives)
        self.num_users, self.num_items = self.trainMatrix.shape

    '''
    example:

    - Inputs: 0	25	5	978824351
              1	133	3	978300174
              2	207	4	978298504
              3	208	4	978294282
              4	222	2	978246585
              5	396	5	978239019
              ...

    - Outputs: [[0, 25],
                [1, 133],
                [2, 207],
                [3, 208],
                [4, 222],
                [5, 396],
                ...]
    '''
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    '''
    example:

    - Inputs: (0,25) 1064 174 2791 3373 269 2678 1902 3641 ...
              (1,133) 1072 3154	3368 3644 549 1810 937 1514 ...
              ...

    - Outputs: [[1064, 174, 2791, 3373, 269, 2678, 1902, 3641, ...],
               [1072, 3154,	3368, 3644, 549, 1810, 937, 1514, ...],
               ...]
    '''
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    '''
    example:

    - Inputs: 0	32	4	978824330
              0	34	4	978824330
              0	4	5	978824291
              0	35	4	978824291
              0	30	4	978824291
              0	29	3	978824268
              ...

    - Outputs:  (0, 32)	1.0
                (0, 34)	1.0
                (0, 4)	1.0
                (0, 35)	1.0
                (0, 30)	1.0
                (0, 29)	1.0 
                ...
    '''
    def load_training_file_as_matrix(self, filename):
        '''
        读取 .rating 文件和 Return dok 矩阵
        .rating 文件的第一行是：num_users \ t  num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()
        print "already load the trainMatrix..."
        return mat

    '''
    example:

    - Inputs: same as load_training_file_as_matrix;

    - Outputs: [[32,
                 34,
                 4,
                 35,
                 30,
                 29,
                ...
                ]]
    '''
    def load_training_file_as_list(self, filename):
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
                    lists.append(items)
                    items = []
                    u_ += 1
                index += 1
                #if index<300:
                items.append(i)
                line = f.readline()
        lists.append(items)
        print "already load the trainList..."
        return lists