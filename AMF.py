from __future__ import absolute_import
from __future__ import division
import os
import math
import numpy as np
import tensorflow as tf
from multiprocessing import Pool   #multiprocessing包是Python中的多进程管理包，调用Poll进程池
from multiprocessing import cpu_count
import argparse  #导入命令行解析包
import logging
from time import time
from time import strftime
from time import localtime
from Dataset import Dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
_user_input = None1
_item_input_pos = None
_batch_size = None
_index = None
_model = None
_sess = None
_dataset = None
_K = None
_feed_dict = None
_output = None    #_output值为1时评估含有噪音的部分，值为0时评估不含噪音部分

#argparse实现在命令行中传递参数

def parse_args():
    parser = argparse.ArgumentParser(description="Run AMF.")   #创建一个命令行解析处理器，将命令行解析成 Python 数据类型所需的全部信息。

    #nargs - 命令行参数应当消耗的数目。
    #type：把从命令行输入的结果转成设置的类型
    #help：参数命令的介绍
    #default：设置参数的默认值

    parser.add_argument('--path', nargs='?', default='Data/', #输入数据路径
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='yelp', #选择一个数据集
                        help='Choose a dataset.')
    parser.add_argument('--verbose', type=int, default=1,  #多少轮迭代输出一次
                        help='Evaluate per X epochs.')
    parser.add_argument('--batch_size', type=int, default=512,   #抽取样本大小
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=2000,     #APR迭代总次数
                        help='Number of epochs.')
    parser.add_argument('--embed_size', type=int, default=64,   #MF分解出来的物品矩阵和用户矩阵的列数
                        help='Embedding size.')
    parser.add_argument('--dns', type=int, default=1,  #对于每一个正样本对应dns个负样本
                        help='number of negative sample for each positive in dns.')
    parser.add_argument('--reg', type=float, default=0,  #MF优化不加噪音构成的损失函数时用的正则化
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--lr', type=float, default=0.05,   #学习率
                        help='Learning rate.')
    parser.add_argument('--reg_adv', type=float, default=1,   #MF优化加了噪音构成的损失函数时用的正则化
                        help='Regularization for adversarial loss')
    parser.add_argument('--restore', type=str, default=None,   #用于判断是否恢复权重
                        help='The restore time_stamp for weights in \Pretrain')
    parser.add_argument('--ckpt', type=int, default=100,   #多少轮迭代保存一次
                        help='Save the model per X epochs.')
    parser.add_argument('--task', nargs='?', default='',  #对当前的任务命名
                        help='Add the task name for launching experiments')
    parser.add_argument('--adv_epoch', type=int, default=0,    #BPR迭代次数
                        help='Add APR in epoch X, when adv_epoch is 0, it\'s equivalent to pure AMF.\n '
                             'And when adv_epoch is larger than epochs, it\'s equivalent to pure MF model. ')
    parser.add_argument('--adv', nargs='?', default='grad',  #生成对抗样本的方法：梯度下降法或随机法 （详细转220行）
                        help='Generate the adversarial sample by gradient method or random method')
    parser.add_argument('--eps', type=float, default=0.5,  #l2范化的最小值边界
                        help='Epsilon for adversarial weights.')
    return parser.parse_args()


# data sampling and shuffling 数据采样和改组

# input: dataset(Mat, List, Rating, Negatives), batch_choice, num_negatives
# output: [_user_input_list, _item_input_pos_list]
def sampling(dataset):
    _user_input, _item_input_pos = [], []
    for (u, i) in dataset.trainMatrix.keys(): #遍历训练集
        # positive instance 相关的项目
        _user_input.append(u)
        _item_input_pos.append(i)
    return _user_input, _item_input_pos


def shuffle(samples, batch_size, dataset, model):
    global _user_input
    global _item_input_pos
    global _batch_size
    global _index
    global _model
    global _dataset
    _user_input, _item_input_pos = samples   #训练集中用户和相关项目一一对应构成的list列表
    _batch_size = batch_size   #抽取样本大小
    _index = range(len(_user_input))  #创建一个整数列表[0,len(_user_input))
    _model = model   #MF_BPR
    _dataset = dataset  #数据初始化Dataset.py
    np.random.shuffle(_index)   #打乱顺序函数
    num_batch = len(_user_input) // _batch_size

    pool = Pool(cpu_count())  #默认值是系统上最大可用的CPU数量
    res = pool.map(_get_train_batch, range(num_batch))   #range(num_batch):要处理的数据列表，_get_train_batch：处理range(num_batch)列表中数据的函数
    pool.close()  #关闭进程池（pool），使其不在接受新的任务
    pool.join()  #主进程阻塞等待子进程的退出
    # 把_get_train_batch函数的四个返回值一一分别赋给这四个变量
    user_list = [r[0] for r in res]             #user_list：随机选取的用户集合
    item_pos_list = [r[1] for r in res]         #item_pos_list：每一个随机选取的用户对应的相关项目
    user_dns_list = [r[2] for r in res]         #user_dns_list：对应user_list随机选取的用户，只是一个user_list中的用户对应在user_dns_list有dns个相同的用户
    item_dns_list = [r[3] for r in res]         #item_dns_list：user_dns_list中的每一个用户对应的随机选取的与之不相关的项目的集合
    return user_list, item_pos_list, user_dns_list, item_dns_list


def _get_train_batch(i): #一共调用num_batch次
    user_batch, item_batch = [], [] #在训练集中：user_batch：随机选的用户集 item_batch：每个随机选的用户对应的相关项目集
    user_neg_batch, item_neg_batch = [], []  #user_neg_batch：随机选的用户集  item_neg_batch：每个随机用户在训练集中的dns个不相关的项目组成的集合
    begin = i * _batch_size
    for idx in range(begin, begin + _batch_size):    #idx范围[begin,begin+batch_size)
        user_batch.append(_user_input[_index[idx]])     #在用户集和相关项目中分别随机抽取_batch_size个放入user_batch ，item_batch
        item_batch.append(_item_input_pos[_index[idx]])
        for dns in range(_model.dns):   #一个用户相关的项目对应dns个不相关的项目
            user = _user_input[_index[idx]]  #随机抽取的一个用户
            user_neg_batch.append(user)
            # negtive k
            gtItem = _dataset.testRatings[user][1]  #抽取的那一个用户在测试集中对应的相关的项目赋给gtItem
            j = np.random.randint(_dataset.num_items)  #在[0,num_items)随机抽取一个项目
            while j in _dataset.trainList[_user_input[_index[idx]]]: #如果j项目是该用户的训练集中的相关项目则重新生成
                j = np.random.randint(_dataset.num_items)
            item_neg_batch.append(j)   #将随机生成的与该用户不想关的项目放入item_neg_batch
    return np.array(user_batch)[:, None], np.array(item_batch)[:, None], \     #将形式转化为：array([[x1],[x2],...,[xn]])
           np.array(user_neg_batch)[:, None], np.array(item_neg_batch)[:, None]



# prediction model
class MF:
    def __init__(self, num_users, num_items, args):
        self.num_items = num_items  #项目数
        self.num_users = num_users  #用户数
        self.embedding_size = args.embed_size   #MF分解出来的物品矩阵和用户矩阵的列数
        self.learning_rate = args.lr #学习率
        self.reg = args.reg   #MF优化未加噪音构成的损失函数时用的正则化
        self.dns = args.dns   #对于每一个正样本对应dns个负样本
        self.adv = args.adv   #生成对抗样本的方法：梯度下降法或随机法
        self.eps = args.eps   #加扰动的最大范数约束
        self.adver = args.adver #0代表BPR优化模型，1表示用APR优化
        self.reg_adv = args.reg_adv     #MF优化加了噪音构成的损失函数时用的正则化
        self.epochs = args.epochs    #迭代总次数

       def _create_placeholders(self):
        with tf.name_scope("input_data"):  # tf.name_scope可以让变量有相同的命名，只是限于tf.Variable的变量
            # placeholder，占位符，在tensorflow中类似于函数参数，运行时必须传入值
            self.user_input = tf.placeholder(tf.int32, shape=[None, 1], name="user_input")  # [None, 1]：行数不定，一列
            self.item_input_pos = tf.placeholder(tf.int32, shape=[None, 1], name="item_input_pos")
            self.item_input_neg = tf.placeholder(tf.int32, shape=[None, 1], name="item_input_neg")

    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.embedding_P = tf.Variable( #tf.truncated_normal(shape, mean, stddev) :shape表示生成张量的维度，mean是均值，stddev是标准差
                tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
                name='embedding_P', dtype=tf.float32)  # embedding_P = (users, embedding_size)
            self.embedding_Q = tf.Variable(
                tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                name='embedding_Q', dtype=tf.float32)  # embedding_Q = (items, embedding_size)
            #delta_P，delta_Q是embedding_P，embedding_Q的噪音
            self.delta_P = tf.Variable(tf.zeros(shape=[self.num_users, self.embedding_size]),
                                       name='delta_P', dtype=tf.float32, trainable=False)  # delta_P = (users, embedding_size) 全零数组
            self.delta_Q = tf.Variable(tf.zeros(shape=[self.num_items, self.embedding_size]),
                                       name='delta_Q', dtype=tf.float32, trainable=False)  # delta_Q = (items, embedding_size)

            self.h = tf.constant(1.0, tf.float32, [self.embedding_size, 1], name="h")
            #创建h=[embedding_size,1]全1.0张量

    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            # reduce_sum 应该理解为压缩求和，用于降维
            #tf.nn.embedding_lookup（tensor, id）:tensor就是输入张量，id就是张量对应的索引
            # tf.reduce_sum(x, 1) ：按行求和
            self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)  #shape(None,1,embedding_size)--->>shape(None,embedding_size)
            self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, item_input), #embedding_q = shape(None,embedding_size)
                                             1)  # (b, embedding_size)
            return tf.matmul(self.embedding_p * self.embedding_q, self.h), self.embedding_p, self.embedding_q # (b, embedding_size) * (embedding_size, 1)

    def _create_inference_adv(self, item_input):
        with tf.name_scope("inference_adv"):
            # embedding look up
            self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1) #shape(None,embedding_size)
            self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, item_input),   #shape(None,embedding_size)
                                             1)  # (b, embedding_size)
            # add adversarial noise 增加对抗性噪音
            self.P_plus_delta = self.embedding_p + tf.reduce_sum(tf.nn.embedding_lookup(self.delta_P, self.user_input),1)  #shape(None,embedding_size)
            self.Q_plus_delta = self.embedding_q + tf.reduce_sum(tf.nn.embedding_lookup(self.delta_Q, item_input), 1)
            return tf.matmul(self.P_plus_delta * self.Q_plus_delta, self.h), self.embedding_p, self.embedding_q  # (b, embedding_size) * (embedding_size, 1)
    #构建损失函数
    def _create_loss(self):
        with tf.name_scope("loss"):
            # loss for L(Theta)
            self.output, embed_p_pos, embed_q_pos = self._create_inference(self.item_input_pos)  #返回矩阵R=P*Q 用户矩阵P  物品矩阵Q
            self.output_neg, embed_p_neg, embed_q_neg = self._create_inference(self.item_input_neg)  #返回未加入噪音的R P Q矩阵
            #tf.clip_by_value(A, min, max)：输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。小于min的让它等于min，大于max的元素的值等于max。
            self.result = tf.clip_by_value(self.output - self.output_neg, -80.0, 1e8)
            # self.loss = tf.reduce_sum(tf.log(1 + tf.exp(-self.result))) # this is numerically unstable
            #tf.nn.softplus(features, name = None)：features: 一个Tensor   name: （可选）为这个操作取一个名字。 作用：计算激活函数softplus，即log( exp( features ) + 1)。
            self.loss = tf.reduce_sum(tf.nn.softplus(-self.result))

            # loss to be omptimized 优化损失函数   reg：正则化   tf.square()是对a里的每一个元素求平方
            #reduce_mean：返回一个只有一个元素的张量。
            self.opt_loss = self.loss + self.reg * tf.reduce_mean(tf.square(embed_p_pos) + tf.square(embed_q_pos) + tf.square(embed_q_neg)) # embed_p_pos == embed_q_neg

            if self.adver:
                # loss for L(Theta + adv_Delta)    #返回加入噪音的R P Q矩阵
                self.output_adv, embed_p_pos, embed_q_pos = self._create_inference_adv(self.item_input_pos)
                self.output_neg_adv, embed_p_neg, embed_q_neg = self._create_inference_adv(self.item_input_neg)
                self.result_adv = tf.clip_by_value(self.output_adv - self.output_neg_adv, -80.0, 1e8)
                # self.loss_adv = tf.reduce_sum(tf.log(1 + tf.exp(-self.result_adv)))
                self.loss_adv = tf.reduce_sum(tf.nn.softplus(-self.result_adv))
                self.opt_loss += self.reg_adv * self.loss_adv + self.reg * tf.reduce_mean(tf.square(embed_p_pos) + tf.square(embed_q_pos) + tf.square(embed_q_neg))


    def _create_adversarial(self):
        with tf.name_scope("adversarial"):
            # generate the adversarial weights by random method 通过随机方法生成对抗权重
            if self.adv == "random":
                # generation  随机生成均值为0，标准差为0.01的正态分布
                self.adv_P = tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01)
                self.adv_Q = tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01)

                # normalization and multiply epsilon
                #tf.nn.l2_normalize(x, dim, epsilon=1e-12, name=None) ：x为输入的向量  dim为l2范化的维数，dim取值为0按列或1按行
                #l2范化（按列）：a11=a11/sqrt(a11^2 + a21^2 +...+ an1^2)  a21=a21/sqrt(a11^2 + a21^2 +...+ an1^2) ......
                self.update_P = self.delta_P.assign(tf.nn.l2_normalize(self.adv_P, 1) * self.eps)
                self.update_Q = self.delta_Q.assign(tf.nn.l2_normalize(self.adv_Q, 1) * self.eps)

            # generate the adversarial weights by gradient-based method 通过基于梯度的方法生成对抗权重
            elif self.adv == "grad":
                # return the IndexedSlice Data: [(values, indices, dense_shape)]
                # grad_var_P: [grad,var], grad_var_Q: [grad, var]
                self.grad_P, self.grad_Q = tf.gradients(self.loss, [self.embedding_P, self.embedding_Q])

                # convert the IndexedSlice Data to Dense Tensor
                self.grad_P_dense = tf.stop_gradient(self.grad_P)
                self.grad_Q_dense = tf.stop_gradient(self.grad_Q)

                # normalization: new_grad = (grad / |grad|) * eps
                self.update_P = self.delta_P.assign(tf.nn.l2_normalize(self.grad_P_dense, 1) * self.eps)
                self.update_Q = self.delta_Q.assign(tf.nn.l2_normalize(self.grad_Q_dense, 1) * self.eps)

    def _create_optimizer(self):  #优化器Adagrad优化参数
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.opt_loss)

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()
        self._create_adversarial()


# training
#model：MF_BPR  epoch_start：训练开始  epoch_end：训练结束  time_stamp：时间戳
def training(model, dataset, args, epoch_start, epoch_end, time_stamp):  # saver is an object to save pq
    with tf.Session() as sess:
        # initialized the save op 初始化保存操作
        if args.adver:   #args.adver=1
            ckpt_save_path = "Pretrain/%s/APR/embed_%d/%s/" % (args.dataset, args.embed_size, time_stamp)  # dataset数据集名称，矩阵分解列数，时间戳
            ckpt_restore_path = "Pretrain/%s/MF_BPR/embed_%d/%s/" % (args.dataset, args.embed_size, time_stamp)
        else:
            ckpt_save_path = "Pretrain/%s/MF_BPR/embed_%d/%s/" % (args.dataset, args.embed_size, time_stamp)
            ckpt_restore_path = 0 if args.restore is None else "Pretrain/%s/MF_BPR/embed_%d/%s/" % (args.dataset, args.embed_size, args.restore)
                                                    #如果restore=None，则ckpt_restore_path=0，否则执行else部分
        if not os.path.exists(ckpt_save_path):   #如果不存在路径ckpt_save_path，则创建
            os.makedirs(ckpt_save_path)
        if ckpt_restore_path and not os.path.exists(ckp t_restore_path):# 如果ckpt_restore_path非0，对应的目录不存在，则创建目录
            os.makedirs(ckpt_restore_path)
        #模型的保护与恢复
        saver_ckpt = tf.train.Saver({'embedding_P': model.embedding_P, 'embedding_Q': model.embedding_Q})

        # pretrain or not 是否预先训练，初始化模型的参数
        sess.run(tf.global_variables_initializer())



        # restore the weights when pretrained
        if args.restore is not None or epoch_start:  #如果不是第一次训练可以加载模型
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_restore_path + 'checkpoint')) #tf.train.get_checkpoint_state函数通过checkpoint文件找到模型文件名。
            if ckpt and ckpt.model_checkpoint_path:
                saver_ckpt.restore(sess, ckpt.model_checkpoint_path)
        # initialize the weights 从头开始初始化权重
        else:
            logging.info("Initialized from scratch")
            print "Initialized from scratch"

        # initialize for Evaluate
        eval_feed_dicts = init_eval_model(model, dataset)  #加载评估模型
        #返回eval_feed_dicts格式如下：
        #[(array([[0],[0],...,[0]],dtype=int32),array([[66],[67],[68],...])),(array([[1],[1],...], dtype=int32), array([[1],[2],[3],...])),...]
        # sample the data  对数据进行抽样
        samples = sampling(dataset)   #返回_user_input, _item_input_pos，即训练集中的用户和相关项目构成的一一对应的列表

        # initialize the max_ndcg to memorize the best result 初始化max_ndcg以记住最佳结果
        max_ndcg = 0
        best_res = {}

        # train by epoch 按照训练次数来进行训练
        for epoch_count in range(epoch_start, epoch_end+1):

            # initialize for training batches
            batch_begin = time()   #记录开始时间
            batches = shuffle(samples, args.batch_size, dataset, model)
            batch_time = time() - batch_begin  #记录时间差

            # compute the accuracy before training 在训练前计算准确度
            prev_batch = batches[0], batches[1], batches[3]   #构成一个三元组：用户 与该用户相关的相关项目 不相关项目
            _, prev_acc = training_loss_acc(model, sess, prev_batch, output_adv=0)   #output_adv值为1时代表调用返回MF含有噪音部分，为0时调用返回不含噪音部分
                    #prev_acc：由相关项目得到的矩阵output_pos和不相关项目得到的output_neg的差值
            # training the model
            train_begin = time()
            train_batches = training_batch(model, sess, batches, args.adver)   #训练模型优化器，返回优化了数据的user_input, item_input_pos, item_input_neg
            train_time = time() - train_begin

            if epoch_count % args.verbose == 0:    #verbose:多少轮迭代输出一次
                _, ndcg, cur_res = output_evaluate(model, sess, dataset, train_batches, eval_feed_dicts,
                                                   epoch_count, batch_time, train_time, prev_acc, output_adv=0)

            # print and log the best result
            if max_ndcg < ndcg:
                max_ndcg = ndcg
                best_res['result'] = cur_res
                best_res['epoch'] = epoch_count

            if model.epochs == epoch_count:
                print "Epoch %d is the best epoch" % best_res['epoch']
                for idx, (hr_k, ndcg_k, auc_k) in enumerate(np.swapaxes(best_res['result'], 0, 1)):
                    res = "K = %d: HR = %.4f, NDCG = %.4f AUC = %.4f" % (idx + 1, hr_k, ndcg_k, auc_k)
                    print res

            # save the embedding weights
            if args.ckpt > 0 and epoch_count % args.ckpt == 0:
                saver_ckpt.save(sess, ckpt_save_path + 'weights', global_step=epoch_count)

        saver_ckpt.save(sess, ckpt_save_path + 'weights', global_step=epoch_count)

#model：MF   dataset:数据集处理  train_batches：优化了数据的user_input, item_input_pos, item_input_neg
#eval_feed_dicts：加载评估模型，返回eval_feed_dicts格式如下：
#[(array([[0],[0],...,[0]],dtype=int32),array([[66],[67],[68],...])),(array([[1],[1],...], dtype=int32), array([[1],[2],[3],...])),...]
#epoch_count：第epoch_count轮    batch_time：时间差    train_time：优化数据的时间差
#prev_acc：由相关项目得到的矩阵output_pos和不相关项目得到的output_neg的差值    output_adv：初始值为0

def output_evaluate(model, sess, dataset, train_batches, eval_feed_dicts, epoch_count, batch_time, train_time, prev_acc,
                    output_adv):
    loss_begin = time()
    train_loss, post_acc = training_loss_acc(model, sess, train_batches, output_adv)     #train_loss：训练中的损失值
    loss_time = time() - loss_begin

    eval_begin = time()
    result = evaluate(model, sess, dataset, eval_feed_dicts, output_adv)    #对模型的效果进行评估，返回hr, ndcg, auc
    eval_time = time() - eval_begin

    # check embedding    检查嵌入
    embedding_P, embedding_Q = sess.run([model.embedding_P, model.embedding_Q])

    hr, ndcg, auc = np.swapaxes(result, 0, 1)[-1]
    res = "Epoch %d [%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f ACC = %.4f ACC_adv = %.4f [%.1fs], |P|=%.2f, |Q|=%.2f" % \
          (epoch_count, batch_time, train_time, hr, ndcg, prev_acc,
           post_acc, eval_time, np.linalg.norm(embedding_P), np.linalg.norm(embedding_Q))

    print res

    return post_acc, ndcg, result


# input: batch_index (shuffled), model, sess, batches.
# do: train the model optimizer  训练模型优化器
def training_batch(model, sess, batches, adver=False):
    user_input, item_input_pos, user_dns_list, item_dns_list = batches  #用户 对应相关项目  用户（dns=1，即与前面的形式完全相同）  对应的不相关的项目
    # dns for every mini-batch
    # dns = 1, i.e., BPR
    if model.dns == 1:     #用户的一个相关项目对应的dns个不相关的项目
        item_input_neg = item_dns_list
        # for BPR training
        for i in range(len(user_input)):
            feed_dict = {model.user_input: user_input[i],
                         model.item_input_pos: item_input_pos[i],
                         model.item_input_neg: item_input_neg[i]}
            if adver:
                sess.run([model.update_P, model.update_Q], feed_dict)
            sess.run(model.optimizer, feed_dict)
    # dns > 1, i.e., BPR-dns
    elif model.dns > 1:
        item_input_neg = []
        for i in range(len(user_input)):
            # get the output of negtive sample    得到负样本的输出
            feed_dict = {model.user_input: user_dns_list[i],
                         model.item_input_neg: item_dns_list[i]}
            output_neg = sess.run(model.output_neg, feed_dict)
            # select the best negtive sample as for item_input_neg   为item_input_neg选择最佳负片样本
            item_neg_batch = []
            for j in range(0, len(output_neg), model.dns):
                item_index = np.argmax(output_neg[j: j + model.dns])
                item_neg_batch.append(item_dns_list[i][j: j + model.dns][item_index][0])
            item_neg_batch = np.array(item_neg_batch)[:, None]
            # for mini-batch BPR training
            feed_dict = {model.user_input: user_input[i],
                         model.item_input_pos: item_input_pos[i],
                         model.item_input_neg: item_neg_batch}
            sess.run(model.optimizer, feed_dict)
            item_input_neg.append(item_neg_batch)
    return user_input, item_input_pos, item_input_neg


# calculate the gradients   计算梯度
# update the adversarial noise   更新对抗性噪音
def adv_update(model, sess, train_batches):
    user_input, item_input_pos, item_input_neg = train_batches
    # reshape mini-batches into a whole large batch
    user_input, item_input_pos, item_input_neg = \
        np.reshape(user_input, (-1, 1)), np.reshape(item_input_pos, (-1, 1)), np.reshape(item_input_neg, (-1, 1))
    feed_dict = {model.user_input: user_input,
                 model.item_input_pos: item_input_pos,
                 model.item_input_neg: item_input_neg}

    return sess.run([model.update_P, model.update_Q], feed_dict)


# input: MF, sess, batches
# output: training_loss
def training_loss_acc(model, sess, train_batches, output_adv):   #output_adv初始值为0
    train_loss = 0.0
    acc = 0
    num_batch = len(train_batches[1])    #与用户相关的项目共分的组数，与100行的num_batch是一样的
    user_input, item_input_pos, item_input_neg = train_batches   #用户  相关项目  不相关项目
    for i in range(len(user_input)):   #len(user_input)=len(train_batches[1])=num_batch
        # print user_input[i][0]. item_input_pos[i][0], item_input_neg[i][0]
        feed_dict = {model.user_input: user_input[i],      #feed_dict的作用是给使用placeholder创建出来的tensor赋值
                     model.item_input_pos: item_input_pos[i],
                     model.item_input_neg: item_input_neg[i]}
        if output_adv:  #output_adv值为1时代表调用返回MF含有噪音部分，为0时调用返回不含噪音部分
            loss, output_pos, output_neg = sess.run([model.loss_adv, model.output_adv, model.output_neg_adv], feed_dict)
        else:
            loss, output_pos, output_neg = sess.run([model.loss, model.output, model.output_neg], feed_dict)
        train_loss += loss
        acc += ((output_pos - output_neg) > 0).sum() / len(output_pos)    #由相关项目得到的矩阵output_pos和不相关项目得到的output_neg的差值
    return train_loss / num_batch, acc / num_batch


def init_eval_model(model, dataset):
    begin_time = time() #开始时间
    global _dataset
    global _model
    _dataset = dataset    #数据集
    _model = model      #MF_BPR

    pool = Pool(cpu_count())   #参数CPU_COUNT指定了可以同时使用的CPU的数量，默认值是系统上最大可用的CPU数量  Pool类可以提供指定数量的进程供用户调用
    feed_dicts = pool.map(_evaluate_input, range(_dataset.num_users))   #map()函数会将[0,num_users)的列表元素一个个的传入_evaluate_input函数
    pool.close() #关闭进程池（pool），使其不在接受新的任务。
    pool.join()  #主进程阻塞等待子进程的退出， join方法要在close或terminate之后使用。

    print("Load the evaluation model done [%.1f s]" % (time() - begin_time))   #加载评估模型完成，输出加载模型所用的时间
    return feed_dicts


def _evaluate_input(user):
    # generate items_list   生成items_list
    test_item = _dataset.testRatings[user][1]  #测试集中每一个user对应的相关用户
    item_input = set(range(_dataset.num_items)) - set(_dataset.trainList[user])#_dataset.trainList[user]：训练集中所有与user用户相关用户构成的列表[item0,item1,...,itemn]
    #item_input:num_item在[0,num_items)范围内与用户不相关的项目组成的集合，{item0,item1,item2,...}
    if test_item in item_input:     #如果所测试项目test_item在不相关集合item_input内，就把他从集合中剔除
        item_input.remove(test_item)
    item_input = list(item_input)    #将数据形式由集合该为列表,[item0,item1,item2,...]
    item_input.append(test_item)   #在列表item_input末尾添加test_item
    user_input = np.full(len(item_input), user, dtype='int32')[:, None]   #np.full：创建一个由常数user填充的数组[ len(item_input) , 1] ,array类型
    item_input = np.array(item_input)[:, None]   #将item_input该为[ len(item_input) , 1] ,array类型,原数据不变，只改变类型
    return user_input, item_input

#评估模型
def evaluate(model, sess, dataset, feed_dicts, output_adv):
    global _model    #MF
    global _K
    global _sess
    global _dataset
    global _feed_dicts
    global _output
    _dataset = dataset    #数据集处理
    _model = model     #MF
    _sess = sess
    _K = 100
    _feed_dicts = feed_dicts    #加载的评估模型，形式如下
    #[(array([[0],[0],...,[0]],dtype=int32),array([[66],[67],[68],...])),(array([[1],[1],...], dtype=int32), array([[1],[2],[3],...])),...]
    _output = output_adv   #初始值为0，_output值为1时评估含有噪音的部分，值为0时评估不含噪音部分

    res = []
    for user in range(_dataset.num_users):
        res.append(_eval_by_user(user))    #将每一轮的数据预测加入到列表res之后
    res = np.array(res)
    hr, ndcg, auc = (res.mean(axis=0)).tolist()

    return hr, ndcg, auc


def _eval_by_user(user):
    # get prredictions of data in testing set    获得测试集中的数据预测
    user_input, item_input = _feed_dicts[user]
    feed_dict = {_model.user_input: user_input, _model.item_input_pos: item_input}
    if _output:     #_output值为1时评估含有噪音的部分，值为0时评估不含噪音部分
        predictions = _sess.run(_model.output_adv, feed_dict)
    else:
        predictions = _sess.run(_model.output, feed_dict)

    neg_predict, pos_predict = predictions[:-1], predictions[-1]
    position = (neg_predict >= pos_predict).sum()    #由相关数据得到的预测和不相关数据的预测的差值

    # calculate from HR@1 to HR@100, and from NDCG@1 to NDCG@100, AUC   从HR @ 1到HR @ 100，从NDCG @ 1到NDCG @ 100，AUC计算
    hr, ndcg, auc = [], [], []
    for k in range(1, _K + 1):
        hr.append(position < k)
        ndcg.append(math.log(2) / math.log(position + 2) if position < k else 0)
        auc.append(1 - (position / len(neg_predict)))  # formula: [#(Xui>Xuj) / #(Items)] = [1 - #(Xui<=Xuj) / #(Items)]

    return hr, ndcg, auc

def init_logging(args, time_stamp):
    path = "Log/%s_%s/" % (strftime('%Y-%m-%d_%H', localtime()), args.task)
    if not os.path.exists(path):   #os.path.exists()就是判断括号里的文件是否存在的意思，括号内的可以是文件路径。
        os.makedirs(path)     #os.makedirs() 方法用于递归创建目录
    logging.basicConfig(filename=path + "%s_log_embed_size%d_%s" % (args.dataset, args.embed_size, time_stamp),
                        level=logging.INFO)  #filename: 指定日志文件名，level: 设置日志级别
    logging.info(args) #输出日志的信息
    print args


if __name__ == '__main__':

    time_stamp = strftime('%Y_%m_%d_%H_%M_%S', localtime()) #显示当前时间

    # 初始化参数和日志记录
    args = parse_args()  #初始化参数
    init_logging(args, time_stamp)  #初始化记录

    # initialize dataset 初始化数据集
    dataset = Dataset(args.path + args.dataset)

    args.adver = 0  #0代表BPR优化模型，1表示用APR优化
    # initialize MF_BPR models 初始化MF_BPR模型
    MF_BPR = MF(dataset.num_users, dataset.num_items, args)
    MF_BPR.build_graph()

    print "Initialize MF_BPR"

    # start training
    training(MF_BPR, dtaaset, args, epoch_start=0, epoch_end=args.adv_epoch-1, time_stamp=time_stamp)

    args.adver = 1
    # instialize AMF model
    AMF = MF(dataset.num_users, dataset.num_items, args)
    AMF.build_graph()

    print "Initialize AMF"

    # start training
    training(AMF, dataset, args, epoch_start=args.adv_epoch, epoch_end=args.epochs, time_stamp=time_stamp)
