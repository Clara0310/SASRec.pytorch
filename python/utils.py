import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

def build_index(dataset_name):

    ui_mat = np.loadtxt('data/%s.txt' % dataset_name, dtype=np.int32)

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neq(1, itemnum + 1, ts)          # Don't need "if nxt != 0"
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 4:                          # To be rigorous, the training set needs at least two data points to learn
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

# TODO: merge evaluate functions for test and val set

# ---------------------------------------------------------
# 請將這段程式碼覆蓋原本 python/utils.py 中的 evaluate 和 evaluate_valid
# ---------------------------------------------------------

def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    # [修改] 定義我們要評估的 K 值列表
    Ks = [5, 10, 20]
    
    # [修改] 使用字典來儲存不同 K 的結果
    NDCG = {k: 0.0 for k in Ks}
    HT = {k: 0.0 for k in Ks}
    
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
        
    # [修改] 全排名需要所有商品的列表 (1 ~ itemnum)
    all_item_idx = list(range(1, itemnum + 1))

    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        # Test 階段：輸入序列包含 Train + Valid
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
            
        rated = set(train[u])
        rated.add(0)
        rated.add(valid[u][0]) # Test 時，Valid 也是已知歷史，需加入 rated 避免被當成負樣本 (視你的實驗設定而定，通常建議加)
        
        # [修改] 全排名預測
        # 輸入 [u], [seq], [所有商品]
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], all_item_idx]])
        predictions = predictions[0] # 取得所有商品的分數向量

        # 取得正確答案的商品 ID 與其分數
        target_item = test[u][0]
        target_score = predictions[target_item - 1] # item_idx 是 1-based，轉 array index 需 -1

        # [修改] Masking: 將歷史買過的商品分數設為無限大 (代表排名最後)
        for i in rated:
            if i != target_item:
                predictions[i - 1] = np.inf
        
        # [修改] 計算排名 (有多少個商品分數比正確答案好/更小)
        rank = (predictions < target_score).sum().item()

        valid_user += 1

        # [修改] 針對每個 K 計算指標
        for k in Ks:
            if rank < k:
                NDCG[k] += 1 / np.log2(rank + 2)
                HT[k] += 1
                
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    # [修改] 回傳字典
    return {k: NDCG[k] / valid_user for k in Ks}, {k: HT[k] / valid_user for k in Ks}


def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    Ks = [5, 10, 20]
    NDCG = {k: 0.0 for k in Ks}
    HT = {k: 0.0 for k in Ks}
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
        
    all_item_idx = list(range(1, itemnum + 1))

    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        # Valid 階段：輸入序列只包含 Train
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], all_item_idx]])
        predictions = predictions[0]

        target_item = valid[u][0]
        target_score = predictions[target_item - 1]

        for i in rated:
            if i != target_item:
                predictions[i - 1] = np.inf

        rank = (predictions < target_score).sum().item()

        valid_user += 1

        for k in Ks:
            if rank < k:
                NDCG[k] += 1 / np.log2(rank + 2)
                HT[k] += 1
                
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return {k: NDCG[k] / valid_user for k in Ks}, {k: HT[k] / valid_user for k in Ks}







# evaluate on test set
# def evaluate(model, dataset, args):
    # [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    # NDCG = 0.0
    # HT = 0.0
    # valid_user = 0.0

    # if usernum>10000:
    #     users = random.sample(range(1, usernum + 1), 10000)
    # else:
    #     users = range(1, usernum + 1)
    # for u in users:

    #     if len(train[u]) < 1 or len(test[u]) < 1: continue

    #     seq = np.zeros([args.maxlen], dtype=np.int32)
    #     idx = args.maxlen - 1
    #     seq[idx] = valid[u][0]
    #     idx -= 1
    #     for i in reversed(train[u]):
    #         seq[idx] = i
    #         idx -= 1
    #         if idx == -1: break
    #     rated = set(train[u])
    #     rated.add(0)
        
    #     # [修改 1] 建立全量商品列表 (1 到 itemnum)
    #     #item_idx = [test[u][0]]
    #     item_idx = list(range(1, itemnum + 1))
        
    #     # [修改 2] 預測所有商品的分數
    #     # 注意: 如果顯卡記憶體(VRAM)不足，這裡可能需要分批(batch)預測，但通常 datasets 不大時可直接跑
    #     predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
    #     predictions = predictions[0] # 取得 (itemnum,) 的分數陣列
        
    #     # [修改 3] 取得目標商品(Ground Truth)的分數與索引
    #     target_item = test[u][0]
    #     target_item_idx = target_item - 1 # 因為 item_idx 是從 1 開始，對應到 array index 要減 1
    #     target_score = predictions[target_item_idx]
        
    #     # [修改 4] Masking (遮蔽): 將訓練集出現過的商品分數設為無限大 (代表表現最差)
    #     # 因為 predictions 是負的 logits (越小越好)，所以設為 np.inf 讓它排到最後面
    #     for item in rated:
    #         if item != target_item: # 當然不能遮蔽掉正確答案
    #             predictions[item - 1] = np.inf
                
    #     # for _ in range(100):
    #     #     t = np.random.randint(1, itemnum + 1)
    #     #     while t in rated: t = np.random.randint(1, itemnum + 1)
    #     #     item_idx.append(t)


    #     predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
    #     predictions = predictions[0] # - for 1st argsort DESC

    #     # [修改 5] 計算排名
    #     # 方法 A: 使用 argsort (較慢但直觀)
    #     # rank = predictions.argsort().argsort()[target_item_idx].item()
        
    #     # 方法 B: 直接計算有多少個商品的分數優於(小於)目標商品 (較快)
    #     rank = (predictions < target_score).sum().item()
    #     # rank = predictions.argsort().argsort()[0].item()

    #     valid_user += 1

    #     if rank < 10:
    #         NDCG += 1 / np.log2(rank + 2)
    #         HT += 1
    #     if valid_user % 100 == 0:
    #         print('.', end="")
    #         sys.stdout.flush()

    # return NDCG / valid_user, HT / valid_user


# evaluate on val set
# def evaluate_valid(model, dataset, args):
    # [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    # NDCG = 0.0
    # valid_user = 0.0
    # HT = 0.0
    # if usernum>10000:
    #     users = random.sample(range(1, usernum + 1), 10000)
    # else:
    #     users = range(1, usernum + 1)
    # for u in users:
    #     if len(train[u]) < 1 or len(valid[u]) < 1: continue

    #     seq = np.zeros([args.maxlen], dtype=np.int32)
    #     idx = args.maxlen - 1
    #     for i in reversed(train[u]):
    #         seq[idx] = i
    #         idx -= 1
    #         if idx == -1: break

    #     rated = set(train[u])
    #     rated.add(0)
    #     item_idx = [valid[u][0]]
    #     for _ in range(100):
    #         t = np.random.randint(1, itemnum + 1)
    #         while t in rated: t = np.random.randint(1, itemnum + 1)
    #         item_idx.append(t)

    #     predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
    #     predictions = predictions[0]

    #     rank = predictions.argsort().argsort()[0].item()

    #     valid_user += 1

    #     if rank < 10:
    #         NDCG += 1 / np.log2(rank + 2)
    #         HT += 1
    #     if valid_user % 100 == 0:
    #         print('.', end="")
    #         sys.stdout.flush()

    # return NDCG / valid_user, HT / valid_user
