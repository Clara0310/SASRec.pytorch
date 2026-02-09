import os
import time
import torch
import argparse

from model import SASRec
from utils import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--norm_first', action='store_true', default=False)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':

    u2i_index, i2u_index = build_index(args.dataset)
    
    # global dataset
    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    # num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    num_batch = (len(user_train) - 1) // args.batch_size + 1
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    # f.write('epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n')
    f.write('epoch val_NDCG@5 val_HR@5 val_NDCG@10 val_HR@10 val_NDCG@20 val_HR@20 test_NDCG@5 test_HR@5 test_NDCG@10 test_HR@10 test_NDCG@20 test_HR@20\n')
    
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = SASRec(usernum, itemnum, args).to(args.device) # no ReLU activation in original SASRec implementation?
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0

    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)
    
    model.train() # enable model training
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
            
    
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
    
    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            # torch.norm(param) returns the square root of the sum of squared weights (‖w‖₂), 
            # should be torch.norm(param)**2 or the way below which is faster.
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.sum(param ** 2)    
            loss.backward()
            adam_optimizer.step()
            print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs

        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            
            # 取得評估結果 (現在是字典)
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            
            # [修改] 定義一個小函數來格式化輸出字串
            def get_metric_str(metrics):
                ndcg, hr = metrics
                # 產生類似 "N@5:0.123, H@5:0.200 | N@10:..." 的字串
                return " | ".join([f"N@{k}:{ndcg[k]:.4f} H@{k}:{hr[k]:.4f}" for k in [5, 10, 20]])

            print(f'\nEpoch: {epoch}, Time: {T:.2f}s')
            print(f'Valid: {get_metric_str(t_valid)}')
            print(f'Test:  {get_metric_str(t_test)}')
            
            # [修改] 儲存最佳模型邏輯
            # 這裡我們通常選定一個主要指標 (例如 NDCG@10) 來決定是否儲存 Best Model
            current_val_score = t_valid[0][10] + t_valid[1][10] # NDCG@10 + HR@10
            best_val_score = best_val_ndcg + best_val_hr # 這是舊變數，你可以自己調整邏輯
            
            
            # 為了相容原本的變數名，我們這裡簡單更新
            if t_valid[0][10] > best_val_ndcg or t_valid[1][10] > best_val_hr:
                best_val_ndcg = max(t_valid[0][10], best_val_ndcg)
                best_val_hr = max(t_valid[1][10], best_val_hr)
                # ... 這裡保留原本的 torch.save 邏輯 ...
                folder = args.dataset + '_' + args.train_dir
                fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
                torch.save(model.state_dict(), os.path.join(folder, fname))

            # [修改] 寫入 log 檔 (轉成字串即可)
            #f.write(str(epoch) + ' ' + str(t_valid) + ' ' + str(t_test) + '\n')
            # [建議修改] 將字典格式化為易讀的字串寫入 log
            # 格式範例: epoch val_NDCG@5 val_HR@5 val_NDCG@10 ... test_NDCG@5 ...
            
            log_str = f"{epoch}"
            for k in [5, 10, 20]:
                log_str += f" {t_valid[0][k]:.4f} {t_valid[1][k]:.4f}" # Valid NDCG, HR
            for k in [5, 10, 20]:
                log_str += f" {t_test[0][k]:.4f} {t_test[1][k]:.4f}"   # Test NDCG, HR
            
            f.write(log_str + '\n')
            f.flush()
            t0 = time.time()
            model.train()
    
        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))
    
    f.close()
    sampler.close()
    print("Done")
