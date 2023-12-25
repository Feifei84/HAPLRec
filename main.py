import torch
from time import time
import utility.parser
from model import Model as Mymodel
import pickle
import utility.batch_test
from utility.data_loader import Data
from APL import APL


def main(type_, drop_ratio, n=10):
    args = utility.parser.parse_args()
    utility.batch_test.set_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda', args.gpu_id)
    else:
        device = torch.device('cpu')

    with open(args.data_path + args.dataset + '/uu_nei_' + str(n) + '.pkl', 'rb') as fs:
        uu = pickle.load(fs)
    with open(args.data_path + args.dataset + '/ii_nei_' + str(n) + '.pkl', 'rb') as fs:
        ii = pickle.load(fs)

    dataset = Data(args.data_path + args.dataset)

    Model = Mymodel(dataset.bipartite_graph, uu, ii, args.dim, args.GCNLayer, args.batch_size, device, dataset.num_users, dataset.num_items, drop_ratio, type_).to(device)
    Model.to(device)

    apl = APL(Model.parameters())
    opt = torch.optim.Adam(Model.parameters(), lr=args.lr)

    best_report_recall = 0.
    best_report_ndcg = 0.
    best_report_epoch = 0
    early_stop = 0
    for epoch in range(args.epochs):
        start_time = time()
        if epoch % args.verbose == 0:
            result = utility.batch_test.Test(dataset, Model, device, eval(args.topK), args.multicore,
                                             args.test_batch_size, long_tail=False)
            if result['recall'][0] > best_report_recall:
                early_stop = 0
                best_report_epoch = epoch + 1
                best_report_recall = result['recall'][0]
                best_report_ndcg = result['ndcg'][0]
            else:
                early_stop += 1

            if early_stop >= 50:
                print("early stop! best epoch:", best_report_epoch, "bset_recall:", best_report_recall, ',best_ndcg:',
                      best_report_ndcg)
                with open('./result/' + args.dataset + "/result_" + str(n) + ".txt", "a") as f:
                    f.write(type_+" ")
                    f.write(str(drop_ratio) + " ")
                    f.write(str(best_report_epoch) + " ")
                    f.write(str(best_report_recall) + " ")
                    f.write(str(best_report_ndcg) + "\n")
                break
            else:
                print("recall:", result['recall'], ",precision:", result['precision'], ',ndcg:', result['ndcg'])

        Model.train()
        sample_data = dataset.sample_data_to_train_all()
        users = torch.Tensor(sample_data[:, 0]).long()
        pos_items = torch.Tensor(sample_data[:, 1]).long()
        neg_items = torch.Tensor(sample_data[:, 2]).long()

        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)

        users, pos_items, neg_items = utility.batch_test.shuffle(users, pos_items, neg_items)
        num_batch = len(users) // args.batch_size + 1
        average_loss = 0.
        average_reg_loss = 0.
        average_intra_loss = 0.

        user_sub1, user_sub2 = Model.graph_construction(Model.u_graph)
        item_sub1, item_sub2 = Model.graph_construction(Model.i_graph)
        for batch_i, (batch_users, batch_positive, batch_negative) in enumerate(
                utility.batch_test.mini_batch(users, pos_items, neg_items, batch_size=args.batch_size)):
            batch_mf_loss, batch_emb_loss = Model.bpr_loss(batch_users, batch_positive, batch_negative)
            batch_emb_loss = args.l2 * batch_emb_loss
            batch_loss = batch_emb_loss + batch_mf_loss

            cl_user_loss, cl_item_loss = Model.cal_cl_loss(user_sub1, user_sub2, item_sub1, item_sub2, batch_users, batch_positive)

            opt.zero_grad()
            apl.zero_grad()
            average_intra_loss = average_intra_loss + cl_user_loss.item() + cl_item_loss.item()
            apl.step([batch_loss, cl_user_loss, cl_item_loss], [1, 0.01, 0.01])
            opt.step()

            average_loss += batch_mf_loss.item()
            average_reg_loss += batch_emb_loss.item()

        average_loss = average_loss / num_batch
        average_reg_loss = average_reg_loss / num_batch
        average_intra_loss = average_intra_loss / num_batch
        end_time = time()
        print("\t Epoch: %4d| train time: %.3f | train_loss:%.4f + %.4f + %.4f" % (
            epoch + 1, end_time - start_time, average_loss, average_reg_loss, average_intra_loss))

    print("best epoch:", best_report_epoch)
    print("best recall:", best_report_recall)
    print("best ndcg:", best_report_ndcg)

if __name__ == '__main__':
    main('ED', 0.1)

