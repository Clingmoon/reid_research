import torch
import numpy as np
import os
from utils.reranking import re_ranking


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def org_cosine_similarity(qf, gf):

    q_norm = torch.norm(qf, p=2, dim=1, keepdim=True)
    g_norm = torch.norm(gf, p=2, dim=1, keepdim=True)
    qf = qf.div(q_norm.expand_as(qf))  # torch.Size([3873, 2048])
    gf = gf.div(g_norm.expand_as(gf))  # torch.Size([3384, 2048])
    dist_mat = - torch.mm(qf, gf.t())

    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def _eval_sorted_indices(indices, q_pids, g_pids, q_camids, g_camids, max_rank):
    all_cmc = []
    all_AP = []
    num_valid_q = 0.

    for q_idx in range(indices.shape[0]):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]

        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        orig_cmc = (g_pids[order] == q_pid).astype(np.int32)[keep]
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = tmp_cmc / (np.arange(1, tmp_cmc.shape[0] + 1) * 1.0)
        AP = (tmp_cmc * orig_cmc).sum() / num_rel
        all_AP.append(AP)

    return all_cmc, all_AP, num_valid_q


def _finalize_cmc_map(all_cmc, all_AP, num_valid_q):
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    return all_cmc, mAP


def _query_chunk_size():
    return max(1, int(os.environ.get('EVAL_QUERY_CHUNK_SIZE', 128)))


def eval_func_chunked(qf, gf, q_pids, g_pids, q_camids, g_camids, max_rank=50,
                      metric='euclidean', query_chunk_size=None, qf2=None, gf2=None):
    num_q = qf.shape[0]
    num_g = gf.shape[0]
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    query_chunk_size = query_chunk_size or _query_chunk_size()
    all_cmc = []
    all_AP = []
    num_valid_q = 0.

    with torch.no_grad():
        qf = qf.float()
        gf = gf.float()
        if metric == 'euclidean':
            gf_t = gf.t().contiguous()
            gf_square = torch.pow(gf, 2).sum(dim=1, keepdim=True).t()
        elif metric in ('cosine', 'cosine_sum'):
            qf = torch.nn.functional.normalize(qf, dim=1, p=2)
            gf = torch.nn.functional.normalize(gf, dim=1, p=2)
            gf_t = gf.t().contiguous()
            if metric == 'cosine_sum':
                qf2 = torch.nn.functional.normalize(qf2.float(), dim=1, p=2)
                gf2 = torch.nn.functional.normalize(gf2.float(), dim=1, p=2)
                gf2_t = gf2.t().contiguous()
        else:
            raise ValueError('Unsupported chunked metric: {}'.format(metric))

        for start in range(0, num_q, query_chunk_size):
            end = min(start + query_chunk_size, num_q)
            q_chunk = qf[start:end]

            if metric == 'euclidean':
                dist_chunk = torch.pow(q_chunk, 2).sum(dim=1, keepdim=True) + gf_square
                dist_chunk.addmm_(q_chunk, gf_t, beta=1, alpha=-2)
            elif metric == 'cosine':
                dist_chunk = -torch.mm(q_chunk, gf_t)
            else:
                dist_chunk = -torch.mm(q_chunk, gf_t)
                dist_chunk.addmm_(qf2[start:end], gf2_t, beta=1, alpha=-1)

            indices = np.argsort(dist_chunk.cpu().numpy(), axis=1)
            chunk_cmc, chunk_AP, chunk_valid_q = _eval_sorted_indices(
                indices,
                q_pids[start:end],
                g_pids,
                q_camids[start:end],
                g_camids,
                max_rank,
            )
            all_cmc.extend(chunk_cmc)
            all_AP.extend(chunk_AP)
            num_valid_q += chunk_valid_q

            del dist_chunk, indices

    return _finalize_cmc_map(all_cmc, all_AP, num_valid_q)


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.feats0 = []
        self.feats1 = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, feat0, feat1, pid, camid = output
        self.feats.append(feat.cpu())
        self.feats0.append(feat0.cpu())
        self.feats1.append(feat1.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        feats0 = torch.cat(self.feats0, dim=0)
        feats1 = torch.cat(self.feats1, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
            feats0 = torch.nn.functional.normalize(feats0, dim=1, p=2)  # along channel
            feats1 = torch.nn.functional.normalize(feats1, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        qf0 = feats0[:self.num_query]
        qf1 = feats1[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        gf0 = feats0[self.num_query:]
        gf1 = feats1[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
            distmat01 = org_cosine_similarity(qf0, gf0)
            distmat02 = org_cosine_similarity(qf1, gf1)
            cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
            cmc01, mAP01 = eval_func(distmat01, q_pids, g_pids, q_camids, g_camids)
            cmc02, mAP02 = eval_func(distmat02, q_pids, g_pids, q_camids, g_camids)
            cmc03, mAP03 = eval_func(distmat01 + distmat02, q_pids, g_pids, q_camids, g_camids)

        else:
            query_chunk_size = _query_chunk_size()
            print('=> Computing metrics with chunked evaluation, query chunk size={}'.format(query_chunk_size))
            cmc, mAP = eval_func_chunked(
                qf, gf, q_pids, g_pids, q_camids, g_camids,
                max_rank=self.max_rank,
                metric='euclidean',
                query_chunk_size=query_chunk_size,
            )
            cmc01, mAP01 = eval_func_chunked(
                qf0, gf0, q_pids, g_pids, q_camids, g_camids,
                max_rank=self.max_rank,
                metric='cosine',
                query_chunk_size=query_chunk_size,
            )
            cmc02, mAP02 = eval_func_chunked(
                qf1, gf1, q_pids, g_pids, q_camids, g_camids,
                max_rank=self.max_rank,
                metric='cosine',
                query_chunk_size=query_chunk_size,
            )
            cmc03, mAP03 = eval_func_chunked(
                qf0, gf0, q_pids, g_pids, q_camids, g_camids,
                max_rank=self.max_rank,
                metric='cosine_sum',
                query_chunk_size=query_chunk_size,
                qf2=qf1,
                gf2=gf1,
            )
        return cmc, mAP, cmc01, mAP01, cmc02, mAP02, cmc03, mAP03

