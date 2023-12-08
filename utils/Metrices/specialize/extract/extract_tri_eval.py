# codes from
# https://github.com/open-event-hub/title2event_baselines

import pandas as pd


def evaluate(df: pd.DataFrame, pred_col='pred_event_triples', ans_col='event_triples', detail: bool = True):
    """
    compute precision, recall, and F1-score for model predictions, based on exact match, and print the tabulated results
    args:
        df: the dataframe containing a column of model predictions and a column of golden answers, where each cell of both columns should be a list of triplets
        pred_col: column name for model predictions
        ans_col: column name for golden answers
    return: F1-scores of trigger extraction, argument extraction and triplet extraction
    """
    gold_trp_num = sum(df[ans_col].apply(lambda x: len(x)))
    pred_trp_num = sum(df[pred_col].apply(lambda x: len(x)))
    gold_trg_num, gold_arg_num, pred_trg_num, pred_arg_num = 0, 0, 0, 0
    trg_match_cnt, arg_match_cnt, triple_match_cnt = 0, 0, 0
    for idx, row in df.iterrows():
        local_triple_match_cnt = 0
        gold_trips = list(row[ans_col])
        gold_trgs = [trp[1] for trp in row[ans_col]]
        gold_sbjs = [{"P": trp[1], "S": trp[0]}
                     for trp in row[ans_col] if trp[0] != '']
        gold_objs = [{"P": trp[1], "O": trp[2]}
                     for trp in row[ans_col] if trp[2] != '']
        gold_trg_num += len(gold_trgs)
        gold_arg_num += (len(gold_sbjs) + len(gold_objs))
        for pred in row[pred_col]:
            pred_trg_num += 1
            if len(pred) == 1:
                pred = ["", pred[0], ""]
            elif len(pred) == 2:
                pred.append("")
            if pred in gold_trips:
                local_triple_match_cnt += 1
                triple_match_cnt += 1
                gold_trips.remove(pred)

            if pred[0] != '':
                pred_arg_num += 1
            if pred[2] != '':
                pred_arg_num += 1
            if pred[1] in gold_trgs:
                trg_match_cnt += 1
                gold_trgs.remove(pred[1])
            if {"P": pred[1], "S": pred[0]} in gold_sbjs:
                arg_match_cnt += 1
                gold_sbjs.remove({"P": pred[1], "S": pred[0]})
            if {"P": pred[1], "O": pred[2]} in gold_objs:
                arg_match_cnt += 1
                gold_objs.remove({"P": pred[1], "O": pred[2]})

    def F1(p, r): return "{:.5f}".format(2*p*r/(p+r+1e-8))
    trg_p, arg_p, trp_p = trg_match_cnt/pred_trg_num, arg_match_cnt / \
        pred_arg_num, triple_match_cnt/pred_trp_num
    trg_r, arg_r, trp_r = trg_match_cnt/gold_trg_num, arg_match_cnt / \
        gold_arg_num, triple_match_cnt/gold_trp_num
    trg_f, arg_f, trp_f = F1(trg_p, trg_r), F1(arg_p, arg_r), F1(trp_p, trp_r)

    header = ["task", "Precision", "Recall", "F1"]
    rows = [
        ("Trigger Identification", trg_p, trg_r, trg_f),
        ("Argument Identification", arg_p, arg_r, arg_f),
        ("Triple Identification", trp_p, trp_r, trp_f),
    ]
    if detail:
        # print(tabulate(rows, headers=header))
        print("gold num > {}".format(
            {"trp": gold_trg_num, "trg": gold_trg_num, "arg": gold_arg_num}))
        print("pred num > {}".format(
            {"trp": pred_trp_num, "trg": pred_trg_num, "arg": pred_arg_num}))
        print("trg match: %d, arg match: %d, trp match: %d" %
              (trg_match_cnt, arg_match_cnt, triple_match_cnt))

    return trg_f, arg_f, trp_f


if __name__ == '__main__':
    a = [[['i', 'fu', 'u']],
         [['u', 'fu', 'm']],
         [['u', 'dd', 'ff']]]
    b = [[['i', 'fu', 'u'], ['ii', 'af', 'af']],
         [['u', 'fu', 'm']],
         [['u', 'dd', 'ff']]]

    cur = pd.DataFrame()
    cur['pred_event_triples'] = a
    cur['event_triples'] = b
    print(cur)
    print(evaluate(cur))
