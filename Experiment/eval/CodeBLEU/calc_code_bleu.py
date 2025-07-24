import argparse
import os
import re
import bleu
import weighted_ngram_match
import syntax_match
import dataflow_match
import jsonlines
from nltk.tokenize import word_tokenize
import Levenshtein
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def exact_match(reference, generated):
    return reference == generated

def edit_distance(reference, generated):
    # 创建一个矩阵来存储编辑距离
    dp = [[0] * (len(generated) + 1) for _ in range(len(reference) + 1)]

    # 初始化边界条件
    for i in range(len(reference) + 1):
        dp[i][0] = i
    for j in range(len(generated) + 1):
        dp[0][j] = j

    # 计算编辑距离
    for i in range(1, len(reference) + 1):
        for j in range(1, len(generated) + 1):
            if reference[i - 1] == generated[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[len(reference)][len(generated)]

def _edit_similarity(reference, generated):
    max_length = max(len(reference), len(generated))
    if max_length == 0:
        print(f"refence: {reference}\ngenerated: {generated}")
        return 0
    # distance = edit_distance(reference, generated)
    distance = Levenshtein.distance(reference, generated)
    similarity = 1 - (distance / max_length)
    return similarity

def tokenize_code(code):
    code = re.sub(r'([.,!?(){}[\]])', r' \1 ', code)
    code = re.sub(r'\s+', ' ', code)
    tokens = word_tokenize(code)
    return tokens

def evaluate_completion(data_path, lang, output_dir = '../result'):
    pre_references = []
    hypothesis = []
    with jsonlines.open(data_path, "r") as f:
        for obj in f:
            reference = obj["reference"]
            completions = obj["completions"]
            pre_references.append(reference.strip())
            hypothesis.append([completion.strip() for completion in completions])
    
    references = []
    for i in range(len(hypothesis)):
        references.append([pre_references[i]] * len(hypothesis[i]))

    tokenized_hyps = [[tokenize_code(x) for x in hypos] for hypos in hypothesis]
    tokenized_refs = [[tokenize_code(x) for x in refs] for refs in references]

    # Calculate exact match score
    exact_match_scores = []
    for i in range(len(references)):
        max_exact_match_score = 0
        for j in range(len(references[i])):
            exact_match_score = exact_match(references[i][j], hypothesis[i][j])
            max_exact_match_score = max(max_exact_match_score, exact_match_score)
        exact_match_scores.append(max_exact_match_score)
    exact_match_score = sum(exact_match_scores) / len(exact_match_scores)

    # Calculate Edit Similarity
    edit_similaritys = []
    for i in range(len(references)):
        max_edit_similarity = 0
        for j in range(len(references[i])):
            edit_similarity_ = _edit_similarity(references[i][j], hypothesis[i][j])
            max_edit_similarity = max(max_edit_similarity, edit_similarity_)
        edit_similaritys.append(max_edit_similarity)
    edit_similarity = sum(edit_similaritys) / len(edit_similaritys)

    # Calculate ngram match score
    ngram_match_scores = []
    for i in range(len(tokenized_refs)):
        max_ngram_match_score = 0
        for j in range(len(tokenized_refs[i])):
            ngram_match_score = bleu.corpus_bleu([[tokenized_refs[i][j]]], [tokenized_hyps[i][j]])
            max_ngram_match_score = max(max_ngram_match_score, ngram_match_score)
        ngram_match_scores.append(max_ngram_match_score)
    ngram_match_score = sum(ngram_match_scores) / len(ngram_match_scores)

    # Calculate weighted ngram match score
    keywords = [x.strip() for x in open('CodeBLEU/keywords/' + lang + '.txt', 'r', encoding='utf-8').readlines()]

    def make_weights(reference_tokens, key_word_list):
        return {token: 1 if token in key_word_list else 0.2 for token in reference_tokens}

    weighted_ngram_match_scores = []
    for i in range(len(tokenized_refs)):
        max_weighted_ngram_match_score = 0
        for j in range(len(tokenized_refs[i])):
            weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(
                [[[tokenized_refs[i][j], make_weights(tokenized_refs[i][j], keywords)]]], [tokenized_hyps[i][j]]
            )
            max_weighted_ngram_match_score = max(max_weighted_ngram_match_score, weighted_ngram_match_score)
        weighted_ngram_match_scores.append(max_weighted_ngram_match_score)
    weighted_ngram_match_score = sum(weighted_ngram_match_scores) / len(weighted_ngram_match_scores)

    # reference_flat = [' '.join(x) for refs in references for x in refs]
    # generated_flat = [' '.join(x) for hypos in references for x in hypos]

    # accuracy = accuracy_score(reference_flat, generated_flat)
    # precision = precision_score(reference_flat, generated_flat, average='weighted')
    # recall = recall_score(reference_flat, generated_flat, average='weighted')
    # f1 = f1_score(reference_flat, generated_flat, average='weighted')


    output_path = os.path.join(output_dir, 'Sub_Code_completion.csv')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_path):
        df = pd.DataFrame(columns=[
            'model', 'task', 'language',
            'edit_similarity', 'ngram_match', 'weighted_ngram_match', 'Average',
            # 'accuracy', 'precision', 'recall', 'f1',
            'exact_match'
        ])
    else:
        df = pd.read_csv(output_path)
    model = data_path.split('/')[-4]
    task = data_path.split('/')[-2]

    # Append results to dataframe(使用round函数将结果乘以100并保留两位小数)
    new_row = [{
        'model': model, 'task': task, 'language': lang,
        'edit_similarity': round(edit_similarity * 100, 2),
        'ngram_match': round(ngram_match_score * 100, 2),
        'weighted_ngram_match': round(weighted_ngram_match_score * 100, 2),
        'Average': round((edit_similarity + ngram_match_score + weighted_ngram_match_score) * 100 / 3, 2),
        # 'accuracy': round(accuracy * 100, 2),
        # 'precision': round(precision * 100, 2),
        # 'recall': round(recall * 100, 2),
        # 'f1': round(f1 * 100, 2)
        'exact_match': round(exact_match_score * 100, 2),
    }]
    new_df = pd.DataFrame(new_row)
    df = pd.concat([df, new_df], ignore_index=True)

    df.to_csv(output_path, index=False)

    # Print results(乘以100并保留两位小数)
    print("Edit Similarity: ", round(edit_similarity * 100, 2))
    print("Ngram Match Score: ", round(ngram_match_score * 100, 2))
    print("Weighted Ngram Match Score: ", round(weighted_ngram_match_score * 100, 2))
    print("Average: ", round((edit_similarity + ngram_match_score + weighted_ngram_match_score) * 100 / 3, 2))
    # print("Accuracy: ", round(accuracy * 100, 2))
    # print("Precision: ", round(precision * 100, 2))
    # print("Recall: ", round(recall * 100, 2))
    # print("F1: ", round(f1 * 100, 2))
    print("Exact Match: ", round(exact_match_score * 100, 2))


def evaluate_generation(args):
    lang = args.lang
    alpha, beta, gamma, theta = [float(x) for x in args.params.split(',')]

    pre_references = []
    hypothesis = []
    with jsonlines.open(args.data_path, "r") as f:
        for obj in f:
            reference = obj["reference"]
            completions = obj["completions"]
            pre_references.append(reference.strip())
            hypothesis.append([completion.strip() for completion in completions])

    references = []
    for i in range(len(hypothesis)):
        references.append([pre_references[i]] * len(hypothesis[i]))

    tokenized_hyps = [[tokenize_code(x) for x in hypos] for hypos in hypothesis]
    tokenized_refs = [[tokenize_code(x) for x in refs] for refs in references]

    # Calculate exact match score
    exact_match_scores = []
    for i in range(len(references)):
        max_exact_match_score = 0
        for j in range(len(references[i])):
            exact_match_score = exact_match(references[i][j], hypothesis[i][j])
            max_exact_match_score = max(max_exact_match_score, exact_match_score)
        exact_match_scores.append(max_exact_match_score)
    exact_match_score = sum(exact_match_scores) / len(exact_match_scores)

    # Calculate Edit Similarity
    edit_similaritys = []
    for i in range(len(references)):
        max_edit_similarity = 0
        for j in range(len(references[i])):
            edit_similarity_ = _edit_similarity(references[i][j], hypothesis[i][j])
            max_edit_similarity = max(max_edit_similarity, edit_similarity_)
        edit_similaritys.append(max_edit_similarity)
    edit_similarity = sum(edit_similaritys) / len(edit_similaritys)

    # Calculate ngram match score
    ngram_match_scores = []
    for i in range(len(tokenized_refs)):
        max_ngram_match_score = 0
        for j in range(len(tokenized_refs[i])):
            ngram_match_score = bleu.corpus_bleu([[tokenized_refs[i][j]]], [tokenized_hyps[i][j]])
            max_ngram_match_score = max(max_ngram_match_score, ngram_match_score)
        ngram_match_scores.append(max_ngram_match_score)
    ngram_match_score = sum(ngram_match_scores) / len(ngram_match_scores)

    # Calculate weighted ngram match score
    keywords = [x.strip() for x in open('eval/CodeBLEU/keywords/' + args.lang + '.txt', 'r', encoding='utf-8').readlines()]

    def make_weights(reference_tokens, key_word_list):
        return {token: 1 if token in key_word_list else 0.2 for token in reference_tokens}

    weighted_ngram_match_scores = []
    for i in range(len(tokenized_refs)):
        max_weighted_ngram_match_score = 0
        for j in range(len(tokenized_refs[i])):
            weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(
                [[[tokenized_refs[i][j], make_weights(tokenized_refs[i][j], keywords)]]], [tokenized_hyps[i][j]]
            )
            max_weighted_ngram_match_score = max(max_weighted_ngram_match_score, weighted_ngram_match_score)
        weighted_ngram_match_scores.append(max_weighted_ngram_match_score)
    weighted_ngram_match_score = sum(weighted_ngram_match_scores) / len(weighted_ngram_match_scores)

    # Calculate syntax match score
    syntax_match_scores = []
    for i in range(len(references)):
        max_syntax_match_score = 0
        for j in range(len(references[i])):
            syntax_match_score = syntax_match.corpus_syntax_match([[references[i][j]]], [hypothesis[i][j]], args.lang)
            max_syntax_match_score = max(max_syntax_match_score, syntax_match_score)
        syntax_match_scores.append(max_syntax_match_score)
    syntax_match_score = sum(syntax_match_scores) / len(syntax_match_scores)

    # Calculate dataflow match score
    dataflow_match_scores = []
    for i in range(len(references)):
        max_dataflow_match_score = -1
        for j in range(len(references[i])):
            dataflow_match_score = dataflow_match.corpus_dataflow_match([[references[i][j]]], [hypothesis[i][j]], args.lang)
            max_dataflow_match_score = max(max_dataflow_match_score, dataflow_match_score)
        if max_dataflow_match_score == -1:
            continue
        dataflow_match_scores.append(max_dataflow_match_score)
    dataflow_match_score = sum(dataflow_match_scores) / len(dataflow_match_scores)
    print("len(dataflow_match_scores): ", len(dataflow_match_scores))

    # Calculate CodeBLEU score
    code_bleu_score = alpha * ngram_match_score + beta * weighted_ngram_match_score + gamma * syntax_match_score + theta * dataflow_match_score

    path = args.data_path
    output_path = os.path.join(args.output_dir, "Code_generation.csv")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(output_path):
        df = pd.DataFrame(columns=[
            'model', 'task', 'language',
            'Ngram Match Score', 'Weighted Ngram Match Score', 'Syntax Match Score', 'Dataflow Match Score',
            'CodeBLEU Score', 'Edit Similarity', 'Average', 'Exact Match'
        ])
    else:
        df = pd.read_csv(output_path)
    
    model = path.split('/')[-4]
    task = path.split('/')[-2]

    # Append results to dataframe(使用round函数将结果乘以100并保留两位小数)
    new_rows = []
    new_rows.append({
        'model': model,
        'task': task,
        'language': lang,
        'Ngram Match Score': round(ngram_match_score * 100, 2),
        'Weighted Ngram Match Score': round(weighted_ngram_match_score * 100, 2),
        'Syntax Match Score': round(syntax_match_score * 100, 2),
        'Dataflow Match Score': round(dataflow_match_score * 100, 2),
        'CodeBLEU Score': round(code_bleu_score * 100, 2),
        'Edit Similarity': round(edit_similarity * 100, 2),
        'Average': round((edit_similarity + ngram_match_score + weighted_ngram_match_score + syntax_match_score + dataflow_match_score) * 100 / 5, 2),
        'Exact Match': round(exact_match_score * 100, 2)
    })
    new_df = pd.DataFrame(new_rows)
    df = pd.concat([df, new_df], ignore_index=True)

    # Save results to csv file
    df.to_csv(output_path, index=False)

    # Print results(乘以100并保留两位小数)
    print("Edit Similarity: {:.2f}%".format(edit_similarity * 100))
    print("Ngram Match Score: {:.2f}%".format(ngram_match_score * 100))
    print("Weighted Ngram Match Score: {:.2f}%".format(weighted_ngram_match_score * 100))
    print("Syntax Match Score: {:.2f}%".format(syntax_match_score * 100))
    print("Dataflow Match Score: {:.2f}%".format(dataflow_match_score * 100))
    print("CodeBLEU Score: {:.2f}%".format(code_bleu_score * 100))
    print('Average: {:.2f}%'.format((edit_similarity + ngram_match_score + weighted_ngram_match_score + syntax_match_score + dataflow_match_score) * 100 / 5))
    print("Exact Match: {:.2f}%".format(exact_match_score * 100))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='data path')
    parser.add_argument('--lang', type=str, required=True, 
                        choices=['java','js','c_sharp','php','go','python','ruby'],
                        help='programming language')
    parser.add_argument('--work', type=str, required=True, choices=['generation', 'completion'],
                        help='work type')
    parser.add_argument('--params', type=str, default='0.25,0.25,0.25,0.25',
                        help='alpha, beta and gamma')
    parser.add_argument('--output_dir', type=str, default='./results', help='output directory')

    args = parser.parse_args()

    if args.work == 'generation':
        evaluate_generation(args)
    elif args.work == 'completion':
        evaluate_completion(args.data_path, args.lang, args.output_dir)
    else:
        raise ValueError("Invalid work type")


if __name__ == "__main__":
    main()
