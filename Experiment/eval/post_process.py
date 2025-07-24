import re
import textwrap
import argparse
from tree_sitter import Language, Parser
import jsonlines
import json
import tree_sitter_python as tspython

parsers = {}

for lang in ['Python', 'Java']:
    LANGUAGE = Language(tspython.language())
    parser = Parser()
    parser.set_language(LANGUAGE)
    parsers[lang] = parser

def extract_code_from_response(completion, lang, args):
    lines = completion.split("\n")
    codes = []

    start = None
    end = None
    if args.parsed:
        return [completion]
    for i, line in enumerate(lines):
        line = line.strip()
        if line.lower().startswith("```"+lang.lower()) or line.lower().startswith("<code>") or "<code>" in line:
            start = i + 1
        elif line.startswith("```") or '</code>' in line:
            if start is not None:
                end = i
                codes.append('\n'.join(lines[start:end]))
                start = None
                end = None
    if codes:
        return codes
    
    start = None
    end = None
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("```") or line.startswith("<code>") or '</code>' in line:
            if start is None:
                start = i + 1
            else:
                end = i
                codes.append('\n'.join(lines[start:end]))
                start = None
                end = None
    if codes:
        return codes
    
    if start is None:
        start = 0
        end = len(lines)
    elif end is None:
        end = len(lines)
    code_lines = lines[start:end]
    code = "\n".join(code_lines)
    return [code]

def search_function(code, function_name, lang, verbose = False):
    if verbose:
        print(f'pase with lang {lang}\n\n the input code is {code} \n\n the function name is {function_name}')
    parser = parsers[lang]
    tree = parser.parse(bytes(code, "utf8"))
    root = tree.root_node
    target_type = "function_definition" if lang == "Python" else "method_declaration"

    def search_function(node):
        for child in node.children:
            if child.type == target_type:
                node_name = child.child_by_field_name("name")
                if node_name and node_name.text.decode("utf8") == function_name:
                    return child
            result = search_function(child)
            if result:
                return result
        return None
    
    function_node = search_function(root)
    #print('function node is ', function_node)
    if function_node:
        return remove_comments(function_node.text.decode("utf8"), lang)
    else:
        return ""

def remove_comments(code, language):
    if language.lower() == 'python':
        code = re.sub(r'#.*', '', code)
        code = re.sub(r'\'\'\'(.*?)\'\'\'', '', code, flags=re.DOTALL)
        code = re.sub(r'\"\"\"(.*?)\"\"\"', '', code, flags=re.DOTALL)
    elif language.lower() == 'java':
        code = re.sub(r'//.*', '', code)
        code = re.sub(r'/\*(.*?)\*/', '', code, flags=re.DOTALL)
    code_lines = [line for line in code.split("\n") if line.strip() != ""]
    code = "\n".join(code_lines)
    return unify_indentation(code)


def unify_indentation(code):
    return textwrap.dedent(code)

def load_jsonl(file):
    records = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip any empty lines
                data = json.loads(line)
                records.append(data)
    return records
def post_process(path, work, lang, args):
    parts = path.split('/')
    result_path = "/".join(parts[:-1] + ["post_" + parts[-1]])
    processed_path = "/".join(parts[:-1] + ["processed_" + parts[-1]])
    # result_path = path.split('/')
    # result_path[-1] = "post_" + result_path[-1]
    result = []
    # result_path = "/".join(result_path)
    new_results = []
    ori_data = load_jsonl('./data.jsonl')
    with jsonlines.open(path, "r") as f:
        if work == 'generation':
            for obj in f:
                sig = obj["function_signature"]
                namespace = obj["version"]
                prompt_result = obj["prompt"]
                completions = obj["completions"]
                if completions == None:
                    continue
                if not isinstance(completions, list):
                    completions = [completions]
                function_name = obj["focal_name"]
                reference_result = remove_comments(obj["solution"], lang)
                completions_result = []
                for completion in completions:
                    #print('completions is ', completion)
                    if completion == None:
                        continue
                    codes = extract_code_from_response(completion, lang, args)
                    assert codes != []
                    # function_code = None
                    for code in codes:
                        # fistline = code.split("\n")[0]
                        # if fistline != sig:
                        #     print(f"focal name:{function_name}\nsignature:{sig}\ngenerated:{fistline}\n\n***************************************")
                        # temp_code = search_function(code, function_name, lang)
                        # if function_code is None or len(temp_code.split()) > len(function_code.split()):
                        #     function_code = temp_code
                        function_code = search_function(code, function_name, lang)
                        if function_code == "":
                            function_code = code
                        if function_code not in completions_result:
                            completions_result.append(function_code)
                if completions_result == []:
                    continue
                result.append({
                    "namespace": namespace,
                    "prompt": prompt_result,
                    "reference": reference_result,
                    "completions": completions_result
                })
                for d in ori_data:
                    if d['namespace'] == namespace:
                        #print('result 0 is ',completions_result[0])
                        completion = '\n'.join(completions_result[0].split('\n')[1:])
                        #print('completion is ', completion)
                        d['completion'] = completion
                        new_results.append(d)
            with jsonlines.open(processed_path, "w") as f:
                for obj in new_results:
                    f.write(obj)
        elif work == 'completion':
            for obj in f:
                namespace = obj["namespace"]
                prompt_result = obj["prompt"]
                completions = obj["completions"]
                reference_result = remove_comments(obj["reference"], lang)
                code_snippet = re.findall(r'### Code snippet:\n```'+lang+r'\n(.*?)\n```', prompt_result, re.DOTALL)[0]
                code_snippet = remove_comments(code_snippet, lang)
                completions_result = []
                for completion in completions:
                    #print('completions is ', completion)
                    codes = extract_code_from_response(completion, lang, args)
                    # completion_result = None
                    for code in codes:
                        # temp_code = code.replace(code_snippet, "")
                        # if completion_result is None or len(temp_code.split()) > len(completion_result.split()):
                        #     completion_result = temp_code
                        function_code = search_function(code, obj['name'], lang)
                        if function_code != "":
                            flag = False
                            for i in range(len(code_snippet.split("\n"))):
                                temp_snippet = "\n".join(code_snippet.split("\n")[-(i+1):])
                                if function_code.count(temp_snippet) == 1:
                                    function_code = function_code.split(temp_snippet)[1]
                                    flag = True
                                    break
                            if not flag:
                                line_count_code_snippet = len(code_snippet.split("\n"))
                                last_line_code_snippet = code_snippet.split("\n")[-1]
                                if len(function_code.split("\n")) > line_count_code_snippet:
                                    if function_code.split("\n")[line_count_code_snippet] == last_line_code_snippet:
                                        function_code = "\n".join(function_code.split("\n")[line_count_code_snippet+1:])
                                    elif function_code.split("\n")[line_count_code_snippet-1] == last_line_code_snippet:
                                        function_code = "\n".join(function_code.split("\n")[line_count_code_snippet:])
                                    elif function_code.split("\n")[line_count_code_snippet-2] == last_line_code_snippet:
                                        function_code = "\n".join(function_code.split("\n")[line_count_code_snippet-1:])
                        else:
                            function_code = remove_comments(code, lang)
                        if function_code not in completions_result:
                            completions_result.append(function_code)
                result.append({
                    "namespace": namespace,
                    "prompt": prompt_result,
                    "reference": reference_result,
                    "completions": completions_result
                })
        else:
            raise ValueError("Invalid work type")
    with jsonlines.open(result_path, "w") as f:
        for obj in result:
            f.write(obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='data path')
    parser.add_argument('--lang', type=str, default="Python",
                        choices=['Java', 'js', 'c_sharp', 'php', 'go', 'Python', 'ruby'],
                        help='programming language')
    parser.add_argument('--parsed', action='store_true')
    parser.add_argument('--work', type=str, default='generation', choices=['generation', 'completion'],
                        help='work type')

    args = parser.parse_args()
    work = 'generation'
    post_process(args.data_path, args.work, args.lang, args)
