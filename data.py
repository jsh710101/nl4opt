from random import random
import json
import numpy as np


def get_dataset(split, tokenizer):
    TAG_TO_ID = {'NONE': 0, 'VAR1': 1, 'VAR2': 2, 'VAR3': 3, 'VAR4': 4, 'PARAM': 5, 'CONST_DIR': 6, 'LIMIT': 7, 'OBJ_DIR': 8, 'OBJ_NAME': 9}
    ID_TO_VAR = {0: 'x', 1: 'y', 2: 'z', 3: 'w'}
    OP_TO_SYMBOL = {'GREATER_OR_EQUAL': '≥', 'LESS_OR_EQUAL': '≤'}

    with open(f'{split}.jsonl', 'r') as file:
        lines = file.readlines()

    dataset = [list(json.loads(line).values())[0] for line in lines]

    encodings = []
    for data in dataset:
        encoding = tokenizer(data['document'], return_offsets_mapping=True)

        for span in data['spans']:
            if 'text' not in span:
                span['text'] = data['document'][span['start']:span['end']]

        spans = sorted([((span['start'], span['end']), span['label'], span['text']) for span in data['spans']])
        spans.append(((10000, 10000), None, None))

        encoding['token_type_ids'] = []
        for token_range in encoding['offset_mapping']:
            tag_range, tag, tag_text = spans[0]

            while token_range[0] >= tag_range[1]:
                del spans[0]
                tag_range, tag, tag_text = spans[0]

            if token_range[1] <= tag_range[0]:
                encoding['token_type_ids'].append(TAG_TO_ID['NONE'])
            elif token_range[0] < tag_range[1]:
                if tag == 'VAR':
                    tag += str(data['order_mapping'][tag_text] + 1)
                encoding['token_type_ids'].append(TAG_TO_ID[tag])
            else:
                print('ERROR')
        del encoding['offset_mapping']

        labels = ''

        if data['obj_declaration']['type'] == 'objvar':
            variables = [data['order_mapping'][var] for var in data['obj_declaration']['vars']]
            variables = [f'{ID_TO_VAR[var]}' for var in sorted(variables)]
            variables = ' + '.join(variables)
            labels += f' {variables} ;'
        elif data['obj_declaration']['type'] == 'objective':
            variables = [(data['order_mapping'][var], param) for var, param in data['obj_declaration']['terms'].items()]
            variables = [f'{param} {ID_TO_VAR[var]}' for var, param in sorted(variables)]
            variables = ' + '.join(variables)
            labels += f' {variables} ;'
        else:
            print('ERROR')

        for const in data['const_declarations']:
            if const['type'] in ['lowerbound', 'upperbound'] and '%' in const['limit']:
                const['type'] = 'ratio'
            elif const['type'] == 'ratio' and const['limit'].isdecimal():
                if int(const['limit']) > 100:
                    const['type'] = 'lowerbound' if const['operator'] == 'GREATER_OR_EQUAL' else 'upperbound'
                else:
                    const['limit'] += '%'
        
        if split == 'train':
            for old_const_dir in ['must not', 'can not', 'cannot']:
                consts = [const for const in data['const_declarations'] if old_const_dir in const['direction']]

                if len(consts) > 0 and random() < 0.3:
                    old_const_dir = tokenizer.encode(f' {old_const_dir}', add_special_tokens=False)
                    new_const_dir = tokenizer.encode(' must', add_special_tokens=False, padding='max_length', max_length=len(old_const_dir))

                    for i in range(len(encoding['input_ids'])):
                        if encoding['input_ids'][i: i + len(old_const_dir)] == old_const_dir:
                            encoding['input_ids'][i: i + len(old_const_dir)] = new_const_dir

                    for const in consts:
                        const['operator'] = 'LESS_OR_EQUAL' if const['operator'] == 'GREATER_OR_EQUAL' else 'GREATER_OR_EQUAL'

            while tokenizer.pad_token_id in encoding['input_ids']:
                i = encoding['input_ids'].index(tokenizer.pad_token_id)
                del encoding['input_ids'][i], encoding['attention_mask'][i], encoding['token_type_ids'][i]

        for const_type in ['lowerbound', 'upperbound', 'xy', 'xby', 'sum', 'linear', 'ratio']:
            consts = [const for const in data['const_declarations'] if const['type'] == const_type]

            if const_type in ['lowerbound', 'upperbound']:
                consts = [(data['order_mapping'][const['var']], OP_TO_SYMBOL[const['operator']], const['limit']) for const in consts]
                consts = [f' {ID_TO_VAR[var]} {op} {limit} ;' for var, op, limit in sorted(consts)]
                labels += ''.join(consts)

            elif const_type == 'xy':
                consts = [(data['order_mapping'][const['x_var']], OP_TO_SYMBOL[const['operator']], data['order_mapping'][const['y_var']]) for const in consts]
                consts = [f' {ID_TO_VAR[var1]} {op} {ID_TO_VAR[var2]} ;' for var1, op, var2 in sorted(consts)]
                labels += ''.join(consts)

            elif const_type == 'xby':
                consts = [(data['order_mapping'][const['x_var']], OP_TO_SYMBOL[const['operator']], const['param'], data['order_mapping'][const['y_var']]) for const in consts]
                consts = [f' {ID_TO_VAR[var1]} {op} {param} {ID_TO_VAR[var2]} ;' for var1, op, param, var2 in sorted(consts)]
                labels += ''.join(consts)

            elif const_type == 'sum':
                consts = [(OP_TO_SYMBOL[const['operator']], const['limit']) for const in consts]
                consts = [f' sum {op} {limit} ;' for op, limit in sorted(consts)]
                labels += ''.join(consts)

            elif const_type == 'linear':
                consts = [(sorted([(data['order_mapping'][var], param) for var, param in const['terms'].items()]), OP_TO_SYMBOL[const['operator']], const['limit']) for const in consts]

                def check_duplicates(l):
                    return len(l) != len(set(l))

                def get_limit_start(limit):
                    limit_dict = {span['text']: span['start'] for span in data['spans'] if span['label'] == 'LIMIT'}
                    param_dict = {span['text']: span['start'] for span in data['spans'] if span['label'] == 'PARAM'}
                    token_dict = {token['text']: token['start'] for token in data['tokens']}

                    if limit in limit_dict:
                        return limit_dict[limit]
                    elif limit in param_dict:
                        return param_dict[limit]
                    else:
                        return token_dict.get(limit, 10000)

                def get_param_start(param):
                    param_dict = {span['text']: span['start'] for span in data['spans'] if span['label'] == 'PARAM'}
                    token_dict = {token['text']: token['start'] for token in data['tokens']}

                    if param in param_dict:
                        return param_dict[param]
                    else:
                        return token_dict.get(param, 10000)

                if not check_duplicates([const[2] for const in consts]):
                    consts = sorted(consts, key=lambda const: get_limit_start(const[2]))
                else:
                    vars_set = {tuple(var for var, param in const[0]) for const in consts}
                    if len(vars_set) == 1:
                        for i in range(len(vars_set.pop())):
                            if not check_duplicates([const[0][i][1] for const in consts]):
                                consts = sorted(consts, key=lambda const: get_param_start(const[0][i][1]))
                                break

                def vars_to_str(variables):
                    variables = [f'{param} {ID_TO_VAR[var]}' for var, param in variables]
                    return ' + '.join(variables)

                consts = [f' {vars_to_str(variables)} {op} {limit} ;' for variables, op, limit in consts]
                labels += ''.join(consts)

            elif const_type == 'ratio':
                consts = [(data['order_mapping'][const['var']], OP_TO_SYMBOL[const['operator']], const['limit']) for const in consts]
                consts = [f' {ID_TO_VAR[var]} {op} {limit} sum ;' for var, op, limit in sorted(consts)]
                labels += ''.join(consts)

            else:
                print('ERROR')

        encoding['labels'] = tokenizer.encode(labels)
        encodings.append(encoding)

    return encodings


def labels_to_canonical(labels):
    WORD_TO_VAL = {
        'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
        'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19', 'twenty': '20',
        'thirty': '30', 'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70', 'eighty': '80', 'ninety': '90', 'hundred': '100', 'thousand': '1000', 'hundred thousand': '100000',
        'twice': '2', 'thrice': '3', 'half': '0.5', 'third': '0.33333333', 'quarter': '0.25', 'fourth': '0.25', 'fifth': '0.2', 'tenth': '0.1'}
    VAR_TO_ID = {'x': 0, 'y': 1, 'z': 2, 'w': 3}

    def val_to_num(value):
        value = value.replace(' percent', '%')
        percent = '%' in value

        for char in ['%', ',', '$.', ' times', ' minutes', 'a ', 'one ']:
            value = value.replace(char, '')
        if value in WORD_TO_VAL:
            value = WORD_TO_VAL[value]

        try:
            value = float(value)
        except ValueError:
            value = 0.0
        return round(value * 0.01, 8) if percent else value

    if ';' not in labels:
        labels += ';'

    obj, *consts = labels.split(';')[:-1]

    if '≥' in obj or '≤' in obj:
        consts = [obj] + consts
        variables = [var for var in VAR_TO_ID if f' {var} ' in ''.join(consts)]
        obj = ' + '.join(variables)

    obj_terms = np.zeros(len(VAR_TO_ID))
    try:
        for term in obj.split('+'):
            term = term.strip()
            if term != '':
                var, param = term[-1], term[:-2]
                obj_terms[VAR_TO_ID[var]] += val_to_num(param) if param != '' else 1
    except:
        pass

    const_terms = np.zeros((len(consts), len(VAR_TO_ID) + 1))
    for i, const in enumerate(consts):
        try:
            lowerbound = '≥' in const
            left, right = const.split('≥' if lowerbound else '≤')

            if 'sum' in left:
                const_terms[i, :len(VAR_TO_ID)] += (obj_terms != 0.0).astype(float)
            else:
                for term in left.split('+'):
                    term = term.strip()
                    if term != '':
                        var, param = term[-1], term[:-2]
                        const_terms[i, VAR_TO_ID[var]] += val_to_num(param) if param != '' else 1

            if 'sum' in right:
                ratio = val_to_num(right.replace('sum', '').strip())
                const_terms[i, :len(VAR_TO_ID)] -= ratio * (obj_terms != 0.0).astype(float)
            else:
                for term in right.split('+'):
                    term = term.strip()
                    if term[-2:].strip() in VAR_TO_ID:
                        var, param = term[-1], term[:-2]
                        const_terms[i, VAR_TO_ID[var]] -= val_to_num(param) if param != '' else 1
                    else:
                        const_terms[i, -1] += val_to_num(term)

            if lowerbound:
                const_terms[i] *= -1.0
        except:
            pass

    return obj_terms, const_terms
