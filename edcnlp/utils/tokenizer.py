import itertools

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def character_offset(sentence, tokens):
    # Given a sentence in string and tokenized version
    # find character offset for each token
    start = 0
    res = []
    for token in tokens:
        begin_index = sentence.find(token, start)
        end_index = begin_index + len(token)
        start = end_index

        assert token == sentence[begin_index:end_index]
        res.append({'word': token, 'characterOffsetBegin': begin_index, 'characterOffsetEnd': end_index})
    return res

def whitespace_tokenizer(sentence):
    tokens = whitespace_tokenize(sentence)
    tokens_with_offset = character_offset(sentence, tokens)
    return tokens, tokens_with_offset

def convert_tokens_misc(tokens):
    # convert stanza tokens character misc to character offset begin and end
    for t in tokens:
        start = t['misc'].split('|')[0]
        end = t['misc'].split('|')[1]
        start = start.split('=')[-1]
        end = end.split('=')[-1]
        t['characterOffsetBegin'] = int(start)
        t['characterOffsetEnd'] = int(end)
    return tokens


def find_all(a_str, sub):
    # given a string sentence and a string
    # find all matched string in this sentence
    # return all starting position
    start = 0
    candidate = []
    while start <= len(a_str):
        start = a_str.find(sub, start)
        if start == -1:
            break
        candidate.append(start)
        start += len(sub)
    return candidate


def find_phrase_index(sentence, tokens_with_offset, words, string):
    # given a string sentence and a phrase string and tokenized sentence
    # find matched string in sentence
    # find span for all matched string in tokenized sentence

    all_start_char = find_all(sentence, string)
    span_candidate = []
    for start_char in all_start_char:
        end_char = start_char + len(string)
        assert sentence[start_char: end_char] == string

        start_token_idx = -1
        end_token_idx = -1
        exact_start_match_flag = -1
        exact_end_match_flag = -1
        for token_idx, t in enumerate(tokens_with_offset):
            if t['characterOffsetBegin'] == start_char:
                start_token_idx = token_idx
                # exact match start
                exact_start_match_flag = 1
                break
            elif t['characterOffsetBegin'] > start_char:
                start_token_idx = token_idx
                exact_start_match_flag = 0 # [Anti-XXX, is] and string = XXX
                break

        assert start_token_idx != -1, 'words: {}, string: [START]{}[END]'.format(words, string)

        for token_idx, t in enumerate(tokens_with_offset):
            if t['characterOffsetEnd'] == end_char:
                end_token_idx = token_idx + 1
                exact_end_match_flag = 1
                break
            elif t['characterOffsetEnd'] > end_char:
                end_token_idx = token_idx + 1
                exact_end_match_flag = 0
                break

        assert end_token_idx != -1, 'words: {}, string: [START]{}[END]'.format(words, string)
        # join tokens
        characters = ''.join(words[start_token_idx:end_token_idx])
        # remove all ' '
        characters = characters.replace(' ', '')

        # corner case for annotation string failure cases:
        # for example: U.S. embassy, but stirng = S. embassy
        # resolutioin, but string = esolution
        # international standards, but string = nternational standards
        if len(characters) < len(string.replace(' ', '')):
            # missing one previous token
            characters = ''.join(words[start_token_idx-1:end_token_idx])
            characters = characters.replace(' ', '')
            exact_start_match_flag = 0
        # cut length
        if exact_start_match_flag == 1 and exact_end_match_flag == 1:
            # exact match
            pass
        elif exact_start_match_flag == 1 and exact_end_match_flag == 0:
            # [U.S., X] and string is 'U.S'
            characters = characters[:len(string.replace(' ',''))]
        elif exact_start_match_flag == 0 and exact_end_match_flag == 1:
            # [AntiXXX] and string is XXX
            characters = characters[-len(string.replace(' ','')):]
        else:
            # [here] and string is he
            # skip this
            continue

        if characters == string.replace(' ', ''):
            span_candidate.append((start_token_idx, end_token_idx))

    if span_candidate == []:

        print('[ERROR]: no span')
        print('ch: ', characters)
        print('string: ', string)
        print('sentence: ', sentence)
        print('words: ', words)

    return span_candidate

def clean(s):
    for c in '!"#$%&\'()*+-/:;<=>?@[\\]^_`{|}~':
        s = s.replace(c, ' ')
    return s

def get_span_from_multi_spans(dic):
    # dic = {'x': [(1,2),(3,4)], 'y': [(5,6), (7,8)]}
    val = list(dic.values())
    comb = list(itertools.product(*val))
    final_start = -1
    final_end = -1
    min_diff = 999999
    for v in comb:
        start = min(v, key=lambda t: t[0])[0]
        end = max(v, key=lambda t: t[1])[1]

        diff = end - start
        if diff < min_diff:
            final_start = start
            final_end = end
    return final_start, final_end


def get_span_from_list_of_string(w_list, sent, tokens_with_offset, words):
    span_candidate_dic = {}
    for w in w_list:
        span_candidate_w = find_phrase_index(sent, tokens_with_offset, words, w)
        span_candidate_dic[w] = span_candidate_w
    # create combination
    start, end = get_span_from_multi_spans(span_candidate_dic)
    return start, end

def create_label_mask(sent, segment_alignment):
    # For annotation projection
    mask = [0 for _ in range(len(sent.split()))]
    aligned_index = [int(t.split('-')[1]) for t in segment_alignment.split()]
    for idx in aligned_index:
        mask[idx] = 1
    return mask