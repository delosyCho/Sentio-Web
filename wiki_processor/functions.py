string = '(asdasdasDA)asdasdsad(Asdasd)[asdasdasdasda]'

def Bracket_Processing(string, start, end):
    result = ''

    is_bracket = False

    for i in range(len(string)):
        if string[i] == start:
            is_bracket = True
        elif string[i] == end:
            is_bracket = False
        elif is_bracket == False:
            result += string[i]

    return result
#< >


def Bracket_Processing2(string, start, end):
    result = ''

    is_bracket = False

    for i in range(len(string)):
        if string[i] == start:
            is_bracket = True
        elif string[i] == end:
            is_bracket = False
            result += string[i]
        elif is_bracket == False:
            result += string[i]

    return result


def Bracket_Processing3(string, start, end):
    result = ''

    is_bracket = False

    for i in range(len(string)):
        if i + 1 < len(string):
            if string[i] == start[0] and string[i + 1] == start[1]:
                is_bracket = True
            elif string[i] == end:
                is_bracket = False
                result += string[i]
            elif is_bracket == False:
                result += string[i]
        else:
            if string[i] == end:
                is_bracket = False
                result += string[i]
            elif is_bracket == False:
                result += string[i]
    return result


def Bracket_Processing4(string, start, end):
    result = ''

    bracket_count = 0
    is_bracket = False

    #print(len(string))

    for i in range(len(string)):
        if i + 1 < len(string) and i > 0:
            if string[i] == start[0] and string[i + 1] == start[1]:
                bracket_count += 1
            elif string[i - 1] == end[0] and string[i] == end[1]:
                bracket_count -= 1
            elif bracket_count == 0:
                result += string[i]
        elif i == 0:
            if string[i] == start[0] and string[i + 1] == start[1]:
                bracket_count += 1
        else:
            if bracket_count == 0:
                result += string[i]
        #print(string[i])
        #print(bracket_count)

    return result


def Bracket_Processing4_(string, start, end):
    result = ''

    bracket_count = 0

    for i in range(len(string)):
        if i + 1 < len(string) and i > 0:
            if string[i] == start[0] and string[i + 1] == start[1]:
                if bracket_count == 0:
                    result += string[i]

                bracket_count += 1
            elif string[i - 1] == end[0] and string[i] == end[1]:
                bracket_count -= 1

                if bracket_count == 0:
                    result += string[i - 1]
            elif bracket_count <= 1:
                result += string[i]
        elif i == 0:
            if string[i] == start[0] and string[i + 1] == start[1]:
                bracket_count += 1
        else:
            if bracket_count <= 1:
                result += string[i]
        #print(string[i])
        #print(bracket_count)

    return result
# [[  ]]


def Bracket_Processing5(string, start, end, end2):
    result = ''

    bracket_count = 0
    is_bracket = False
    is_bracket_end = False

    #print(len(string))

    for i in range(len(string)):
        if i + 1 < len(string) and i > 0:
            if is_bracket is True and is_bracket_end is True:
                if string[i - 2] == end2[0] and string[i - 1] == end2[1]:
                    is_bracket = False
                    is_bracket_end = False
                else:
                    if string[i] != start[0] and string[i] != end2[0]:
                        result += string[i]
            elif string[i] == start[0] and string[i + 1] == start[1]:
                temp_index = i
                token_exist = False
                while temp_index < len(string) - 2:
                    if string[temp_index] == end[0]:
                        token_exist = True
                    if string[temp_index] == end2[0] and string[temp_index + 1] == end2[1]:
                        if token_exist is True:
                            is_bracket = True
                            is_bracket_end = False
                            temp_index += len(string)
                        else:
                            temp_index += len(string)
                    else:
                        temp_index += 1
            elif string[i] == end[0]:
                is_bracket_end = True
        elif i == 0:
            if string[i] == start[0] and string[i + 1] == start[1]:
                temp_index = i
                token_exist = False
                while temp_index < len(string) - 2:
                    if string[temp_index] == end[0]:
                        token_exist = True
                    if string[temp_index] == end2[0] and string[temp_index + 1] == end2[1]:
                        if token_exist is True:
                            is_bracket = True
                            is_bracket_end = False
                            temp_index += len(string)
                        else:
                            temp_index += len(string)
                    else:
                        temp_index += 1

        if is_bracket is False and is_bracket_end is False:
            result += string[i]

        #print(string[i])
        #print(bracket_count)

    return result
#[[, | 경우를 처리함


def Bracket_Processing6(string, start, end):
    result = ''

    is_bracket = False

    #print(len(string))
    if len(start) > len(end):
        length = len(start)
    else:
        length = len(end)

    for i in range(len(string)):

        if i + length < len(string) and i >= 0:
            is_start = True
            is_end = True

            for j in range(len(start)):
                if string[i + j] != start[j]:
                    is_start = False

            for j in range(len(end)):
                if string[i - (len(end) - 1 - j)] != end[j]:
                    is_end = False

            if is_start is True:
                is_bracket = True
            elif is_end is True:
                is_bracket = False
            elif is_bracket is False:
                result += string[i]
        elif i == 0:
            if string[i] == start[0] and string[i + 1] == start[1]:
                is_bracket = True
            else:
                result += string[i]
        else:
            if is_bracket is False:
                result += string[i]
        #print(string[i])
        #print(bracket_count)

    return result
#<ref> <ref/>


def Bracket_Processing_file(string, start, start2, end):
    result = ''

    is_bracket = False
    bracket_count = 0

    # print(len(string))
    if len(start) > len(end):
        length = len(start)
    else:
        length = len(end)

    for i in range(len(string)):
        if i + length < len(string) and i >= 0:
            is_start = True
            is_start2 = True
            is_end = True

            for j in range(len(start)):
                if string[i + j] != start[j]:
                    is_start = False

            for j in range(len(start2)):
                if string[i + j] != start2[j]:
                    is_start2 = False

            for j in range(len(end)):
                if string[i - (len(end) - 1 - j)] != end[j]:
                    is_end = False

            if is_start is True:
                is_bracket = True
                bracket_count += 1

            if is_start2 is True:
                bracket_count += 1

            if is_end is True:
                bracket_count -= 1
                if bracket_count == 0:
                    is_bracket = False

        if is_bracket is False:
            result += string[i]

            # print(string[i])
            # print(bracket_count)

    return result
#remove file tag

def preprocess(result_):
    result_ = Bracket_Processing(result_, '(', ')')
    # result_ = Bracket_Processing4_(result_, '[[', ']]')
    result_ = Bracket_Processing5(result_, '[[', '|', ']]')
    result_ = Bracket_Processing6(result_, '<ref>', '</ref>')
    result_ = Bracket_Processing_file(result_, '[[파일:', '[[', ']]')
    result_ = Bracket_Processing(result_, '<', '>')

    result_ = result_.replace('〈', '')
    result_ = result_.replace('〉', '')
    result_ = result_.replace('-', ' ')
    result_ = result_.replace('\'\'\'', '')
    result_ = result_.replace('”', '')
    result_ = result_.replace('“', '')
    result_ = result_.replace('《', '')
    result_ = result_.replace('》', '')
    result_ = result_.replace(' ( ', '(')
    result_ = result_.replace(' \" ', '')
    result_ = result_.replace('\"', '')
    result_ = result_.replace('.', '')
    result_ = result_.replace('[[', '')
    result_ = result_.replace(']]', '')
    result_ = result_.replace(',', '')
    result_ = result_.replace('\'\'', '')
    result_ = result_.replace('\'', '')
    result_ = result_.replace('‘', '')
    result_ = result_.replace('’', '')

    return result_


