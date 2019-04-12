def sort(result):
    for i in range(len(result)):
        k = 0
        m = 0
        for j in range(len(result[i])):
            if m < result[i][j]:
                m = result[i][j]
                k = j
        for j in range(len(result[i])):
            result[i][j] = 0
        result[i][k] = 1
    return result


def rate(result, label):
    r = 0
    result = result.tolist()
    for i in range(len(result)):
        if result[i] == label[i]:
            r += 1
    return format(r/len(result), '.0%')
