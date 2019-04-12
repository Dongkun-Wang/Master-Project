import xlrd
import xlwt


def read_case(related_path, sheet_name):
    file = xlrd.open_workbook(related_path)              # 读取元数据文件
    sheet = file.sheet_by_name(sheet_name)               # 读取元数据表格
    dataset = []                                         # 创建空表
    for i in range(sheet.nrows):
        dataset.append(sheet.row_values(i))              # 循环读取数据
    return dataset


def write_syn(related_path, syn):
    workbook = xlwt.Workbook()                           # 创建新文件
    for k in range(len(syn)):
        worksheet = workbook.add_sheet('syn' + str(k))   # 创建新工作表
        for i in range(len(syn[k])):
            for j in range(len(syn[k][0])):
                worksheet.write(i, j, syn[k][i][j])      # 循环写入数据
    workbook.save(related_path)                          # 保存


"""
def read_syn(related_path, sheet_name):
    file = xlrd.open_workbook(related_path)              # 读取权值矩阵文件
    sheet = file.sheet_by_name(sheet_name)               # 读取权值矩表格阵
    dataset = []                                         # 创建空表
    for i in range(sheet.nrows):
        dataset.append(sheet.row_values(i))              # 循环读取数据
    return dataset
"""

