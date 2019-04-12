import numpy as np
import Analysis
import Import
import BP

print("正在训练神经网络")
# 训练数据输入
TrainSet_x = Import.read_case('DataFile/StatusAndResult.xls', 'Sheet1_x')
TrainSet_y = Import.read_case('DataFile/StatusAndResult.xls', 'Sheet2_y')
# 训练神经网络
nt = BP.training(TrainSet_x, TrainSet_y)
# 训练结果保存
Import.write_syn('DataFile/Syn.xls', nt.weights)
# 测试数据输入
Test_input = np.array(Import.read_case('DataFile/Test.xls', 'Sheet1_x'))
Test_output = Import.read_case('DataFile/Test.xls', 'Sheet2_y')
# 测试数据输出
result = BP.forward(Test_input, nt)[nt.num_layers-1]
# 测试结果整理
result_sort = Analysis.sort(result)
print("\n整理后的测试结果\n", result_sort)
# 正确率计算
rate = Analysis.rate(result, Test_output)
print("\n测试正确率为:\n", rate)

