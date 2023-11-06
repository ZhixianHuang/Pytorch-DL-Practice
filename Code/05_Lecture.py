import torch

'''
四个核心部分
1. Prepare dataset 准备数据集
2. Design model using class 定义模型类(inherit from nn.Module继承自neural Netword父类)
3. Construct loss and optimizer 设计损失函数与优化器 (using PyTorch API 调用预封装接口)
4. Training cycle 训练循环(forward, backward, update 正向传播, 反向传播, 更新梯度)
'''

#  Prepare dataset
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])

# Design model using class
class LinearModel(torch.nn.Module):  # 新定义类继承自父nn.Module类
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # 继承父类的初始化
        self.linear = torch.nn.Linear(1,1)  # 定义一个线性层 是LinearModel的一个成员变量 与self.name = "Cadian"不同，后者是一个简单的属性，前者是一个PyTorch模型层，用于进行线性变化层操作
                                            # (1,1)是指输入x和输出y的特征维度，这里数据集中的x和y的特征都是1维的
                                            # 该线性层需要学习的参数是w和b  获取w/b的方式分别是~linear.weight/linear.bias
        '''
        self.hidden_layer = torch.nn.Linear(3, 5)  # 输入特征维度是3,隐藏层节点数是5
        self.output_layer = torch.nn.Linear(5, 2)  # 隐藏层输出作为输入,输出特征维度是2
        猜测写法
        '''

    def forward(self,x):  # 重写forward函数是因为下方实例化的时候(Line 91)可以直接被调用
                          # 为什么forward方法需要被重写？
                          # 实例直接被调用的时候会直接调用__call__方法，虽然没有编写但是可以调用的原因是在父类torch.nn.Module的时候已经被实现
                          # torch.nn.Module的__call__方法调用了forward函数，所以forward函数需要重写
        y_pred = self.linear(x)  # 调用了线性层 self.linear 对输入 x 进行前向计算，得到了预测值 y_pred。
        return y_pred
    
model = LinearModel()  # 实例化

# Construct loss and optimizer
criterion = torch.nn.MSELoss(reduction='sum')
'''
size_average(已弃用):
默认值为 True,表示在计算平均损失时,是否应该除以样本的数量。
该参数在较新的版本中已经弃用,而被 reduction 参数取代。如果 size_average 为 True,则相当于 reduction='mean'；如果为 False，则相当于 reduction='sum'。

reduce(已弃用):
默认值为 True,表示是否在计算损失时进行降维操作。
这个参数在较新的版本中已经弃用,而被 reduction 参数取代。如果 reduce 为 True,则相当于 reduction='mean'；如果为 False，则相当于 reduction='none'。

reduction:
该参数指定了如何对损失进行汇总。可以选择以下三个选项之一：
'none'：不对损失进行汇总,保留每个样本的损失值。
'sum'：将所有样本的损失值相加,得到一个标量值。
'mean'：将所有样本的损失值求平均,得到一个标量值。
默认值为 'mean'。'''

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # model.parameters会返回模型中所有可学习的参数，这些参数包括了神经网络中的权重和偏置等需要通过反向传播算法进行优化的参数。
                                                          # 哪些参数需要学习是在类的初始化的时候已经涵盖了所有可学习参数 
                                                          # 定义了 LinearModel 类时，其中包含了一个线性层 self.linear = torch.nn.Linear(1, 1)
                                                          # 这个线性层是模型中的一个可学习参数，它包括了权重 w 和偏置 b
                                                          # 通过 model.parameters() 来获取模型中所有的可学习参数时，这个线性层中的权重和偏置也会被包括在内
                                                          # 通过在类的初始化中定义了可学习的参数（比如线性层），然后在训练时将这些参数传给优化器，告诉了优化器哪些参数需要被学习和更新
'''
总结一下：
在类的初始化中定义了线性层(或其他可学习的参数)时,这些参数会被PyTorch自动识别为需要学习的参数。
在训练过程中，通过调用 model.parameters() 来获取模型中所有需要学习的参数
然后传给优化器，使得优化器可以更新这些参数以最小化损失函数。
'''

# Training cycle
for epoch in range(1000):
    y_pred = model(x_data)  # 正向传播，生成运算图
    loss = criterion(y_pred, y_data)  # 计算损失
    
    print('epoch=', epoch, loss)  # 此处的loss会自动调用__str__方法，不会生成运算图，可以直接使用
    optimizer.zero_grad()  # 清空梯度，防止梯度累加
    loss.backward()  # 反向传播，与Lecture04相同，自动计算梯度，清空运算图，以便下次生成运算图（动态生成运算图）
    optimizer.step()  # 更新梯度
    
print('w=', model.linear.weight.item())
print('b=', model.linear.bias.item())
'''
model = LinearModel()  # 实例化后
model 中包含了初始化时候定义的成员变量linear
而linear中包含了两个可学习参数weight和bias所以调用它们分别是
model.linear.weight和model.linear.bias
由于只有一个权重和一个bias所以可以使用.item()提取成浮点数
weight = model.linear.weight.item()
bias = model.linear.bias.item()
'''

x_test = torch.tensor([4.0])
y_test = model(x_test)  # 实例被调用，默认调用__call__方法，其中会调用forward函数 返回值赋值给y_test
print('y_pred=', y_test.data)