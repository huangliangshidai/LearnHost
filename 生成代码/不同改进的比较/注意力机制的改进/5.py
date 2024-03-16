def minimize_factors(n, arr):
    ones = arr.count(1)
    twos = arr.count(2)
    threes = arr.count(3)

    # 计算红色元素 2 和红色元素 3 分别的因子数量
    red_factors = twos * 2 + threes * 3

    # 计算白色元素的因子数量
    white_factors = ones

    # 尽可能将红色元素 2 变成红色元素 3
    while twos > 0 and threes > 0:
        twos -= 1
        threes -= 1
        red_factors -= 2
        red_factors += 3

    # 如果还有剩余的红色元素 2，将其中两个变成红色元素 3
    if twos >= 2:
        twos -= 2
        red_factors -= 4
        red_factors += 6

    # 计算最终的因子数量和
    result = red_factors + white_factors

    return result

# 输入数组的大小
n = int(input())
# 输入数组的元素，并用空格分隔后转换为整数列表
arr = list(map(int, input().split()))

# 调用函数，计算并输出结果
result = minimize_factors(n, arr)
print(result)
