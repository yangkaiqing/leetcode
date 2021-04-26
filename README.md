



# **数据结构与算法Python**版

## 一、动态规划

### 1.1 动态规划解题步骤：

**a. 确定状态**

​    即dp表示什么含义，比如在最大子序列和问题中，dp[i]的含义为，以nums[i]为结尾的子序列的的最大和。

**b. 确定动态转移方程**

   以最大子序列和问题为例，动态转移方程为


$$
dp[i] = max(dp[i-1], 0) + nums[i]
$$
  其含义为，若以nums[i-1]结尾的的子序列和为负值，则以nums[i]结尾的最大子序列和为nums[i]。

**c.确定初始条件**

  仍以最大子序列和为例，初始化条件为 
$$
dp[0] = nums[0]
$$
  表示以nums[0]为结尾的的最大和子序列为nums[0]

### 1.2 例题

a. 最大子序列和

​    问题描述：给定一个整数数组nums, 找到一个具有最大和的连续子数组(子数组最少包含一个元素)，并返回其最大和。

​    示例：输入： [-2, 1,-3, **4, -1, 2, 1, -5,** 4] 

​               输出：6

​               连续数组[4,-1,2,1]的和最大，最大为6。

```python
 def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
        dp     = n * [0]
        #初始化条件
        dp[0]  = nums[0]
        retval = dp[0]
        for x in range(1,n):
            #动态转移方程
            dp[x]  = max(dp[x-1], 0) + nums[x]
            #更新最大值
            retval = max(retval, dp[x])
        return retval
```

b. 买卖股票的最佳时机

​    问题描述：给定一个数组prices，它的第i个元素prices[i]表示一支股票在第天的价格。只可以选择在**某一天**买入这只股票，并且选择在**未来某一天不同日子**卖出，求可以获得的最大利润。你**最多只能完成一次**交易。

​    示例：输入：[7,**1**,5,3,**6**,4]

​               输出：5

​               表示在第2天买入，第5天卖出，利润最大，最大为5。

​    题解（方法二）：由于只能完成一次交易，每天只有两种状态，买入股票hold，不买入股票nohold，第x+1天不买入的时候nohold，手中无股票，可能是上一天就没有股票nohold，也可能上一天有，今天卖出即hold+prices[x];hold表示x+1天结束的时候我们持有股票的最大利润，一个是上一天本来就持有股票，一个是今天买入-prices[i]。

```python
#方法一：将问题转化为求解[prices[i]-prices[i-1]]中的最大和连续子序列
def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0
        n      = len(prices)
        dp     = n * [0]
        dp[0]  = 0 # 第一天不能卖股票，所以为0
        cur    = dp[0]
        retval = dp[0]
        for x in range(1, n):
            # 相当于求解新数组prices[x] - prices[x-1]的最大和连续子序列
            cur    = max(cur, 0) + prices[x] - prices[x-1] 
            retval = max(retval, cur)
        return retval
#方法二
def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0
        n      = len(prices)
        hold   = -prices[0] #第一天买入
        nohold = 0          #第一天不买入
        for x in range(1,n)xxx
            #第x+1天结束的时候没持有股票
            #第一种情况就是第x+1天我们即没买也没卖，那么最大利润就是第i天没持有股票的最大利润
            #第二种情况就是第x+1天我们卖了一支股票，那么最大利润就是第i天持有股票的最大利润
            nohold = max(nohold, hold+prices[x])
            # x+1天结束的时候我们持有股票的最大利润
            hold = max(hold, -prices[x])
        return nohold
 #方法三：贪心
```

c. 买卖股票的最佳时机II

​     问题描述：给定一个数组 `prices` ，其中 `prices[i]` 是一支给定股票第 `i` 天的价格。**你可以尽可能地完成更多的交易（多次买卖一支股票）**。**你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。**

​     示例：输入：prices = [7,1,5,3,6,4]

​                输出：7

​                在第二天买入，第三天卖出。第四天买入，第五天卖出，利润最大，最大为5-1+6-3=7。

​                

​     题解：方法一：动态规划同上；

​                 方法二：贪心，因为可以多次买卖，所以只要有利润(prices[x]-prices[x-1]>0)就卖出。 

```python
# 方法一：动态规划
def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0
        n      =  len(prices)
        hold   =  -prices[0] # 第一天买入
        nohold =  0          # 第一天没有买入
        for x in range(1,n):
            # 表示第x+1天交易完成后，手里没有股票的最大利润
            # nohlod 没有任何交易
            # hold+prices[x] 卖掉当天手里的股票，既然可以卖，则手里有股票，前一天利润加上当天利润
            nohold = max(nohold, hold+prices[x])
            # 表示x+1天交易完成后，手里持有股票的最大利润
            # hold没有交易
            # nohold-prices[x] 买入股票
            hold   = max(hold, nohold-prices[x])
        return nohold
# 方法二：贪心，有利可图就卖股票
def maxProfit(self, prices: List[int]) -> int:
        retval = 0
        for x in range(1, len(prices)):
            retval = max(prices[x]-prices[x-1], 0) + retval
        return retval
```

d.买卖股票的最佳时机III 

​		包含手续费，这里只要卖出的时候减去手续费就好了。

```python
def maxProfit(self, prices: List[int], fee: int) -> int:
        if not prices:
            return 0
        hold   = -prices[0]
        nohold = 0
        for x in range(1, len(prices)):
            hold   = max(hold,   nohold - prices[x])
            nohold = max(nohold, hold + prices[x] - fee)
        return nohold
```

e.礼物的最大价值

​		问题描述：在一个 m*n 的棋盘的每一格都放有一个礼物，**每个礼物都有一定的价值（价值大于 0）**。你可以从棋盘的左上角开始拿格子里的礼物，**并每次向右或者向下移动一格、直到到达棋盘的右下角**。给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物？

​		示例：

```
输入：[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
输出：12 
路径1->3->5->2->1可以拿到最多礼物
```

   	解析：dp[x][y]表示下xy位置可以获得的最大价值礼物，由于只可以向右向下移动,也就是说dp[x][y]的计算结果只能来自所以x-1，y以及x，y-1。即

$$
dp[x][y] = max(dp[x-1][y],dp[x][y-1])+grid[x][y]
$$

​		初始条件：
$$
dp[0][0]=grid[0][0]
$$

```python
#方法一，
def maxValue(self, grid: List[List[int]]) -> int:
        if not grid:
            return 0
        m = len(grid)
        n = len(grid[0])
        dp = [[0]*n for _ in range(m)]
        for x in range(m):
            for y in range(n):
                if x == 0 and y == 0:
                    dp[x][y] = grid[x][y]
                else:
                    dp[x][y] = max(dp[x-1][y],dp[x][y-1]) + grid[x][y]
        return dp[x][y]
#方法二，可以把dp转为一维数组
```

f.不同路径

		问题描述：一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。问总共有多少条不同的路径？
		输入：m=7,n=3
		输出：28

​			解析：用dp[x][y]表示到达xy位置时，不同的路径数目，和上一题类似，由于只能向右向下运动，所以动态规划的递推关系式为：
$$
dp[x][y] = dp[x-1][y]+dp[x][y-1]
$$
​			初始化条件为
$$
dp[0][0] = 1
$$

```python
def uniquePaths(self, m: int, n: int) -> int:
        dp = [ [0]*n for _ in range(m)]
        for x in range(m):
            for y in range(n):
                if x==0 and y==0:
                    dp[x][y] = 1
                else:
                    dp[x][y] = dp[x-1][y] + dp[x][y-1]
        return dp[x][y]
```

