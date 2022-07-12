# 【学习笔记】GitModel统计分析

> &#8195;&#8195;本文是Datawhale和GitModel开源项目的学习记录，一方面是梳理知识框架，另一方面是提升的代码熟练度。这里十分感谢Datawhle贡献者提供的项目资源。
>- 项目源：https://github.com/Git-Model/Modeling-Universe/tree/main/Data-Story
>- 开始时间：2022年7月9日
>- 进度：Python基础（Task已完成，总结ing...）

# 学习路径

本次项目的主要目标有三个：
- 学习**建模基础知识**，了解从数据导入、探索性数据分析和建立回归/分类模型的过程；
- 阅读资料和代码练习，提高对Pandas、Scipy、Matplotlib等**常用库的实践能力**；
- 通过项目提供的案例，举一反三，提升**统计分析应用能力**。

因此，进一步拆解本次学习项目的目标，可以得到具体的学习任务（这里并不完全按照Datawhale组队学习的Task）。每完成一个学习任务，就在任务后作标记，本文也会随着学习的进度更新。

## 1. Python基础
目标是学习环境搭建，熟悉基础语法，会使用相应的库。
  - 成功**搭建Python环境**，配置好需要使用的Packages。==【Done】==
  - **快速阅读**[**JoyfulPandas**](http://joyfulpandas.datawhale.club/Content/index.html)，关注目录内容，以便后期翻阅。==【Done】==
  - **完成“ Task1：Pandas 动手学入门”的35道题目**。==【Doing】==

## 2. 建模基础
目标是学习建模的主要过程，包括EDA、可视化、回归/分类模型。
  - 熟悉EDA过程，**完成“Task2：EDA初体验”**。
  - 实现可视化，**完成“Task5：数据可视化”**。
  - 学习回归分析原理，**完成 “Task3：统计分析之回归分析 ”**。
  - 学习回归分类原理，**完成“Task4：统计分析之分类分析”**。

## 3. 项目实战
目标是将上述的知识融会贯通，举一反三。
  - 选择一道赛题，**完整走一遍建模流程**。



# 一、Python基础
## 1. 环境搭建
> - 系统：**Win10**
> - 版本：**Python3.7**
> - 工具：**PyCharm**、**Anaconda(Jupyter Lab)**、**Typora**

**Anaconda**对于新人来说挺友好的，内置了Numpy 、Pandas等常用科学库，安装即可使用。在做数据处理与分析时，通常会使用Jupyter Notebook或Jupyter Lab。个人更推荐后者，其模块化管理功能优秀，支持分屏和Markdown目录，适合作文档管理工具和文本阅读，Code & Write。

**PyCharm**是Python的IDE工具，主要是用于Python开发，功能更多但配置也较为复杂。于个人而言，PyCharm主要是用来看源代码与说明文档，代码规范化和存储为.py文件，偶尔用作远程访问服务器。另外，PyCharm支持使用Anaconda的开发环境，对各类包或是环境管理起来也是比较方便。

**Typora**是基于Markdown的写作工具，优点是所见即所得，而缺点是付费（早期是免费的）。关于支持Markdown的工具，个人比较推荐飞书，免费、功能更多且存储方便，缺点是没办法离线写作。
## 2. Pandas 动手学入门
Pandas基于Numpy进行扩展，是数据处理与数据分析的重要工具。Datawhale的[**JoyfulPandas**](http://joyfulpandas.datawhale.club/Content/index.html)十分适合Pandas入门者去阅读，章节知识划分详细且有具体例子。另外，去年有写过关于Numpy和Pandas数据结构的特点（[链接](https://blog.csdn.net/weixin_44790239/article/details/115033911?spm=1001.2014.3001.5501)），`list`、`set`、`dict`、`ndarray`、`series`、`dataframe`的转化过程，涉及到深度学习，还会接触到`tensor`。这些数据结构是处理数据的基础，因此在学习Pandas时需要体会其每一次变化的逻辑。
>本人具备有一定的Python基础，因此直接跳过基础语法部分。
>这里也给大家推荐一些可以学习基础语法的路径。
>入门：B站“Python”教程，任意一个视频。
>进阶：[Intermediate Python(英文)](https://github.com/yasoob/intermediatePython)、[Intermediate Python(中译)](http://shouce.jb51.net/Python_jj/index.html)，需要时再阅读。

## 3. 习题摘录

（补充ing...）
