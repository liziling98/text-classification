# text-classification

新闻文本分类，数据来源于Reuters 1996-1997年间的新闻。

类别主要分为industry, region 和 topics三类，数据包含超过80万条新闻，每条新闻对应多个标签，一共出现了752个标签，其中industry和topics类别下的labels存在父子类关系。

标签分布存在不平衡的关系，超过一半的标签出现次数不超过200，同时又有8个标签出现次数超过1万。

将在毕业设计中使用如SVM，logistic regression之类的传统机器学习算法，以及CNN，RNN之类的深度学习算法。
