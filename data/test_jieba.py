#encoding: utf-8

import jieba

# seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
# print("Full Mode: " + "/ ".join(seg_list))  # 全模式
#
seg_list = jieba.cut("我来到北京清华大学。", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式
#
# seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
# print(", ".join(seg_list))
#
# seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
# print(", ".join(seg_list))

seg_list = jieba.cut("小明硕士毕业于中国科学院计算所。",cut_all=False)  # 默认是精确模式
print(", ".join(seg_list))

# seg_list = []
# str = "小明硕士毕业于中国科学院计算所"
# print(len(str))
# for i in range(len(str)/3):
#     seg_list.append(str[i*3:(i+1)*3])
# print(", ".join(seg_list))

a = "abc。"
print(a[-1])
if a[-1] == "。":
    print(a)