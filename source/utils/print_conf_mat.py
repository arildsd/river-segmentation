import numpy as np
a =[[ 6281134,45844,   331735,   189377, 5337,27823],
    [  109060,  1603065,   201620,81742,0,15796],
    [  271077,   173761, 18006698,   968519,   220320,36318],
    [  189144, 8947,  1300605, 10269082,   423204,17085],
    [   34860, 1517,   363549,   376577,  1945997,16595],
    [   64198,20895,   397217,   165104,81861,56673]]

for i in range(len(a)):
    for j in range(len(a[0])):
        a[i][j] = str(a[i][j])
s = [",".join(l) for l in a]
print(s)
print("\n\n")
s = "\n".join(s)
print(s)
