print(1^0)
print(0^0)
li=[1,2,3,4]
print("li before:", li)
def change(li2):
    li2[0]=100
change(li)
print("li AFTER:", li)

x=5
def change2(x):
    x=89
print("x before:", x)
change2(x)
print("x after:", x)
