# %%
from numpy import random

center_aligment = 50

# %%
print("TASK 1".center(center_aligment))

expression = input("IsPalindrom? ")
isPalindrom = lambda e: e ==e[::-1]
print(f"Is '{expression}' a palindrome: {isPalindrom(expression)}")

# %%
print("TASK 2".center(center_aligment))

expression = input("Find longest word: ").strip().split(' ')
expression.sort(key=lambda e: len(e))
print(f"Longest word is: {expression[-1]}")

# %%
print("TASK 3".center(center_aligment))

numbers_list = random.randint(0, 200, 50)
even = len([i for i in numbers_list if i % 2 == 0])
odd = len(numbers_list) - even
print(f"numbers: {list(numbers_list)}\nFirst test:  odd: {odd}, even: {even}")
#Another code
even = sum(map(lambda x: x % 2 == 0, numbers_list))
odd = len(numbers_list) - even
print(f"Second test: odd: {odd}, even: {even}")


# %%
print("TASK 4".center(center_aligment))

diction = {"a": "b", "c":"d", "e":"f", "g":"o"}
newdiction = {}
for k, e in diction.items():
    newdiction.update({e:k})
print(f"Was:    {diction}\nBecome: {newdiction}")

newdiction = dict([(e,k) for k,e in diction.items()])
print(f"Another way: {newdiction}")



# %%
print("TASK 5".center(center_aligment))

def fib(n : int) -> int:
    if(n == 0): return 0
    if(n == 1): return 1
    return fib(n-1) + fib(n-2)

print(fib(9))

# %%
print("TASK 6".center(center_aligment))

file_name = input("File name to calculate: ").strip()

with open(file_name, "r") as file:
    cnt_lines, cnt_words, cnt_symbols = 0, 0, 0
    file_content = file.readlines()
    cnt_lines = len(file_content)
    for line in file_content:
        for word in line.split(' '):
            cnt_words+=1
            for symbol in word:
                cnt_symbols +=1
        cnt_symbols += len(line.split(' ')) - 1 #Костыль Считаем пробелы
print(f"Count of lines: {cnt_lines}\nCount of words: {cnt_words}\nCount of symbols: {cnt_symbols}")


# %%
print("TASK 7".center(center_aligment))

q, cnt_elems, elem = [int(x) for x in input("Введите коэффицент q, до какого члена вывести прогрессию, первый член прогрессии. (Вводить через пробел) ").split(' ') if x != '']
for i in range(cnt_elems+1):
    print(elem, end=' ')
    elem *= q


