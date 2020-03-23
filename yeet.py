import numpy as np

arr = np.asarray([3, 4, 5, 6, 7])
num = 10

def do_thing(arr, num):
    for i in range(len(arr)):
        for j in range(i+1, len(arr), 1):
            if arr[i]+arr[j] == num:
                print(arr[i], ",", arr[j])

do_thing(arr, num)