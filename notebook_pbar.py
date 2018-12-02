from time import time,sleep
from IPython.display import clear_output
import numpy as np

def progress_bar(x,y):
    n = y
    m = 1
    max_prog_bar_width = 40
    while n > max_prog_bar_width:
        m += 1
        n = int(y/m)
    x2 = int(x/m)
    y2 = int(y/m)
    prog_bar = str(u"\u2588"*x2+'-'*(y2-x2-1)+" ")
    pct_complete = str(int(((x2+1)/y2)*1000)/10)+'% Complete '
    item_of_tot = "- Item: "+str(x+1)+" of "+str(y)
    status_string = str(prog_bar+pct_complete+item_of_tot)
    return status_string

def pbar(timelist, then_time, denominator, numerator):
    if len(timelist) != 5:
        timelist = [0.5, 0.5, 0.5, 0.5, 0.5]
    pb = progress_bar(numerator,denominator)
    new_t = time()
    timelist.append(then_time - new_t)
    timelist = timelist[1:]
    avg_time_delta = int(np.abs(np.mean(timelist))*100)/100
    rem = int(avg_time_delta/60*((denominator-numerator)/5)*100)/100
    print(str(pb+" Time Between: "+str(avg_time_delta)+" Remaining: "+str(rem)+" mins"))
    clear_output(wait=True)
    return new_t, timelist

timelist = [0.5, 0.5, 0.5, 0.5, 0.5]
then_time = time()