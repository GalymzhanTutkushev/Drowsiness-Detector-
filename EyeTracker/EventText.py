import time
begin=time.time()

def write_event(text):
    current_time=time.time()
    with open("events.txt", "a") as file:
        file.write("\n"+"Прошло времени: "+str(current_time-begin)+" "+text)

with open("events.txt", "w") as file:
    file.write("Начало записи " + str(begin))
curp = 0
while True:
    cur_time = round(time.time()-begin)
    if cur_time % 600 == 0:
        cur = cur_time / 60
        if cur != curp:
            with open("events.txt", "a") as file:
                file.write("\nНормальное состояние. Прошло минут от начала:" + str(cur))
        curp = cur


write_event("end")


