from math import sqrt


# distance formula
def has_colided(x1,y1,x2,y2,ofset):
    distance = sqrt(pow(x2-x1,2) + pow(y2-y1,2))
    if distance <=ofset:
        return True
    else:
        return False

def distance():
    distance = sqrt(pow(x2-x1,2) + pow(y2-y1,2))
    return distance