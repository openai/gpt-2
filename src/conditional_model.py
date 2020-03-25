def recursive(listy):
    
    while listy:
        print(listy)
        # listy = listy[1:]
        listy.append(recursive(listy[1:])*2)
        
        return listy

listy = [1,2,3] + [10,20,30,40]
print(listy)
print(recursive(listy))