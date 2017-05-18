while True:
    num = raw_input("Enter a number: ")
    if num == "done" : break
    try:        num = int(num)
    except:
        print "invalid input"        continue
        numbers = list (num)  largest = None smallest = None  for num in numbers:    if smallest == None:        num = smallest        for num in numbers:    if largest == None:        num = largest  print "Maximum is", largest print "minimum is", smallest
