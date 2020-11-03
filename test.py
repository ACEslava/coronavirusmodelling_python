import concurrent.futures 

def test(change):
    if change == 'a': return [0,1], 'a'
    elif change == 'b': return [2,3], 'b'
    elif change == 'g': return [4,5], 'g'
    elif change == 'd': return [6,7], 'd'
    elif change == 'z': return [8,9], 'z'
    elif change == 'o': return [10,11], 'o'

if __name__ == '__main__':
    args = [letter for element, letter in enumerate('abgdzo')]
    with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(test, args)
            executor.shutdown(wait=True)
    
    A = [0]*6
    B = [0]*6
    for element, sums in enumerate(list(results)):
        A[element], B[element] = sums[0:2]
    
    print(A,B)
        