import operator as op
from functools import reduce

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2

def stats(n_compounds=8):
    unique_mixture_numbers = ncr(n_compounds,1) + ncr(n_compounds,2) + ncr(n_compounds,3) + ncr(n_compounds,4) + ncr(n_compounds,5) +ncr(n_compounds,6) +ncr(n_compounds,7) +ncr(n_compounds,8) 
    # +ncr(n_compounds,9) +ncr(n_compounds,10) +ncr(n_compounds,11) +ncr(n_compounds,12)



    unique_1C_mixture_numbers = ncr(n_compounds,1) 
    unique_2C_mixture_numbers = ncr(n_compounds,2) 
    unique_3C_mixture_numbers = ncr(n_compounds,3) 
    unique_4C_mixture_numbers = ncr(n_compounds,4) 
    unique_5C_mixture_numbers = ncr(n_compounds,5) 
    unique_6C_mixture_numbers = ncr(n_compounds,6) 
    unique_7C_mixture_numbers = ncr(n_compounds,7) 
    unique_8C_mixture_numbers = ncr(n_compounds,8) 




    print('Total 1-C combinations:', unique_1C_mixture_numbers)
    print('Total 2-C combinations:', unique_2C_mixture_numbers)
    print('Total 3-C combinations:', unique_3C_mixture_numbers)
    print('Total 4-C combinations:', unique_4C_mixture_numbers)
    print('Total 5-C combinations:', unique_5C_mixture_numbers)
    print('Total 6-C combinations:', unique_6C_mixture_numbers)
    print('Total 7-C combinations:', unique_7C_mixture_numbers)
    print('Total 8-C combinations:', unique_8C_mixture_numbers)

    print('Total combinations:', unique_mixture_numbers)