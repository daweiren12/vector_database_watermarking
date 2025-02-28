def float_to_bin(num):
    if num < 0:
        num = -num
    if num == int(num):
        re = bin(int(num))[2::] + '.' + '0'
        return re
    else:
        integer_part = int(num)
        decimal_part = num - integer_part
        integer_part_bin = bin(integer_part)
        temp = decimal_part
        temp_flo = []
        while 1:
            temp = temp * 2
            temp_flo += str(int(temp))
            if temp > 1:
                temp = temp - int(temp)
            elif temp < 1:
                pass
            else:
                break
        re = integer_part_bin + '.' + ''.join(temp_flo)
        re = str(re)[2::]
        return re

def bin_to_float(b):
        if '.' in b:
            int_part, frac_part = b.split('.')
            frac = sum(int(bit) * (2 ** -(i + 1)) for i, bit in enumerate(frac_part))
            return int(int_part, 2) + frac
        return int(b, 2)