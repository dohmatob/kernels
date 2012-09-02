import re
import sys
from numpy import savetxt, array, vstack, fromstring

def ublas_matrix_to_numpy(filename):
    with open(filename, 'r') as ifh:
        raw_data = ifh.read()
        data = array([])
        for item in re.finditer("\((?P<row>[^\(]+?)\)", raw_data):
            row = fromstring(item.group('row'), sep=',')
            if len(data) == 0:
                data = row
            else:
                data = vstack((data, row))
                
        ifh.close()

    return data

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Usage: python %s [OPTIONS] <path_to_ublas_kernel_file>"%sys.argv[0]
        sys.exit(-1)

    kernel = ublas_matrix_to_numpy(sys.argv[1])

    savetxt("%s.numpyarray"%sys.argv[1], kernel)
