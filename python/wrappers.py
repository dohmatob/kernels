import numpy as np
import re


def load_boost_array(filename):
    fd = open(filename)
    data = fd.read()
    fd.close()

    m1 = re.search("^\[(?P<nr>\d+?),(?P<nc>\d+?)\]\((?P<lines>.+?)\)$",
                    data)
    if m1 is None:
        return None
    else:
        nr = int(m1.group('nr'))
        nc = int(m1.group('nc'))
        m2 = re.finditer('\((?P<line>.+?)\)', m1.group("lines"))
        if m2 is None:
            return None
        else:
            lines = np.array([np.fromstring(m3.group('line'), sep=",")
                              for m3 in m2])

            return lines if lines.shape == (nr, nc) else None


def write_libsvm_train_data(X, Y, output_filename):
    with open(output_filename, 'a') as fd:
        fd.write("\n".join(["%i %s" % (Y[i], " ".join(["%i:%s" % (
                                    j + 1, X[i][j])
                                                       for j in xrange(
                                    len(X[i]))]))
                            for i in xrange(len(X))]))


def read_mmsk_model(leafs_filename, kernel_filename):
    lfd = open(leafs_filename)
    raw = lfd.read()
    lfd.close()

    lines = raw.rstrip('\n').split("\n")
    assert len(lines) > 2

    l, k, m = [int(x.split('=')[1]) for x in lines[:3]]

    leafs = {}
    for line in lines[3:]:
        x, y = line.split(':')
        leafs[x] = dict((int(item.group("index")),
                         int(item.group("count")))
                        for item in re.finditer(
                "\((?P<index>\d+?),(?P<count>\d+?)\)", y))

    return l, k, m, leafs, load_boost_array(kernel_filename)
