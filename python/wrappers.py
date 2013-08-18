import numpy as np
import re


def load_boost_array(filename):
    with open(filename) as fd:
        data = fd.read()

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
