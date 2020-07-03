import numpy as np
class Splitter:
    def split(self, XI):
        thresh = np.unique(XI)
        if len(thresh) <= 10:
            ret = []
            for t in thresh:
                ret.append(((XI>=t).astype(int), t))
            return ret
        else:
            mn = XI.min()
            mx = XI.max()
            l = (mx-mn) / 10
            ret = []
            m = mn
            i = 1
            if l == 0:
                ret.append(((XI>=mn).astype(int), mn))
                return ret
            while m <= mx:
                i+=1
                ret.append(((XI>=m).astype(int), m))
                m += l
            return ret