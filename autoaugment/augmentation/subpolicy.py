
class Subpolicy(object):
    def __init__(self, *operations):
        self.operations = operations

    def __call__(self, X):
        for op in self.operations:
            X = op(X)
        return X

    def __str__(self):
        ret = '['
        for i, op in enumerate(self.operations):
            ret += str(op)
            if i < len(self.operations) - 1:
                ret += ', '
        return ret + ']'
