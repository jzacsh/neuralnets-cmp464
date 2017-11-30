"""
Jonathan Zacsh's solution to homework #3, Nov 14., Part I
"""
# Per homework instructions, following lead from matlab example by professor:
#   http://comet.lehman.cuny.edu/schneider/Fall17/CMP464/Maple/PartialDerivatives1.pdf

class Differentiable:
    """ encapsulation of a function and its derivative """
    def __init__(self, label, initial, rapidity, f, d):
        self.a = initial
        self.func = f
        self.deriv = d
        self.func.name = label
        self.deriv.name = "%sDeriv" % label
        self.step = 0
        self.rapidity = -1*rapidity

    def setLanding(self, descendedTo):
        self.a = descendedTo

    def currently(self):
        return self.a

    def projectDescent(self):
        self.step = float(self.step) + float(1)
        slope = self.deriv(self.a)
        return float(self.a) + float(self.rapidity) * float(slope)

    def labeler_(self, printf_format):
        return printf_format % self.step


# g(x) = x^4+2x-7 ; per matlab example
# g'(x) = 4x^3+2
fExFourth = Differentiable(
        "fExFourth",
        2, # initial value
        0.1, # rapidity
        lambda x: float(x)**4 + 2*float(x) - 7,
        lambda x: 4*(float(x)**3) + 2)

for i in range(0, 20):
    willLandAt = fExFourth.projectDescent()
    movingBy = fExFourth.currently() - willLandAt
    direction = "right" if movingBy > 0 else " left"
    fExFourth.setLanding(willLandAt)
    print("moved %05.30f to the %s, landing at %05.50f" % (abs(movingBy), direction, willLandAt))
