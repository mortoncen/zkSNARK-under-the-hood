import galois
import numpy as np
from utils import generator1, generator2, curve_order, normalize, validate_point

G1 = generator1()
G2 = generator2()


class SRS:
    """Trusted Setup Class aka Structured Reference String"""
    def __init__(self, tau, n = 2):
        self.tau = tau
        self.tau1 = [G1 * int(tau)**i for i in range(0, n + 7)]
        self.tau2 = G2 * int(tau)

    def __str__(self):
        s = f"tau: {self.tau}\n"
        s += "".join([f"[tau^{i}]G1: {str(normalize(point))}\n" for i, point in enumerate(self.tau1)])
        s += f"[tau]G2: {str(normalize(self.tau2))}\n"
        return s

def new_call(self, at, **kwargs):
    if isinstance(at, SRS):
        coeffs = self.coeffs[::-1]
        result = at.tau1[0] * coeffs[0]
        for i in range(1, len(coeffs)):
            result += at.tau1[i] * coeffs[i]
        return result

    return galois.Poly.original_call(self, at, **kwargs)

galois.Poly.original_call = galois.Poly.__call__
galois.Poly.__call__ = new_call

# to switch between encrypted (ECC) and unencrypted mode (prime field p=241)
encrypted = False

# Prime field p
p = 241 if not encrypted else curve_order
Fp = galois.GF(p)

# We have 7 gates, next power of 2 is 8
n = 7
n = 2**int(np.ceil(np.log2(n)))
assert n & n - 1 == 0, "n must be a power of 2"

# Find primitive root of unity (generator)
omega = Fp.primitive_root_of_unity(n)
assert omega**(n) == 1, f"omega (ω) {omega} is not a root of unity"
# ω = 30

roots = Fp([omega**i for i in range(n)])
print(f"roots = {roots}")
# roots = [  1  30 177   8 240 211  64 233]

def pad_array(a, n):
    return a + [0]*(n - len(a))

# witness vectors
a = [2, 2, 3, 4, 4, 8, -28]
b = [2, 2, 3, 0, 9, 36, 3]
c = [4, 4, 9, 8, 36, -28, -25]

# gate vectors
ql = [0, 0, 0, 2, 0, 1, 1]
qr = [0, 0, 0, 0, 0, -1, 0]
qm = [1, 1, 1, 1, 1, 0, 0]
qc = [0, 0, 0, 0, 0, 0, 3]
qo = [-1, -1, -1, -1, -1, -1, -1]

# pad vectors to length n
a = pad_array(a, n)
b = pad_array(b, n)
c = pad_array(c, n)
ql = pad_array(ql, n)
qr = pad_array(qr, n)
qm = pad_array(qm, n)
qc = pad_array(qc, n)
qo = pad_array(qo, n)

# --- padded vectors ----
a = [2, 2, 3, 4, 4, 8, -28, 0]
b = [2, 2, 3, 0, 9, 36, 3, 0]
c = [4, 4, 9, 8, 36, -28, -25, 0]
ql = [0, 0, 0, 2, 0, 1, 1, 0]
qr = [0, 0, 0, 0, 0, -1, 0, 0]
qm = [1, 1, 1, 1, 1, 0, 0, 0]
qc = [0, 0, 0, 0, 0, 0, 3, 0]
qo = [-1, -1, -1, -1, -1, -1, -1, 0]

ai = range(0, n)
bi = range(n, 2*n)
ci = range(2*n, 3*n)

sigma = {
    ai[0]: ai[0], ai[1]: ai[1], ai[2]: ai[2], ai[3]: ci[0], ai[4]: ci[1], ai[5]: ci[3], ai[6]: ci[5], ai[7]: ai[7],
    bi[0]: bi[0], bi[1]: bi[1], bi[2]: bi[2], bi[3]: bi[3], bi[4]: ci[2], bi[5]: ci[4], bi[6]: bi[6], bi[7]: bi[7],
    ci[0]: ai[3], ci[1]: ai[4], ci[2]: bi[4], ci[3]: ai[5], ci[4]: bi[5], ci[5]: ai[6], ci[6]: ci[6], ci[7]: ci[7],
}

k1 = 2
k2 = 4
c1_roots = roots
c2_roots = roots * k1
c3_roots = roots * k2

c_roots = np.concatenate((c1_roots, c2_roots, c3_roots))

check = set()
for r in c_roots:
    assert not int(r) in check, f"Duplicate root {r} in {c_roots}"
    check.add(int(r))

sigma1 = Fp([c_roots[sigma[i]] for i in range(0, n)])
sigma2 = Fp([c_roots[sigma[i + n]] for i in range(0, n)])
sigma3 = Fp([c_roots[sigma[i + 2 * n]] for i in range(0, n)])

def to_galois_array(vector, field):
    # normalize to positive values
    a = [x % field.order for x in vector]
    return field(a)

def to_poly(x, v, field):
    assert len(x) == len(v)
    y = to_galois_array(v, field) if type(v) == list else v
    return galois.lagrange_poly(x, y)

QL = to_poly(roots, ql, Fp)
QR = to_poly(roots, qr, Fp)
QM = to_poly(roots, qm, Fp)
QC = to_poly(roots, qc, Fp)
QO = to_poly(roots, qo, Fp)

print("QL:", QL)
print("QR:", QR)
print("QM:", QM)
print("QC:", QC)
print("QO:", QO)

S1 = to_poly(roots, sigma1, Fp)
S2 = to_poly(roots, sigma2, Fp)
S3 = to_poly(roots, sigma3, Fp)
print("S1:", S1)
print("S2:", S2)
print("S3:", S3)

I1 = to_poly(roots, c1_roots, Fp)
I2 = to_poly(roots, c2_roots, Fp)
I3 = to_poly(roots, c3_roots, Fp)
print("I1:", I1)
print("I2:", I2)
print("I3:", I3)

def to_vanishing_poly(roots, field):
    # Z^n - 1 = (Z - 1)(Z - w)(Z - w^2)...(Z - w^(n-1))
    return galois.Poly.Degrees([len(roots), 0], coeffs=[1, -1], field=field)

Zh = to_vanishing_poly(roots, Fp)
print("Zh:", Zh)

for x in roots:
    assert Zh(x) == 0
# --- Vanishing Polynomial ---
# Zh = x^8 + 240 which translates to x^8 - 1 (240 == -1 in Prime Field p=241)

def generate_tau(encrypted):
    """ Generate a random tau in Fp or G1 """
    return SRS(Fp.Random(), n) if encrypted else Fp.Random()

tau = generate_tau(encrypted)
print("tau:", tau)
