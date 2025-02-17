import itertools
import random
from tinygrad.helpers import DEBUG
from tinygrad.shape.symbolic import Variable
random.seed(42)

def add_v(expr, rng=None):
  if rng is None: rng = random.randint(0,2)
  return expr + v[rng], rng

def div(expr, rng=None):
  if rng is None: rng = random.randint(1,9)
  return expr // rng, rng

def mul(expr, rng=None):
  if rng is None: rng = random.randint(-4,4)
  return expr * rng, rng

def mod(expr, rng=None):
  if rng is None: rng = random.randint(1,9)
  return expr % rng, rng

def add_num(expr, rng=None):
  if rng is None: rng = random.randint(-4,4)
  return expr + rng, rng

def lt(expr, rng=None):
  if rng is None: rng = random.randint(-4,4)
  return expr < rng, rng

def ge(expr, rng=None):
  if rng is None: rng = random.randint(-4,4)
  return expr >= rng, rng

if __name__ == "__main__":
  ops = [add_v, div, mul, add_num, mod]
  for _ in range(1000):
    upper_bounds = [*list(range(1, 10)), 16, 32, 64, 128, 256]
    u1 = Variable("v1", 0, random.choice(upper_bounds))
    u2 = Variable("v2", 0, random.choice(upper_bounds))
    u3 = Variable("v3", 0, random.choice(upper_bounds))
    v = [u1,u2,u3]
    tape = [random.choice(ops) for _ in range(random.randint(2, 30))]
    # 10% of the time, add a less than or greater than
    if random.random() < 0.05: tape.append(lt)
    elif random.random() < 0.05: tape.append(ge)
    expr = Variable.num(0)
    rngs = []
    for t in tape:
      expr, rng = t(expr)
      if DEBUG >= 1: print(t.__name__, rng)
      rngs.append(rng)
    if DEBUG >=1: print(expr)
    space = list(itertools.product(range(u1.min, u1.max+1), range(u2.min, u2.max+1), range(u3.min, u3.max+1)))
    volume = len(space)
    for (v1, v2, v3) in random.sample(space, min(100, volume)):
      v = [v1,v2,v3]
      rn = 0
      for t,r in zip(tape, rngs): rn, _ = t(rn, r)
      num = eval(expr.render())
      assert num == rn, f"mismatched {expr.render()} at {v1=} {v2=} {v3=} = {num} != {rn}"
      if DEBUG >= 1: print(f"matched {expr.render()} at {v1=} {v2=} {v3=} = {num} == {rn}")
