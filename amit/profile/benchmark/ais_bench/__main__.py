import os
import sys
sys.path.insert(0, os.path.abspath("../../../"))
cur_path = os.path.dirname(os.path.realpath(__file__))
exec(open(os.path.join(cur_path, "infer/__main__.py")).read())