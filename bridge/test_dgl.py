import numpy as np
import os
import sys
import urllib.request
import tempfile
import tarfile
from pathlib import Path
import dgl
import torch
actor = dgl.data.ActorDataset(force_reload=True)
chameleon = dgl.data.ChameleonDataset(force_reload=True)
squirrel = dgl.data.SquirrelDataset(force_reload=True)
A = actor[0].adjacency_matrix().to_dense()