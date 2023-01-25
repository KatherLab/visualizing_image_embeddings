#!/usr/bin/env python3

__author__ = 'Jeff'
__copyright__ = 'Copyright 2023, Kather Lab'
__license__ = 'MIT'
__version__ = '0.1.0'
__maintainer__ = ['Jeff']
__email__ = 'jiefu.zhu@tu-dresden.de'




from cluster_and_plot import DataClustering

dc = DataClustering('out')
dc.run_tSNE()
dc.plot_scatter()
#dc.plot_imgs()
