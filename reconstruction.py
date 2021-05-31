#%%
import numpy as np

from ccpi.optimisation.algorithms import PDHG, FISTA
from ccpi.framework import ImageGeometry, AcquisitionGeometry, AcquisitionData
from ccpi.optimisation.operators import BlockOperator, Gradient, \
                        SymmetrizedGradient, ZeroOperator, Identity, FiniteDiff
from ccpi.optimisation.functions import ZeroFunction, L2NormSquared, \
                      MixedL21Norm, BlockFunction, FunctionOperatorComposition
from ccpi.astra.operators import AstraProjectorSimple, AstraProjectorMC
from ccpi.astra.processors import FBP
from ccpi.plugins.regularisers import FGP_TV, TNV, TGV

# path to read and store data
path = 'your_path'

#%% load preprocessed sinogram
sino = np.load(path+'sino_flux_normalization_1.npy')

#%% set-up geometries
n_channels = 339

angles = np.array(range(0, 120)) * 1.5

# set-up ccpi objects
ag_MC = AcquisitionGeometry('parallel', '2D',  \
                         angles, \
                         pixel_num_h = 460, \
                         pixel_size_h = 0.055,
                         channels = n_channels)  

ad_MC = ag_MC.allocate()
ad_MC.fill(sino_test)

ig_MC = ImageGeometry(voxel_num_x = 460, 
                   voxel_num_y = 460, 
                   voxel_size_x = 0.055, 
                   voxel_size_y = 0.055,
                   channels = n_channels)

op_MC = AstraProjectorMC(ig_MC, ag_MC, 'gpu')

ag = AcquisitionGeometry('parallel', '2D',  \
                         angles, \
                         pixel_num_h = 460, \
                         pixel_size_h = 0.055)  

ad = ag.allocate()
ad.fill(np.mean(sino_test, axis=0))

ig = ImageGeometry(voxel_num_x = 460, 
                   voxel_num_y = 460, 
                   voxel_size_x = 0.055, 
                   voxel_size_y = 0.055)

op_simple = AstraProjectorSimple(ig, ag, 'gpu')

#%% FBP recon
# configure FBP
fbp = FBP(ig_MC, ag_MC, filter_type = 'hann', device = 'gpu')
# pass actual AcquisitionData
fbp.set_input(ad_MC)
# run FBP and get results
recon_fbp_MC = fbp.get_output()

np.save(path+'fbp_MC.npy', recon_fbp_MC.as_array())

# configure FBP
fbp = FBP(ig, ag, filter_type = 'hann', device = 'gpu')
# pass actual AcquisitionData
fbp.set_input(ad)
# run FBP and get results
recon_fbp = fbp.get_output()

np.save(path+'fbp_white_beam.npy', recon_fbp.as_array())
        
#%% TNV reconstruction

alpha = 0.01
g_TNV = TNV(alpha, 50, 5e-6)

# Setup fidelity term
f = FunctionOperatorComposition(0.5 * L2NormSquared(b = ad_MC), op_MC)

x_init = ig_MC.allocate()

# Run FISTA for least squares
fista_TNV = FISTA(x_init = x_init, f = f, g = g_TNV)
fista_TNV.max_iteration = 2000
fista_TNV.update_objective_interval = 100
fista_TNV.run(2000, verbose = True)
    
np.save(path+'fista_tnv_{}.npy'.format(alpha), fista_TNV.get_output().as_array())


#%% TV+TGV reconstruction

op11 = Gradient(ig_MC, correlation='Space')
op12 = ZeroOperator(ig_MC, op11.range_geometry())

op21 = FiniteDiff(ig_MC, direction = 0)
op22 = -Identity(ig_MC)

op31 = ZeroOperator(ig_MC)
op32 = FiniteDiff(ig_MC, direction = 0)

op41 = op_MC
op42 = ZeroOperator(ig_MC, ag_MC)

operator = BlockOperator(op11, op12, 
                         op21, op22, 
                         op31, op32, 
                         op41, op42, shape=(4,2))

alpha = 0.0075
beta = 0.3

gamma = np.sqrt(2) * beta

f1 = alpha * MixedL21Norm()
f2 = beta * L1Norm() 
f3 = gamma * L1Norm()
f4 = 0.5 * L2NormSquared(b=ad_MC)

f = BlockFunction(f1, f2, f3, f4)    
g = ZeroFunction()

# Compute operator Norm
normK = operator.norm()

# Primal & dual stepsizes
sigma = 1
tau = 1 / (sigma*normK**2)

# Setup and run the PDHG algorithm
pdhg = PDHG(f=f, 
            g=g, 
            operator=operator, 
            tau=tau, 
            sigma=sigma)
pdhg.max_iteration = 2000
pdhg.update_objective_interval = 500
pdhg.run(2000)
        
np.save(path+'pdhg_tgv_alpha_{}_beta_{}.npy'.format(alpha,beta), pdhg.get_output()[0].as_array())