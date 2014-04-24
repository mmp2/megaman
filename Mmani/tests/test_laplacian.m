% generates the test data used by test_laplacian.py
%
% 

addpath /mnt/costila/mmp/research/spectral/dominique-epsilon/EpsilonDemo

outfroot = 'testMmani_laplacian'
rad = 0.2;
lambda = 1.5; % renormalization exponent
n = 200;
seed  = 36;
rand( 'seed', seed )
xx1 = rand( 1, n );
xx2 = rand( 1, n );
xx3 = sin( 2*pi*xx1).*sqrt(xx2)

xx = [ xx1; xx2; xx3 ];

epps = rad*rad;
[ A, S ] = similarity( xx', epps )
norms = {'geometric', 'unnormalized', 'randomwalk', 'symmetricnormalized', 'renormalized' };
names = {'geom', 'unnorm', 'rw', 'symnorm', 'reno1_5' };

for ii = 1:lenght( norms );
    opts.lapType = norms{ ii }
    [ L, phi, lam, flag ] = laplacian( A, 2, epps, opts );
    eval( [ 'L' names{ ii } '=L;']);
    eval( [ 'phi' names{ ii } '=phi;']);
    eval( [ 'lam' names{ ii } '=lam;']);

[G, VV, LL, Ginv ] = rmetric( Lgeom, phigeom, 2, 0 );

outfname = [ outfroot '_rad' num2str_(rad) '_lam' num2str_(lambda)
'_n' num2str( n ) '.mat' ]

save( outfname )
